#include "ggml-cuda.h"
#include "ggml.h"
#include "ggml-backend-impl.h"
#include "pidcontroller.h"
#include "sparse_async.h"

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/alibi.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/dmmv.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <float.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>

// SparseInfer HyperParameter
#define N_LAYER 40
//__constant__ float coef[N_LAYER * 5];
//__constant__ float mean_s[N_LAYER * 4];
//__constant__ float scale_s[N_LAYER * 4];
//__constant__ float thresholds_s[N_LAYER];
__constant__ float proportions_s[N_LAYER * 3];

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    cudaGetDevice(&id);

    fprintf(stderr, "CUDA error: %s\n", msg);
    fprintf(stderr, "  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    fprintf(stderr, "  %s\n", stmt);
    // abort with GGML_ASSERT to get a stack trace
    GGML_ASSERT(!"CUDA error");
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

static ggml_cuda_device_info ggml_cuda_init() {
#ifdef __HIP_PLATFORM_AMD__
    // Workaround for a rocBLAS bug when using multiple graphics cards:
    // https://github.com/ROCmSoftwarePlatform/rocBLAS/issues/1346
    rocblas_initialize();
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.device_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_CUDA_MAX_DEVICES);

    int64_t total_vram = 0;
#if defined(GGML_CUDA_FORCE_MMQ)
    fprintf(stderr, "%s: GGML_CUDA_FORCE_MMQ:   yes\n", __func__);
#else
    fprintf(stderr, "%s: GGML_CUDA_FORCE_MMQ:   no\n", __func__);
#endif
#if defined(CUDA_USE_TENSOR_CORES)
    fprintf(stderr, "%s: CUDA_USE_TENSOR_CORES: yes\n", __func__);
#else
    fprintf(stderr, "%s: CUDA_USE_TENSOR_CORES: no\n", __func__);
#endif
    fprintf(stderr, "%s: found %d " GGML_CUDA_NAME " devices:\n", __func__, info.device_count);
    for (int id = 0; id < info.device_count; ++id) {
        int device_vmm = 0;

#if !defined(GGML_USE_HIPBLAS)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // !defined(GGML_USE_HIPBLAS)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        fprintf(stderr, "  Device %d: %s, compute capability %d.%d, VMM: %s\n", id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no");

        info.default_tensor_split[id] = total_vram;
        total_vram += prop.totalGlobalMem;

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
        info.devices[id].cc = 100*prop.major + 10*prop.minor + CC_OFFSET_AMD;
#else
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
        info.devices[id].smpb = prop.sharedMemPerBlock;
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_cuda_pool_leg(int device) :
        device(device) {
    }

    ~ggml_cuda_pool_leg() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaMalloc((void **) &ptr, look_ahead_size));
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        fprintf(stderr, "%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                (uint32_t)(max_size/1024/1024), (uint32_t)(pool_size/1024/1024), (uint32_t)(size/1024/1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        fprintf(stderr, "WARNING: cuda buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
    }
};

// pool with virtual memory
#if !defined(GGML_USE_HIPBLAS)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CU_CHECK(cuMemMap(pool_addr + pool_size, reserve_size, 0, handle, 0));

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // set access
            CUmemAccessDesc access = {};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = device;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            CU_CHECK(cuMemSetAccess(pool_addr + pool_size, reserve_size, &access, 1));

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) (pool_addr + pool_used);
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) (pool_addr + pool_used));
    }
};
#endif // !defined(GGML_USE_HIPBLAS)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int device) {
#if !defined(GGML_USE_HIPBLAS)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device));
}

// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

GGML_CALL static const char * ggml_backend_cuda_buffer_get_name(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->name.c_str();
}

GGML_CALL static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == ggml_backend_cuda_buffer_get_name;
}

GGML_CALL static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    delete ctx;
}

GGML_CALL static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

GGML_CALL static void ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return;
    }

    if (ggml_is_quantized(tensor->type)) {
        // initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size && tensor->view_src == nullptr) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
        }
    }
}

GGML_CALL static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *)tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

GGML_CALL static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *)tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

GGML_CALL static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        if (src_ctx->device == dst_ctx->device) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_ctx->device, src->data, src_ctx->device, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(ctx->dev_ptr, value, buffer->size));
    CUDA_CHECK(cudaDeviceSynchronize());
}

static ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .get_name        = */ ggml_backend_cuda_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

GGML_CALL static const char * ggml_backend_cuda_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    size = std::max(size, (size_t)1); // cudaMalloc returns null for size 0

    void * dev_ptr;
    cudaError_t err = cudaMalloc(&dev_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size/1024.0/1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

GGML_CALL static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

GGML_CALL static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    size_t size = ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    GGML_UNUSED(buft);
}

GGML_CALL static bool ggml_backend_cuda_buffer_type_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    if (!ggml_backend_is_cuda(backend)) {
        return false;
    }

    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return buft_ctx->device == cuda_ctx->device;
}

static ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .supports_backend = */ ggml_backend_cuda_buffer_type_supports_backend,
    /* .is_host          = */ NULL,
};

GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < GGML_CUDA_MAX_DEVICES; i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// cuda split buffer

static int64_t get_row_rounding(ggml_type type, const std::array<float, GGML_CUDA_MAX_DEVICES> & tensor_split) {
    int64_t min_compute_capability = INT_MAX;
    int64_t max_compute_capability = INT_MIN;
    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        if (tensor_split[id] < (id + 1 < ggml_backend_cuda_get_device_count() ? tensor_split[id + 1] : 1.0f)) {
            if (min_compute_capability > ggml_cuda_info().devices[id].cc) {
                min_compute_capability = ggml_cuda_info().devices[id].cc;
            }
            if (max_compute_capability < ggml_cuda_info().devices[id].cc) {
                max_compute_capability = ggml_cuda_info().devices[id].cc;
            }
        }
    }

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    switch(type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return max_compute_capability >= CC_RDNA2 ? 128 : 64;
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
            return 1;
        case GGML_TYPE_Q2_K:
            return max_compute_capability >= CC_RDNA2 ? 128 : 32;
        case GGML_TYPE_Q3_K:
            return min_compute_capability < CC_RDNA2 ? 128 : 64;
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_S:
            return max_compute_capability >= CC_RDNA2 ? 128 : 64;
        default:
            GGML_ASSERT(false);
    }
#else
    switch(type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            return max_compute_capability >= CC_VOLTA ? 128 : 64;
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return 64;
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
            return 1;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_S:
            return max_compute_capability >= CC_VOLTA ? 128 : 64;
        case GGML_TYPE_Q6_K:
            return 64;
        default:
            GGML_ASSERT(false);
    }
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
}

static void get_row_split(int64_t * row_low, int64_t * row_high, const ggml_tensor * tensor, const std::array<float, GGML_CUDA_MAX_DEVICES> & tensor_split, int id) {
    const int64_t nrows = ggml_nrows(tensor);
    const int64_t rounding = get_row_rounding(tensor->type, tensor_split);

    *row_low = id == 0 ? 0 : nrows*tensor_split[id];
    *row_low -= *row_low % rounding;

    if (id == ggml_backend_cuda_get_device_count() - 1) {
        *row_high = nrows;
    } else {
        *row_high = nrows*tensor_split[id + 1];
        *row_high -= *row_high % rounding;
    }
}

static size_t ggml_nbytes_split(const struct ggml_tensor * tensor, int nrows_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return nrows_split*ggml_row_size(tensor->type, tensor->ne[0]);
}

struct ggml_backend_cuda_split_buffer_type_context {
    std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split;
};

struct ggml_backend_cuda_split_buffer_context {
    ~ggml_backend_cuda_split_buffer_context() {
        for (ggml_tensor_extra_gpu * extra : tensor_extras) {
            for (int id = 0; id < GGML_CUDA_MAX_DEVICES; ++id) {
                for (int64_t is = 0; is < GGML_CUDA_MAX_STREAMS; ++is) {
                    if (extra->events[id][is] != nullptr) {
                        CUDA_CHECK(cudaEventDestroy(extra->events[id][is]));
                    }
                }
                if (extra->data_device[id] != nullptr) {
                    CUDA_CHECK(cudaFree(extra->data_device[id]));
                }
            }
            delete extra;
        }
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
};

GGML_CALL static const char * ggml_backend_cuda_split_buffer_get_name(ggml_backend_buffer_t buffer) {
    return GGML_CUDA_NAME "_Split";

    GGML_UNUSED(buffer);
}

static bool ggml_backend_buffer_is_cuda_split(ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == ggml_backend_cuda_split_buffer_get_name;
    GGML_UNUSED(ggml_backend_buffer_is_cuda_split); // only used in debug builds currently, avoid unused function warning in release builds
}

GGML_CALL static void ggml_backend_cuda_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_split_buffer_context * ctx = (ggml_backend_cuda_split_buffer_context *)buffer->context;
    delete ctx;
}

GGML_CALL static void * ggml_backend_cuda_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    return (void *)0x1000;

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_cuda_split_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(tensor->view_src == nullptr); // views of split tensors are not supported

    ggml_backend_cuda_split_buffer_context * ctx = (ggml_backend_cuda_split_buffer_context *)buffer->context;
    ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};
    ctx->tensor_extras.push_back(extra);

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        // FIXME: do not crash if cudaMalloc fails
        // currently, init_tensor cannot fail, it needs to be fixed in ggml-backend first
        ggml_cuda_set_device(id);
        char * buf;
        CUDA_CHECK(cudaMalloc(&buf, size));

        // set padding to 0 to avoid possible NaN values
        if (size > original_size) {
            CUDA_CHECK(cudaMemset(buf + original_size, 0, size - original_size));
        }

        extra->data_device[id] = buf;

        for (int64_t is = 0; is < GGML_CUDA_MAX_STREAMS; ++is) {
            CUDA_CHECK(cudaEventCreateWithFlags(&extra->events[id][is], cudaEventDisableTiming));
        }
    }
    tensor->extra = extra;
}

GGML_CALL static void ggml_backend_cuda_split_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        const char * buf_host = (const char *)data + offset_split;
        CUDA_CHECK(cudaMemcpyAsync(extra->data_device[id], buf_host, original_size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }
}

GGML_CALL static void ggml_backend_cuda_split_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        char * buf_host = (char *)data + offset_split;
        CUDA_CHECK(cudaMemcpyAsync(buf_host, extra->data_device[id], original_size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }
}

GGML_CALL static void ggml_backend_cuda_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static struct ggml_backend_buffer_i ggml_backend_cuda_split_buffer_interface = {
    /* .get_name        = */ ggml_backend_cuda_split_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_cuda_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_split_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_split_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_cuda_split_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda split buffer type

GGML_CALL static const char * ggml_backend_cuda_split_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Split";

    GGML_UNUSED(buft);
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_cuda_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_cuda_split_buffer_context * ctx = new ggml_backend_cuda_split_buffer_context();

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_split_buffer_interface, ctx, size);
}

GGML_CALL static size_t ggml_backend_cuda_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

GGML_CALL static size_t ggml_backend_cuda_split_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_cuda_split_buffer_type_context * ctx = (ggml_backend_cuda_split_buffer_type_context *)buft->context;

    size_t total_size = 0;

    const int64_t ne0 = tensor->ne[0];

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        total_size += ggml_nbytes_split(tensor, nrows_split);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            total_size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return total_size;
}

GGML_CALL static bool ggml_backend_cuda_split_buffer_type_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    return ggml_backend_is_cuda(backend);

    GGML_UNUSED(buft);
}

GGML_CALL static bool ggml_backend_cuda_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_cuda_split_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_split_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_split_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_split_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_split_buffer_type_get_alloc_size,
    /* .supports_backend = */ ggml_backend_cuda_split_buffer_type_supports_backend,
    /* .is_host          = */ ggml_backend_cuda_split_buffer_type_is_host,
};

GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(const float * tensor_split) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::map<std::array<float, GGML_CUDA_MAX_DEVICES>, struct ggml_backend_buffer_type> buft_map;

    std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split_arr = {};

    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + GGML_CUDA_MAX_DEVICES, [](float x) { return x == 0.0f; });
    if (all_zero) {
        tensor_split_arr = ggml_cuda_info().default_tensor_split;
    } else {
        float split_sum = 0.0f;
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); ++i) {
            tensor_split_arr[i] = split_sum;
            split_sum += tensor_split[i];
        }
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); ++i) {
            tensor_split_arr[i] /= split_sum;
        }
    }

    auto it = buft_map.find(tensor_split_arr);
    if (it != buft_map.end()) {
        return &it->second;
    }

    struct ggml_backend_buffer_type buft {
        /* .iface   = */ ggml_backend_cuda_split_buffer_type_interface,
        /* .context = */ new ggml_backend_cuda_split_buffer_type_context{tensor_split_arr},
    };

    auto result = buft_map.emplace(tensor_split_arr, buft);
    return &result.first->second;
}

// host buffer type

GGML_CALL static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

GGML_CALL static const char * ggml_backend_cuda_host_buffer_name(ggml_backend_buffer_t buffer) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // clear the error
        cudaGetLastError();
        fprintf(stderr, "%s: warning: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
            size/1024.0/1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.get_name = ggml_backend_cuda_host_buffer_name;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .supports_backend = */ ggml_backend_cpu_buffer_type()->iface.supports_backend,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

#ifndef GGML_CUDA_PEER_MAX_BATCH_SIZE
#define GGML_CUDA_PEER_MAX_BATCH_SIZE 128
#endif // GGML_CUDA_PEER_MAX_BATCH_SIZE

#define MUL_MAT_SRC1_COL_STRIDE 128

static __global__ void mul_mat_p021_f16_f32(
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nchannels_x, const int nchannels_y) {

    const half * x = (const half *) vx;

    const int row_x = blockDim.y*blockIdx.y + threadIdx.y;
    const int channel = blockDim.z*blockIdx.z + threadIdx.z;
    const int channel_x = channel / (nchannels_y / nchannels_x);

    const int nrows_y = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst = row_x;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x; col_x0 += blockDim.x) {
        const int col_x = col_x0 + threadIdx.x;

        if (col_x >= ncols_x) {
            break;
        }

        // x is transposed and permuted
        const int ix = row_x*nchannels_x*ncols_x + channel_x*ncols_x + col_x;
        const float xi = __half2float(x[ix]);

        const int row_y = col_x;

        // y is not transposed but permuted
        const int iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    const int idst = channel*nrows_dst + row_dst;

    // sum up partial sums and write back result
    tmp = warp_reduce_sum(tmp);

    if (threadIdx.x == 0) {
        dst[idst] = tmp;
    }
}

static __global__ void mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst, const int ncols_x, const int nrows_x,
    const int row_stride_x, const int channel_stride_x, const int channel_x_divisor) {

    const half * x = (const half *) vx;

    const int row_x     = blockDim.y*blockIdx.y + threadIdx.y;
    const int channel   = blockDim.z*blockIdx.z + threadIdx.z;
    const int channel_x = channel / channel_x_divisor;

    const int nrows_y   = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst   = row_x;

    const int idst = channel*nrows_dst + row_dst;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x; col_x0 += blockDim.x) {
        const int col_x = col_x0 + threadIdx.x;

        if (col_x >= ncols_x) {
            break;
        }

        const int row_y = col_x;

        const int ix = channel_x*channel_stride_x + row_x*row_stride_x + col_x;
        const int iy = channel*nrows_y + row_y;

        const float xi = __half2float(x[ix]);

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
    tmp = warp_reduce_sum(tmp);

    if (threadIdx.x == 0) {
        dst[idst] = tmp;
    }
}

static void ggml_mul_mat_p021_f16_f32_cuda(
    const void * vx, const float * y, float * dst, const int ncols_x, const int nrows_x,
    const int nchannels_x, const int nchannels_y, cudaStream_t stream) {

    const dim3 block_nums(1, nrows_x, nchannels_y);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    mul_mat_p021_f16_f32<<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols_x, nrows_x, nchannels_x, nchannels_y);
}

static void ggml_mul_mat_vec_nc_f16_f32_cuda(
    const void * vx, const float * y, float * dst, const int ncols_x, const int nrows_x, const int row_stride_x,
    const int nchannels_x, const int nchannels_y, const int channel_stride_x, cudaStream_t stream) {

    const dim3 block_nums(1, nrows_x, nchannels_y);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    mul_mat_vec_nc_f16_f32<<<block_nums, block_dims, 0, stream>>>
        (vx, y, dst, ncols_x, nrows_x, row_stride_x, channel_stride_x, nchannels_y/nchannels_x);
}

static cudaError_t ggml_cuda_cpy_tensor_2d(
    void * dst, const struct ggml_tensor * src, int64_t i3, int64_t i2, int64_t i1_low, int64_t i1_high, cudaStream_t stream) {

    GGML_ASSERT(ggml_backend_buffer_is_cuda(src->buffer));
    char * src_ptr = (char *) src->data;
    char * dst_ptr = (char *) dst;

    const int64_t ne0 = src->ne[0];
    const int64_t nb0 = src->nb[0];
    const int64_t nb1 = src->nb[1];
    const int64_t nb2 = src->nb[2];
    const int64_t nb3 = src->nb[3];
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t bs = ggml_blck_size(type);
    int64_t i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low*nb1 + i2*nb2 + i3*nb3;
    if (nb0 == ts && nb1 == ts*ne0/bs) {
        return cudaMemcpyAsync(dst_ptr, x, i1_diff*nb1, cudaMemcpyDeviceToDevice, stream);
    } else if (nb0 == ts) {
        return cudaMemcpy2DAsync(dst_ptr, ts*ne0/bs, x, nb1, ts*ne0/bs, i1_diff, cudaMemcpyDeviceToDevice, stream);
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            cudaError_t r = cudaMemcpy2DAsync(rd, ts/bs, rx, nb0, ts/bs, ne0, cudaMemcpyDeviceToDevice, stream);
            if (r != cudaSuccess) {
                return r;
            }
        }
        return cudaSuccess;
    }
}

static void ggml_cuda_op_mul_mat_cublas(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int64_t ldc = id == ctx.device ? ne0 : row_diff;

    const int compute_capability = ggml_cuda_info().devices[id].cc;

    if (compute_capability >= CC_VOLTA && (src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && ggml_is_contiguous(src0) && row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
        // convert src0 and src1 to fp16, multiply as fp16, convert dst to fp32
        ggml_cuda_pool_alloc<half> src0_as_f16(ctx.pool(id));
        if (src0->type != GGML_TYPE_F16) {
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src0->type);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            size_t ne = row_diff*ne00;
            src0_as_f16.alloc(ne);
            to_fp16_cuda(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        const half * src0_ptr = src0->type == GGML_TYPE_F16 ? (const half *) src0_dd_i : src0_as_f16.get();

        ggml_cuda_pool_alloc<half> src1_as_f16(ctx.pool(id));
        if (src1->type != GGML_TYPE_F16) {
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src1->type);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            to_fp16_cuda(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        const half * src1_ptr = src1->type == GGML_TYPE_F16 ? (const half *) src1_ddf_i : src1_as_f16.get();
        ggml_cuda_pool_alloc<half> dst_f16(ctx.pool(id), row_diff*src1_ncols);

        const half alpha_f16 = 1.0f;
        const half beta_f16 = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha_f16, src0_ptr,       CUDA_R_16F, ne00,
                                src1_ptr,       CUDA_R_16F, ne10,
                    &beta_f16,   dst_f16.get(), CUDA_R_16F, ldc,
                    CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
        to_fp32_cuda(dst_f16.get(), dst_dd_i, row_diff*src1_ncols, stream);
    } else {
        ggml_cuda_pool_alloc<float> src0_ddq_as_f32(ctx.pool(id));
        ggml_cuda_pool_alloc<float> src1_ddq_as_f32(ctx.pool(id));

        if (src0->type != GGML_TYPE_F32) {
            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src0->type);
            GGML_ASSERT(to_fp32_cuda != nullptr);
            src0_ddq_as_f32.alloc(row_diff*ne00);
            to_fp32_cuda(src0_dd_i, src0_ddq_as_f32.get(), row_diff*ne00, stream);
        }
        if (src1->type != GGML_TYPE_F32) {
            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src1->type);
            GGML_ASSERT(to_fp32_cuda != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            to_fp32_cuda(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols*ne10, stream);
        }

        const float * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (const float *) src0_dd_i : src0_ddq_as_f32.get();
        const float * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (const float *) src1_ddf_i : src1_ddq_as_f32.get();

        const float alpha = 1.0f;
        const float beta = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasSgemm(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha, src0_ddf_i,  ne00,
                            src1_ddf1_i, ne10,
                    &beta,  dst_dd_i,    ldc));
    }

    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_padded_row_size);
}

static void ggml_cuda_set_peer_access(const int n_tokens, int main_device) {
    static bool peer_access_enabled = false;

    const bool enable_peer_access = n_tokens <= GGML_CUDA_PEER_MAX_BATCH_SIZE;

    if (peer_access_enabled == enable_peer_access) {
        return;
    }

#ifdef NDEBUG
    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        ggml_cuda_set_device(id);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        ggml_cuda_set_device(id);

        for (int id_other = 0; id_other < ggml_backend_cuda_get_device_count(); ++id_other) {
            if (id == id_other) {
                continue;
            }
            if (id != main_device && id_other != main_device) {
                continue;
            }

            int can_access_peer;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
            if (can_access_peer) {
                if (enable_peer_access) {
                    cudaError_t err = cudaDeviceEnablePeerAccess(id_other, 0);
                    if (err != cudaErrorPeerAccessAlreadyEnabled) {
                        CUDA_CHECK(err);
                    }
                } else {
                    cudaError_t err = cudaDeviceDisablePeerAccess(id_other);
                    if (err != cudaErrorPeerAccessNotEnabled) {
                        CUDA_CHECK(err);
                    }
                }
            }
        }
    }

    ggml_cuda_set_device(main_device);
#endif // NDEBUG

    peer_access_enabled = enable_peer_access;

    GGML_UNUSED(main_device);
}

static void ggml_cuda_op_mul_mat(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, ggml_cuda_op_mul_mat_t op,
    const bool convert_src1_to_q8_1) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int64_t nb2 = dst->nb[2];
    const int64_t nb3 = dst->nb[3];

    GGML_ASSERT(ggml_backend_buffer_is_cuda(dst->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_cuda(src1->buffer));
    ggml_backend_cuda_buffer_context * src1_ctx = (ggml_backend_cuda_buffer_context *) src1->buffer->context;
    ggml_backend_cuda_buffer_context * dst_ctx  = (ggml_backend_cuda_buffer_context *) dst->buffer->context;

    GGML_ASSERT(src1->type == GGML_TYPE_F32 || (src1->ne[2] == 1 && src1->ne[3] == 1));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    const int64_t i02_divisor = ne12 / ne02;

    const size_t src0_ts = ggml_type_size(src0->type);
    const size_t src0_bs = ggml_blck_size(src0->type);
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;

    const bool src0_is_contiguous = ggml_is_contiguous(src0);
    const bool src1_is_contiguous = ggml_is_contiguous(src1);

    const int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const bool split = ggml_backend_buffer_is_cuda_split(src0->buffer);
    GGML_ASSERT(!(split && ne02 > 1));
    GGML_ASSERT(!(split && ne03 > 1));
    GGML_ASSERT(!(split && ne02 < ne12));

    ggml_tensor_extra_gpu * src0_extra = split ? (ggml_tensor_extra_gpu *) src0->extra : nullptr;


    std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split;
    if (split) {
        ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *) src0->buffer->buft->context;
        tensor_split = buft_ctx->tensor_split;
    }

    struct dev_data {
        ggml_cuda_pool_alloc<char>  src0_dd_alloc;
        ggml_cuda_pool_alloc<float> src1_ddf_alloc;
        ggml_cuda_pool_alloc<char>  src1_ddq_alloc;
        ggml_cuda_pool_alloc<float>   dst_dd_alloc;

        char  *  src0_dd = nullptr;
        float * src1_ddf = nullptr; // float
        char  * src1_ddq = nullptr; // q8_1
        float *   dst_dd = nullptr;

        int64_t  row_low;
        int64_t row_high;
    };

    dev_data dev[GGML_CUDA_MAX_DEVICES];

    int used_devices = 0;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        // by default, use all rows
        dev[id].row_low  = 0;
        dev[id].row_high = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        if (split) {
            const int64_t rounding = get_row_rounding(src0->type, tensor_split);

            if (id != 0) {
                dev[id].row_low  = ne01*tensor_split[id];
                if (dev[id].row_low < ne01) {
                    dev[id].row_low -= dev[id].row_low % rounding;
                }
            }

            if (id != ggml_backend_cuda_get_device_count() - 1) {
                dev[id].row_high  = ne01*tensor_split[id + 1];
                if (dev[id].row_high < ne01) {
                    dev[id].row_high -= dev[id].row_high % rounding;
                }
            }
        }
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        if ((!split && id != ctx.device) || dev[id].row_low == dev[id].row_high) {
            continue;
        }

        used_devices++;

        const bool src1_on_device = id == src1_ctx->device;
        const bool  dst_on_device = id == dst_ctx->device;

        ggml_cuda_set_device(id);
        cudaStream_t stream = ctx.stream(id, 0);

        if (src0_is_contiguous) {
            dev[id].src0_dd = split ? (char *) src0_extra->data_device[id] : (char *) src0->data;
        } else {
            dev[id].src0_dd = dev[id].src0_dd_alloc.alloc(ctx.pool(id), ggml_nbytes(src0));
        }

        if (src1_on_device && src1_is_contiguous) {
            dev[id].src1_ddf = (float *) src1->data;
        } else {
            dev[id].src1_ddf = dev[id].src1_ddf_alloc.alloc(ctx.pool(id), ggml_nelements(src1));
        }

        if (convert_src1_to_q8_1) {
            dev[id].src1_ddq = dev[id].src1_ddq_alloc.alloc(ctx.pool(id), nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs);

            if (src1_on_device && src1_is_contiguous) {
                quantize_row_q8_1_cuda(dev[id].src1_ddf, dev[id].src1_ddq, ne10, nrows1, src1_padded_col_size, stream);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        if (dst_on_device) {
            dev[id].dst_dd = (float *) dst->data;
        } else {
            const size_t size_dst_ddf = split ? (dev[id].row_high - dev[id].row_low)*ne1 : ggml_nelements(dst);
            dev[id].dst_dd = dev[id].dst_dd_alloc.alloc(ctx.pool(id), size_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    if (split && used_devices > 1) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaEventRecord(src0_extra->events[ctx.device][0], ctx.stream()));
    }

    const int64_t src1_col_stride = split && used_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
    for (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        const int64_t is = split ? (src1_col_0/src1_col_stride) % GGML_CUDA_MAX_STREAMS : 0;
        const int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;

        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            if ((!split && id != ctx.device) || dev[id].row_low == dev[id].row_high) {
                continue;
            }

            const bool src1_on_device = id == src1_ctx->device;
            const bool  dst_on_device = id == dst_ctx->device;
            const int64_t row_diff = dev[id].row_high - dev[id].row_low;

            ggml_cuda_set_device(id);
            cudaStream_t stream = ctx.stream(id, is);

            // wait for main GPU data if necessary
            if (split && (id != ctx.device || is != 0)) {
                CUDA_CHECK(cudaStreamWaitEvent(stream, src0_extra->events[ctx.device][0], 0));
            }

            for (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                const int64_t i03 = i0 / ne12;
                const int64_t i02 = i0 % ne12;

                const size_t src1_ddq_i_offset = (i0*ne11 + src1_col_0) * src1_padded_col_size*q8_1_ts/q8_1_bs;

                // for split tensors the data begins at i0 == i0_offset_low
                char  *  src0_dd_i =  dev[id].src0_dd + (i0/i02_divisor) * (ne01*ne00*src0_ts)/src0_bs;
                float * src1_ddf_i = dev[id].src1_ddf + (i0*ne11 + src1_col_0) * ne10;
                char  * src1_ddq_i = dev[id].src1_ddq +  src1_ddq_i_offset;
                float *   dst_dd_i =   dev[id].dst_dd + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (id == ctx.device) {
                    dst_dd_i += dev[id].row_low; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (src1_is_contiguous) {
                    if (id != ctx.device) {
                        if (convert_src1_to_q8_1) {
                            char * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                            CUDA_CHECK(cudaMemcpyPeerAsync(src1_ddq_i, id, src1_ddq_i_source, ctx.device,
                                                            src1_ncols*src1_padded_col_size*q8_1_ts/q8_1_bs, stream));
                        } else {
                            float * src1_ddf_i_source = (float *) src1->data;
                            src1_ddf_i_source += (i0*ne11 + src1_col_0) * ne10;
                            CUDA_CHECK(cudaMemcpyPeerAsync(src1_ddf_i, id, src1_ddf_i_source, ctx.device,
                                                            src1_ncols*ne10*sizeof(float), stream));
                        }
                    }
                } else if (src1_on_device && !src1_is_contiguous) {
                    CUDA_CHECK(ggml_cuda_cpy_tensor_2d(
                                src1_ddf_i, src1, i03, i02, src1_col_0, src1_col_0+src1_ncols, stream));
                } else {
                    GGML_ASSERT(false);
                }

                if (convert_src1_to_q8_1 && !src1_is_contiguous) {
                    quantize_row_q8_1_cuda(src1_ddf_i, src1_ddq_i, ne10, src1_ncols, src1_padded_col_size, stream);
                    CUDA_CHECK(cudaGetLastError());
                }

                if (src1_col_0 == 0 && !src0_is_contiguous && i02 % i02_divisor == 0) {
                    CUDA_CHECK(ggml_cuda_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_divisor, dev[id].row_low, dev[id].row_high, stream));
                }

                // do the computation
                op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[id].row_low, dev[id].row_high, src1_ncols, src1_padded_col_size, stream);
                CUDA_CHECK(cudaGetLastError());

                // copy dst to host or other device if necessary
                if (!dst_on_device) {
                    void * dst_off_device = dst->data;
                    if (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0 + dev[id].row_low;
#if !defined(GGML_USE_HIPBLAS)
                        // cudaMemcpy2DAsync may fail with copies between vmm pools of different devices
                        cudaMemcpy3DPeerParms p = {};
                        p.dstDevice = ctx.device;
                        p.dstPtr = make_cudaPitchedPtr(dhf_dst_i, ne0*sizeof(float), row_diff, src1_ncols);
                        p.srcDevice = id;
                        p.srcPtr = make_cudaPitchedPtr(dst_dd_i, row_diff*sizeof(float), row_diff, src1_ncols);
                        p.extent = make_cudaExtent(row_diff*sizeof(float), src1_ncols, 1);
                        CUDA_CHECK(cudaMemcpy3DPeerAsync(&p, stream));
#else
                        // HIP does not support cudaMemcpy3DPeerAsync or vmm pools
                        CUDA_CHECK(cudaMemcpy2DAsync(dhf_dst_i, ne0*sizeof(float),
                                                        dst_dd_i, row_diff*sizeof(float),
                                                        row_diff*sizeof(float), src1_ncols,
                                                        cudaMemcpyDeviceToDevice, stream));
#endif
                    } else {
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0;
                        CUDA_CHECK(cudaMemcpyAsync(dhf_dst_i, dst_dd_i, src1_ncols*ne0*sizeof(float), cudaMemcpyDeviceToDevice, stream));
                    }
                }

                // add event for the main device to wait on until other device is done
                if (split && (id != ctx.device || is != 0)) {
                    CUDA_CHECK(cudaEventRecord(src0_extra->events[id][is], stream));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    if (split && ggml_backend_cuda_get_device_count() > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
        is_max = is_max <= GGML_CUDA_MAX_STREAMS ? is_max : GGML_CUDA_MAX_STREAMS;

        ggml_cuda_set_device(ctx.device);
        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            if (dev[id].row_low == dev[id].row_high) {
                continue;
            }
            for (int64_t is = 0; is < is_max; ++is) {
                CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), src0_extra->events[id][is], 0));
            }
        }
    }
}


static __device__ void convert_f16(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const half * x = (const half *) vx;

    // automatic half -> float type cast if dfloat == float
    v.x = x[ib + iqs + 0];
    v.y = x[ib + iqs + 1];
}

static __global__ void set_zero(float * dst, int k) {
    const int64_t col = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    dst[col] = 0.0f;

}

// TODO : Shared 메모리 없이 바로 atomicAdd,
// 1. Shared 메모리만 안쓰는 걸로 global Atomic Add를 하지 않는 버전 확인
// 2. 행으로 WARP을 늘려서 확인
// 3. 열로도 WARP를 적절히 늘려서 확인
// 행으로 WARP을 늘리거나, 열로 늘리거나 여러 가지 경우 테스트 ex (16, 1) (4, 4)
static __global__ void mul_mat_vec_transpose_test(const half * x, const float * y, float * dst, int nrows, int ncols) {
    __shared__ float partial_sum[128];

    if (threadIdx.y == 0) // 대표 WARP가 공유메모리 초기화
        partial_sum[threadIdx.x] = 0.0f;
    __syncthreads();

    const int64_t row = (int64_t)blockIdx.x * blockDim.y + threadIdx.y;

    if (y[row] == 0.0f && threadIdx.y != 0)
        return ;

    const int tid = threadIdx.x;

    for (int i=0; i<ncols; i += blockDim.x) {
        const int col = i + tid;
        atomicAdd(&partial_sum[tid], __half2float(x[row*ncols + col]) * y[row]);
        __syncthreads();
        if (threadIdx.y == 0) {
            atomicAdd(&dst[col], partial_sum[tid]);
            partial_sum[tid] = 0.0f;
        }
        __syncthreads();
    }
}

static __global__ void mul_mat_vec_transpose(const half * x, const float * y, float * dst, int nrows, int ncols) {
    const int64_t row = (int64_t)blockIdx.x;

    if (y[row] == 0.0f)
        return ;

    const int tid = threadIdx.x;

    for (int i=0; i<ncols; i += blockDim.x) {
        const int col = i + tid;
        atomicAdd(&dst[col], __half2float(x[row*ncols + col]) * y[row]);
        //dst_shared[col] = __half2float(x[row*ncols + col]) * y[row];
    }

    //__syncthreads();
    //for (int i=0; i<ncols; i += blockDim.x) {
    //    int col = i + tid;
    //    atomicAdd(&dst[col], dst_shared[col]);
    //}
}

static __global__ void mul_mat_vec(const half * x, const float * y, float * dst, int nrows, int ncols) {
    const int64_t row = (int64_t)blockIdx.y*blockDim.y + threadIdx.y;

    if (row >= nrows) {
        return;
    }

    const int tid = threadIdx.x;

    const int iter_stride = 2*GGML_CUDA_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter

    // Gate_proj
    float tmp = 0.0f;

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter*tid;
        const int ib = (row*ncols + col);
        const int iybs = col;

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            convert_f16(x, ib, j, v);

            tmp += v.x * y[iybs + j];
            tmp += v.y * y[iybs + j + 1];
        }
    }

    // sum up partial sums and write back result
    tmp = warp_reduce_sum(tmp);

    if (tid == 0) {
        dst[row] = tmp;
    }
}

static __global__ void initialize(int *dst) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dst[0] = 0;
        dst[1] = 0;
        dst[2] = 0;
        dst[3] = 0;
    }
}

static __global__ void sparse_error(const float * x, const int32_t * y, int * dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
       // true if value is positive (non-sparse)
       const bool actual_positive = x[idx] > 0;
       const bool predicted_positive = y[idx];

       // combine into 2-bit pattern:
       // actual_positive << 1 | predicted_positive
       const int result = (actual_positive << 1) | predicted_positive;
       
       // result = 0: (!actual_positive && !predicted_positive) -> TP (both sparse)
       // result = 1: (!actual_positive && predicted_positive)  -> FN (actual sparse, pred non-sparse)
       // result = 2: (actual_positive && !predicted_positive)  -> FP (actual non-sparse, pred sparse)
       // result = 3: (actual_positive && predicted_positive)   -> TN (both non-sparse)
       atomicAdd(&dst[((result == 0) ? 0 : 
                      (result == 1) ? 1 : 
                      (result == 2) ? 2 : 3)], 1);
    }
}

static __global__ void mul_mat_vec_sparse_gate(const half * x, const float * y, float * dst, int *sparse_idx, int nrows, int ncols) {
    const int64_t row = (int64_t)blockIdx.y*blockDim.y + threadIdx.y;

    if (row >= nrows) {
        return;
    }

    const int tid = threadIdx.x;

    // Sparse 예측기가 0으로 예측한 것
    if (sparse_idx[row] == 0) {
        if (tid == 0) {
            dst[row] = 0.0f;
        }
        return ;
    }

    const int iter_stride = 2*GGML_CUDA_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter

    // Gate_proj
    float tmp = 0.0f;

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter*tid;
        const int ib = (row*ncols + col);
        const int iybs = col;

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            convert_f16(x, ib, j, v);

            tmp += v.x * y[iybs + j];
            tmp += v.y * y[iybs + j + 1];
        }
    }

    // sum up partial sums and write back result
    tmp = warp_reduce_sum(tmp);

    if (tid == 0) {
        dst[row] = tmp;
        if (tmp < 0.0f) {
            sparse_idx[row] = 0;
        }
    }
}

static __global__ void dequantize_mul_mat_axpy_sparse(const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst, const int ncols, const int nrows) {
    const int row = blockIdx.y*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    //h3가 0이면 리턴
    if (y[row] == 0)
        return;

    extern __shared__ float shared_dst[]; // TODO:dynamic

    const int iter_stride = 2*blockDim.x;
    const int vals_per_iter = iter_stride / blockDim.x; // num quantized vals per thread and i iter

// partial sum for each thread
    float tmp = 0.0f;
    for (int i = 0; i < ncols; i += blockDim.x) {
        shared_dst[i+tid] = 0;
    }
    __syncthreads();

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter*tid;
        const int ib = (row*ncols + col);
        const int iybs = col;

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            convert_f16(vx, ib, j, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
            tmp = v.x * y[row];
            shared_dst[iybs + j + 0] = tmp;
            tmp = v.y * y[row];
            shared_dst[iybs + j + 1] = tmp;

        }
    }
    __syncthreads();

    for (int i = 0; i < ncols; i += blockDim.x) {
        atomicAdd(&dst[i+tid], shared_dst[i+tid]);
    }
}

static __global__ void mul_mat_vec_sparse_up(const half * x, const float * y, float * dst, int *sparse_idx, int nrows, int ncols) {
    const int64_t row = (int64_t)blockIdx.y*blockDim.y + threadIdx.y;

    if (row >= nrows) {
        return;
    }

    const int tid = threadIdx.x;

    // Sparse 예측기가 0으로 예측한 것, 이전 h가 0인 것 return
    if (sparse_idx[row] == 0) {
        if (tid == 0) {
            dst[row] = 0.0f;
        }
        return ;
    }

    const int iter_stride = 2*GGML_CUDA_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter

    // Up_proj
    float tmp = 0.0f;

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter*tid;
        const int ib = (row*ncols + col);
        const int iybs = col;

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            convert_f16(x, ib, j, v);

            tmp += v.x * y[iybs + j];
            tmp += v.y * y[iybs + j + 1];
        }
    }

    // sum up partial sums and write back result
    tmp = warp_reduce_sum(tmp);

    if (tid == 0) {
        dst[row] = tmp;
    }
}

static __global__ void mul_mat_vec_q1(const int32_t * x, const int32_t * y, const int32_t * x_size,
                    const int32_t * y_size, const int32_t * y_outlier, int * dst, int nrows, int ncols, float scale_factor, int il) {
    const int64_t row = (int64_t)blockIdx.x * blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;
    int big_dif = 0;
    int normal_dif = 0;
    int small_dif = 0;
    int outlier_dif = 0;
    for (int i=0; i<ncols/32; i++) {
        int col = i * 32 + tid;
        int32_t bigBit = x_size[row*ncols + col] & y_size[col];
        int32_t normalBit = x_size[row*ncols + col] ^ y_size[col];
        int32_t smallBit = ~(bigBit | normalBit);
        int32_t negativeBit = (x[row*ncols + col] ^ y[col]);

        int big_count = __popc(bigBit);
        int big_negative_count = __popc(bigBit & negativeBit);
        big_dif += 2 * big_negative_count - big_count;

        int normal_count = __popc(normalBit);
        int normal_negative_count = __popc(normalBit & negativeBit);
        normal_dif += 2 * normal_negative_count - normal_count;

        int small_count = 32 - big_count - normal_count;
        int small_negative_count = __popc(smallBit & negativeBit);
        small_dif += 2 * small_negative_count - small_count;

        int outlier_negative_count = __popc(y_outlier[col] & negativeBit);
        int outlier_count = __popc(y_outlier[col]);
        outlier_dif += 2 * outlier_negative_count - outlier_count;
    }
    big_dif = warp_reduce_sum(big_dif);
    normal_dif = warp_reduce_sum(normal_dif);
    small_dif = warp_reduce_sum(small_dif);
    outlier_dif = warp_reduce_sum(outlier_dif);
    if (tid == 0) {
        float t = outlier_dif * proportions_s[il * 3 + 0] + big_dif * proportions_s[il * 3 + 1] + normal_dif * proportions_s[il * 3 + 2] + small_dif;
        if (t > scale_factor)
            dst[row] = 0;
        else
            dst[row] = 1;
    }
}
// BigN * s1 + NorN * s2 > BigP * s1 + NorP * s2
// Big = BigN + BigP
// BigN - bigP = 2 * BigN - Big

static __global__ void quantize_Q1(const float * x, int32_t * dst) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    int32_t tmp = 0;
    for (int k=0; k<32; k++) {
        if (x[i*32 + k] < 0.0f)
            tmp |= ((unsigned int)1 << k);
    }
    dst[i] = tmp;
}

static __device__ __forceinline__ unsigned int warp_reduce_sum_int32(unsigned int x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

static __global__ void reduce_sum_mean(const float * x, float * dst, int ncols, int smemD) {
    extern __shared__ float smem[];

    // boundary check
    unsigned int tid = threadIdx.x;
    unsigned int idx = 4 * blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= ncols) return;

    // calculate lane index and warp index
    int lane_idx = threadIdx.x % warpSize;
    int warp_idx = threadIdx.x / warpSize;

    // Mean
    // block-wide warp reduce + unrolling
    float local_sum = warp_reduce_sum(x[idx]);
    local_sum += warp_reduce_sum(x[idx + blockDim.x]);
    local_sum += warp_reduce_sum(x[idx + blockDim.x * 2]);
    local_sum += warp_reduce_sum(x[idx + blockDim.x * 3]);

    // save warp sum to shared memory
    if (lane_idx == 0)
        smem[warp_idx] = local_sum;
    __syncthreads();

    // last warp reduce
    if (warp_idx == 0)
        local_sum = (threadIdx.x < smemD) ? smem[lane_idx] : 0;

    if (warp_idx == 0)
        local_sum = warp_reduce_sum(local_sum);

    if (tid == 0)
        dst[blockIdx.x] = local_sum;

}

static __global__ void reduce_sum_variance(const float * x, float * dst, int ncols, int smemD) {
    extern __shared__ float smem[];

    // boundary check
    unsigned int tid = threadIdx.x;
    unsigned int idx = 4 * blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= ncols) return;

    // calculate lane index and warp index
    int lane_idx = threadIdx.x % warpSize;
    int warp_idx = threadIdx.x / warpSize;

    float mean = 0.0f;
    mean = (lane_idx < blockDim.x) ? dst[lane_idx] : 0.0f;
    mean = warp_reduce_sum(mean);
    mean /= ncols;

    // block-wide warp reduce + unrolling
    float local_sum = warp_reduce_sum((x[idx] - mean) * (x[idx] - mean));
    local_sum += warp_reduce_sum((x[idx + blockDim.x] - mean) * (x[idx + blockDim.x] - mean));
    local_sum += warp_reduce_sum((x[idx + blockDim.x * 2] - mean) * (x[idx + blockDim.x * 2] - mean));
    local_sum += warp_reduce_sum((x[idx + blockDim.x * 3] - mean) * (x[idx + blockDim.x * 3] - mean));

    // save warp sum to shared memory
    if (lane_idx == 0)
        smem[warp_idx] = local_sum;
    __syncthreads();

    // last warp reduce
    if (warp_idx == 0)
        local_sum = (lane_idx < smemD) ? smem[lane_idx] : 0;

    if (warp_idx == 0)
        local_sum = warp_reduce_sum(local_sum);

    if (tid == 0)
        dst[32 + blockIdx.x] = local_sum;
}

static __global__ void size_quantize_Q1(const float * x, const float * tmp, int32_t * x_outlier, void * dst, int ncols, int smemD, double std_multiplier) {
    int32_t *dst_p = (int32_t *) dst;

    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= ncols) return;

    // calculate lane index and warp index
    int lane_idx = threadIdx.x % warpSize;
    int warp_idx = threadIdx.x / warpSize;

    float mean = 0.0f;
    mean = (lane_idx < smemD) ? tmp[lane_idx] : 0.0f;
    mean = warp_reduce_sum(mean);
    mean /= ncols;

    float variance = 0.0f;
    variance = (lane_idx < smemD) ? tmp[32 + lane_idx] : 0.0f;
    variance = warp_reduce_sum(variance);
    variance = sqrtf(variance / ncols);

    int32_t big_index = 0;
    int32_t outlier_index = 0;
    if (fabsf(x[idx] - mean) > 0.6745 * variance)
        big_index |= ((unsigned int)1 << lane_idx);
    if (fabsf(x[idx] - mean) > std_multiplier * variance)
        outlier_index |= ((unsigned int)1 << lane_idx);
    big_index = warp_reduce_sum_int32(big_index);
    outlier_index = warp_reduce_sum_int32(outlier_index);
    if (lane_idx == 0) {
        dst_p[idx / 32] = big_index;
        x_outlier[idx / 32] = outlier_index;
    }
}

static __global__ void outlier_partial_sum(const float * x, const float * tmp, const half * gate, float *x_outlier, int ncols, int nrows, int varD) {
    unsigned int row = blockIdx.x;
    if (row >= nrows) return;

    // calculate lane index and warp index
    int lane_idx = threadIdx.x % warpSize;
    int warp_idx = threadIdx.x / warpSize;

    float mean = 0.0f;
    mean = (lane_idx < varD) ? tmp[lane_idx] : 0.0f;
    mean = warp_reduce_sum(mean);
    mean /= ncols;

    float variance = 0.0f;
    variance = (lane_idx < varD) ? tmp[32 + lane_idx] : 0.0f;
    variance = warp_reduce_sum(variance);
    variance = sqrtf(variance / ncols);

    for (int i=0; i < ncols; i += blockDim.x) {
        int col = i + warp_idx * 32 + lane_idx;
        if (fabsf(x[col] - mean) > 10 * variance) {
            float value = x[col] * __half2float(gate[row * ncols + col]);
            //atomicAdd(&x_outlier[row], value);
        }
    }
}

static __global__ void sign_cal(const half * x, const float * y, float * dst) {
    const int64_t row = (int64_t)blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;
    const int ncols = 5120;

    int N_N = 0;
    int N_P = 0;
    int P_N = 0;
    int P_P = 0;

    __shared__ float y_shared[5120];

    // Shared Memory 추가 코딩 부분
    for (int i=0; i < ncols; i += (32 * blockDim.y)) {
        const int col = i + tid + 32 * threadIdx.y;
        y_shared[col] = y[col];
    }
    __syncthreads();
    // End of Shared Memory

    for (int i = 0; i < ncols; i += 32) {
        const int col = i + tid;
        const int ib = row*ncols + col;
        if (y_shared[col] < 0.0f) {
            if (__half2float(x[ib]) < 0.0f)
                N_N++;
            else
                N_P++;
        }
        else {
            if (__half2float(x[ib]) < 0.0f)
                P_N++;
            else
                P_P++;
        }
    }

    // sum up partial sums and write back result
    N_N = warp_reduce_sum(N_N);
    N_P = warp_reduce_sum(N_P);
    P_P = warp_reduce_sum(P_P);
    P_N = warp_reduce_sum(P_N);

    if (tid == 0) {
        const int neg_num = N_P + P_N;
        const int pos_num = N_N + P_P;
        if ((neg_num * 100) - (pos_num * 100) > 0)
            dst[row] = 0.0f;
        else
            dst[row] = 1.0f;
    }
}

static __global__ void gemv_relu(const half * x, const float * y, float * dst, int ncols) {
    const int64_t row = (int64_t)blockIdx.y*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    int N_N = 0;
    int N_P = 0;
    int P_N = 0;
    int P_P = 0;

    __shared__ float y_shared[5120];

    // Shared Memory 추가 코딩 부분
    for (int i=0; i < ncols; i += (32 * blockDim.y)) {
        const int col = i + tid + 32 * threadIdx.y;
        y_shared[col] = y[col];
    }
    __syncthreads();
    // End of Shared Memory

    // Gate_proj
    float gate_tmp = 0.0f;

    for (int i = 0; i < ncols; i += 32) {
        const int col = i + tid;
        const int ib = row*ncols + col;
        gate_tmp += __half2float(x[ib]) * y_shared[col];
    }

    // sum up partial sums and write back result
    gate_tmp = warp_reduce_sum(gate_tmp);

    //ReLU
    if (tid == 0) {
        if (gate_tmp < 0.0f)
            dst[row] = 0.0f;
        else
            dst[row] = gate_tmp;
    }
}

//fusion_MLP
static void ggml_cuda_op_fusion_MLP(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * src2, ggml_tensor * src3, ggml_tensor * dst) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int64_t nb2 = dst->nb[2];
    const int64_t nb3 = dst->nb[3];

    GGML_ASSERT(ggml_backend_buffer_is_cuda(dst->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_cuda(src1->buffer));
    ggml_backend_cuda_buffer_context * src1_ctx = (ggml_backend_cuda_buffer_context *) src1->buffer->context;
    ggml_backend_cuda_buffer_context * dst_ctx  = (ggml_backend_cuda_buffer_context *) dst->buffer->context;

    GGML_ASSERT(src1->type == GGML_TYPE_F32 || (src1->ne[2] == 1 && src1->ne[3] == 1));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    const size_t src0_ts = ggml_type_size(src0->type);
    const size_t src0_bs = ggml_blck_size(src0->type);
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;

    const bool src0_is_contiguous = ggml_is_contiguous(src0);
    const bool src2_is_contiguous = ggml_is_contiguous(src2);
    const bool src1_is_contiguous = ggml_is_contiguous(src1);

    const int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const bool split = ggml_backend_buffer_is_cuda_split(src0->buffer);
    GGML_ASSERT(!(split && ne02 > 1));
    GGML_ASSERT(!(split && ne03 > 1));
    GGML_ASSERT(!(split && ne02 < ne12));

    ggml_tensor_extra_gpu * src0_extra = split ? (ggml_tensor_extra_gpu *) src0->extra : nullptr;

    struct dev_data {
        ggml_cuda_pool_alloc<char>  src0_dd_alloc;
        ggml_cuda_pool_alloc<char>  src2_dd_alloc;

        ggml_cuda_pool_alloc<float> src1_ddf_alloc;
        ggml_cuda_pool_alloc<char>  src1_ddq_alloc;
        ggml_cuda_pool_alloc<float>   dst_dd_alloc;

        char  *  src0_dd = nullptr;
        char  *  src2_dd = nullptr;

        float * src1_ddf = nullptr; // float
        char  * src1_ddq = nullptr; // q8_1
        float *   dst_dd = nullptr;

        int64_t  row_low;
        int64_t row_high;
    };

    dev_data dev[GGML_CUDA_MAX_DEVICES];

    int used_devices = 0;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        // by default, use all rows
        dev[id].row_low  = 0;
        dev[id].row_high = ne01;
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        if ((!split && id != ctx.device) || dev[id].row_low == dev[id].row_high) {
            continue;
        }

        used_devices++;

        const bool src1_on_device = id == src1_ctx->device;
        const bool  dst_on_device = id == dst_ctx->device;

        ggml_cuda_set_device(id);
        cudaStream_t stream = ctx.stream(id, 0);

        if (src0_is_contiguous) {
            dev[id].src0_dd = split ? (char *) src0_extra->data_device[id] : (char *) src0->data;
        } else {
            dev[id].src0_dd = dev[id].src0_dd_alloc.alloc(ctx.pool(id), ggml_nbytes(src0));
        }

        if (src2_is_contiguous) {
            dev[id].src2_dd = (char *) src2->data;
        } else {
            dev[id].src2_dd = dev[id].src2_dd_alloc.alloc(ctx.pool(id), ggml_nbytes(src2));
        }

        if (src1_on_device && src1_is_contiguous) {
            dev[id].src1_ddf = (float *) src1->data;
        } else {
            dev[id].src1_ddf = dev[id].src1_ddf_alloc.alloc(ctx.pool(id), ggml_nelements(src1));
        }

        if (dst_on_device) {
            dev[id].dst_dd = (float *) dst->data;
        } else {
            const size_t size_dst_ddf = split ? (dev[id].row_high - dev[id].row_low)*ne1 : ggml_nelements(dst);
            dev[id].dst_dd = dev[id].dst_dd_alloc.alloc(ctx.pool(id), size_dst_ddf);
        }
    }

    const int64_t src1_col_stride = ne11;
    for (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        const int64_t is = 0;
        const int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;

        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            if ((!split && id != ctx.device) || dev[id].row_low == dev[id].row_high) {
                continue;
            }
            const int64_t row_diff = dev[id].row_high - dev[id].row_low;

            ggml_cuda_set_device(id);
            cudaStream_t stream = ctx.stream(id, is);

            // 정확도 테스트
            /*
            cudaStreamSynchronize(stream);

            const int block_num_y = (src3->ne[0] + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
            const dim3 block_nums(block_num_y, 1, 1);
            const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);

            //float *pred_vector;
            //cudaMalloc((void**)&pred_vector, src3->ne[0] * sizeof(float));

            float *true_vector;
            cudaMalloc((void**)&true_vector, src3->ne[0] * sizeof(float));

            //sign_cal<<<block_nums, block_dims, 0, stream>>>((half *)src0->data, (float *)src1->data, pred_vector);
            gemv_relu<<<block_nums, block_dims, 0, stream>>>((half *)src0->data, (float *)src1->data, true_vector, src1->ne[0]);
            */

            // do the computation
            ggml_cuda_op_fusion_mul_mat_vec(ctx, src0, src1, src2, src3, dst, (char *)src0->data, (float *)src1->data, nullptr, (char *)src2->data, (float *)dst->data,
                dev[id].row_low, dev[id].row_high, src1_ncols, src1_padded_col_size, stream);
            CUDA_CHECK(cudaGetLastError());

            /*
            cudaStreamSynchronize(stream);
            int TP = 0;
            int FP = 0;
            int FN = 0;
            int TN = 0;
            int *tmp_d = (int *) malloc(1 * src3->ne[0] * sizeof(int));
            cudaMemcpy(tmp_d, src3->data, 1 * src3->ne[0] * sizeof(int), cudaMemcpyDeviceToHost);
            float *tmp_y = (float *) malloc(1 * src3->ne[0] * sizeof(float));
            cudaMemcpy(tmp_y, true_vector, 1 * src3->ne[0] * sizeof(float), cudaMemcpyDeviceToHost);
            for (int i=0; i<src3->ne[0]; i++) {
                if (tmp_d[i] == 0) {
                    if (tmp_y[i] == 0.0f)
                        TP++;
                    else
                        FP++;
                }
                else {
                    if (tmp_y[i] == 0.0f)
                        FN++;
                    else
                        TN++;
                }
            }
            float precision = (float)TP / (TP + FP) * 100;
            float recall = (float)TP / (TP + FN) * 100;
            float accuracy = (float)(TP + TN) / (TP + FP + FN + TN) * 100;
            //printf("%f,\n", (float) (TP + FP) / 13824 * 100);
            int il;
            memcpy(&il, dst->op_params, sizeof(int));
            FILE *file = fopen("/ssd/workspace/jiho/prosparse-13B_alpha0.3-gsm8k-precision-recall.txt", "a");
            fprintf(file, "il : %d [%f, %f, %f]\n", il, precision, recall, accuracy);
            fclose(file);
            //cudaFree(pred_vector);
            cudaFree(true_vector);
            free(tmp_d);
            free(tmp_y);
            */
        }
    }
}

//variance
static void ggml_cuda_op_variance(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * dst) {
    ggml_cuda_set_device(0);
    cudaStream_t stream = ctx.stream(0, 0);
    int blockDim = 128;
    int sharedMemSize = blockDim / 32 * sizeof(int);
    reduce_sum_mean<<<10, blockDim, sharedMemSize, stream>>>((float *)src0->data, (float *)dst->data, src0->ne[0], blockDim / 32);
    reduce_sum_variance<<<10, blockDim, sharedMemSize, stream>>>((float *)src0->data, (float *)dst->data, src0->ne[0], blockDim / 32);
}


//size_quantize_q1
static void ggml_cuda_op_size_quantize_q1(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                            const ggml_tensor * src2, const ggml_tensor * src3, const ggml_tensor * src4, ggml_tensor * dst) {
    ggml_cuda_set_device(0);
    cudaStream_t stream = ctx.stream(0, 0);
    int blockDim = 256;
    int sharedMemSize = blockDim / 32 * sizeof(float);
    size_quantize_Q1<<<src0->ne[0] / blockDim, blockDim, 0, stream>>>((float *)src0->data, (float *)src1->data, (int32_t *)src2->data,  dst->data, src0->ne[0], 10, dst->op_params[0]);

    // 정확도 테스트
    /*
    cudaStreamSynchronize(stream);
    float * true_vec = (float *) malloc(5120 * sizeof(float));
    cudaMemcpy(true_vec, src0->data, 5120 * sizeof(float), cudaMemcpyDeviceToHost);

    half * gate = (half *) malloc(13824 * 5120 * sizeof(half));
    cudaMemcpy(gate, src3->data, 13824 * 5120 * sizeof(half), cudaMemcpyDeviceToHost);

    //float * d_vec = (float *) malloc(13824 * sizeof(float));
    //cudaMemcpy(d_vec, src3->data, 13824 * sizeof(float), cudaMemcpyDeviceToHost);

    float mean = 0.0;
    float variance = 0.0;
    // Calculate mean
    for (int i = 0; i < 5120; i++) {
        mean += true_vec[i];
    }
    mean /= 5120;

    // Calculate variance
    for (int i = 0; i < 5120; i++) {
        variance += (true_vec[i] - mean) * (true_vec[i] - mean);
    }
    variance /= 5120;
    variance = sqrt(variance);

    float total = 0.0f;
    float outlier_total = 0.0f;
    int total_outlier_count = 0;
    for (int i=0; i<13824; i++) {
        int count = 0;
        int outlier_count = 0;
        float sum = 0.0f;
        float outlier_sum = 0.0f;
        for (int j=0; j<5120; j++) {
            if (abs(true_vec[j] - mean) > 6 * variance) {
                outlier_sum += abs(true_vec[j] * __half2float(gate[i * 5120 + j]));
                outlier_count++;
            }
            else {
                sum += abs(true_vec[j] * __half2float(gate[i * 5120 + j]));
                count++;
            }
        }
        total += sum / count;
        outlier_total += (outlier_count != 0) ? outlier_sum / outlier_count : 0;
        total_outlier_count += outlier_count;
    }
    printf("%d, ", total_outlier_count / 13824);
    //printf("non-outlier : %f, outlier : %f, 비율 : %f\n", total/13824, outlier_total/13824, (outlier_total/13824)/(total/13824));
    FILE *file = fopen("/ssd/workspace/jiho/prosparse-13B_outlier.txt", "a");
    fprintf(file, "[%f, %f, %f],\n", total/13824, outlier_total/13824, (outlier_total/13824)/(total/13824));
    fclose(file);
    free(true_vec);
    free(gate);
    */

    //정확도 테스트 2
    /*
    cudaStreamSynchronize(stream);
    float * true_vec = (float *) malloc(5120 * sizeof(float));
    cudaMemcpy(true_vec, src0->data, 5120 * sizeof(float), cudaMemcpyDeviceToHost);

    half * gate = (half *) malloc(13824 * 5120 * sizeof(half));
    cudaMemcpy(gate, src3->data, 13824 * 5120 * sizeof(half), cudaMemcpyDeviceToHost);

    //float * d_vec = (float *) malloc(13824 * sizeof(float));
    //cudaMemcpy(d_vec, src3->data, 13824 * sizeof(float), cudaMemcpyDeviceToHost);

    float mean = 0.0;
    float variance = 0.0;
    // Calculate mean
    for (int i = 0; i < 5120; i++) {
        mean += true_vec[i];
    }
    mean /= 5120;

    // Calculate variance
    for (int i = 0; i < 5120; i++) {
        variance += (true_vec[i] - mean) * (true_vec[i] - mean);
    }
    variance /= 5120;
    variance = sqrt(variance);

    int total = 0;
    for (int i=0; i<13824; i++) {
        float sum = 0.0f;
        float outlier_sum = 0.0f;
        for (int j=0; j<5120; j++) {
            if (fabs(true_vec[j] - mean) > 10 * variance) {
                outlier_sum += (true_vec[j] * __half2float(gate[i * 5120 + j]));
            }
            else {
                sum += (true_vec[j] * __half2float(gate[i * 5120 + j]));
            }
        }
        if (sum + outlier_sum <= 0) {
            if (sum > 0) {
                total++;
            }
        }
        else {
            if (sum <= 0) {
                //total++;
            }
        }
    }
    printf("이상치를 빼고 하면 결과가 달라지는 비율 : %f\n", (float)total/13824 * 100);
    free(true_vec);
    free(gate);
    */

    // 정확도 테스트 3
    /*
    cudaStreamSynchronize(stream);
    FILE *file = fopen("/ssd/workspace/jiho/prosparse-13B_calibration_c4.txt", "a");
    float * true_vec = (float *) malloc(5120 * sizeof(float));
    cudaMemcpy(true_vec, src0->data, 5120 * sizeof(float), cudaMemcpyDeviceToHost);

    half * gate = (half *) malloc(13824 * 5120 * sizeof(half));
    cudaMemcpy(gate, src3->data, 13824 * 5120 * sizeof(half), cudaMemcpyDeviceToHost);

    int32_t * gate_s = (int32_t *) malloc(13824 * 160 * sizeof(int32_t));
    cudaMemcpy(gate_s, src4->data, 13824 * 160 * sizeof(int32_t), cudaMemcpyDeviceToHost);

    //float * d_vec = (float *) malloc(13824 * sizeof(float));
    //cudaMemcpy(d_vec, src3->data, 13824 * sizeof(float), cudaMemcpyDeviceToHost);

    float mean = 0.0;
    float variance = 0.0;
    // Calculate mean
    for (int i = 0; i < 5120; i++) {
        mean += true_vec[i];
    }
    mean /= 5120;

    // Calculate variance
    for (int i = 0; i < 5120; i++) {
        variance += (true_vec[i] - mean) * (true_vec[i] - mean);
    }
    variance /= 5120;
    variance = sqrt(variance);

    for (int i=0; i<13824; i++) {
        int outlier_dif = 0;
        int big_dif = 0;
        int normal_dif = 0;
        int small_dif = 0;
        float sum = 0.0f;
        int isPositive = 0;
        for (int j=0; j<5120; j++) {
            float mul = true_vec[j] * __half2float(gate[i * 5120 + j]);
            sum += mul;
            int big_gate = ((gate_s[i*160 + j/32] >> (j%32)) & 0x00000001);
            if (abs(true_vec[j] - mean) > 6 * variance) {
                if (mul < 0)
                    outlier_dif += 1;
                else
                    outlier_dif -= 1;
            }
            // 둘 다 큰 수
            if ((abs(true_vec[j] - mean) > 0.6745 * variance) && (big_gate == 1)) {
                if (mul < 0)
                    big_dif += 1;
                else
                    big_dif -= 1;
            }
            // 둘 다 작은 수
            else if ((abs(true_vec[j] - mean) < 0.6745 * variance) && (big_gate == 0)) {
                if (mul < 0)
                    small_dif += 1;
                else
                    small_dif -= 1;
            }
            else {
                if (mul < 0)
                    normal_dif += 1;
                else
                    normal_dif -= 1;
            }
        }
        isPositive = (sum > 0) ? 0 : 1;
        fprintf(file, "[%d, %d, %d, %d, %d],\n", outlier_dif, big_dif, normal_dif, small_dif, isPositive);
    }
    fclose(file);
    free(gate_s);
    free(true_vec);
    free(gate);
    */
}

//quantize_q1
static void ggml_cuda_op_quantize_q1(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * dst) {
    ggml_cuda_set_device(0);
    cudaStream_t stream = ctx.stream(0, 0);
    quantize_Q1<<<dst->ne[0] / 32, 32, 0, stream>>>((float *)src0->data, (int32_t *)dst->data);

    // 정확도 테스트
    /*
    cudaStreamSynchronize(stream);

    float * true_vec = (float *) malloc(5120 * sizeof(float));
    cudaMemcpy(true_vec, src0->data, 5120 * sizeof(float), cudaMemcpyDeviceToHost);

    int * d_vec = (int *) malloc(160 * sizeof(int));
    cudaMemcpy(d_vec, dst->data, 160 * sizeof(int), cudaMemcpyDeviceToHost);

    int total = 0;
    for (int i=0; i<160; i++) {
        for (int j=0; j<32; j++) {
            int tmp = (true_vec[i*32 + j] < 0.0f) ? 1 : 0;
            int tmp2 = ((d_vec[i] >> j) & 0x00000001);
            if (tmp != tmp2)
                total++;
        }
    }
    printf("%d ", total);
    free(true_vec);
    free(d_vec);
    */

}

//mul_mat_vec_q1
static void ggml_cuda_op_mul_mat_vec_q1(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, const ggml_tensor * src4, ggml_tensor * dst) {
    ggml_cuda_set_device(0);
    cudaStream_t stream = ctx.stream(0, 0);
    const int block_num = (dst->ne[0] + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);

    int params[2];
    memcpy(params, dst->op_params, sizeof(int) * 2);
    mul_mat_vec_q1<<<block_num, block_dims, 0, stream>>>((int32_t *)src0->data, (int32_t *)src1->data, (int32_t *)src2->data, (int32_t *)src3->data, (int32_t *)src4->data, (int *)dst->data, dst->ne[0], src0->ne[0], (float) params[0], params[1]);
}

//mul_mat_vec_transpose
//static void ggml_cuda_op_mul_mat_vec_transpose(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
//    ggml_cuda_set_device(0);
//    cudaStream_t stream = ctx.stream(0, 0);
//
//    set_zero<<<dst->ne[0] / 256, 256, 0, stream>>>((float *)dst->data, dst->ne[0]);
//    mul_mat_vec_transpose<<<src0->ne[1], 128, 0, stream>>>((half *)src0->data, (float *)src1->data, (float *)dst->data, src0->ne[1], src0->ne[0]);
//}

static void ggml_cuda_op_mul_mat_vec_transpose(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_set_device(0);
    cudaStream_t stream = ctx.stream(0, 0);
    const int block_num_y = (src0->ne[1] + 1 - 1) / 1;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE * 8, 1, 1);

    set_zero<<<dst->ne[0] / 256, 256, 0, stream>>>((float *)dst->data, dst->ne[0]);
    dequantize_mul_mat_axpy_sparse<<<block_nums, block_dims, src0->ne[0]*sizeof(float), stream>>>((half *)src0->data, (float *)src1->data, (float *)dst->data, src0->ne[0], src0->ne[1]);
}

// ACAS output-contribution drift.
// y_delta is the MLP-output difference (computed upstream as down(h_dense - h_sparse), one dense
// down GEMV -- mathematically identical to down(h_dense)-down(h_sparse) since down is linear).
// Single-block reduction of ||y_delta|| and ||residual||, then
//   contrib = clip(1 - ||y_delta|| / (||residual|| + eps), 0, 1)
// encoded into the I32[4] confusion payload so the existing async controller reads
//   precision = TP/(TP+FP) = contrib   (no controller changes needed).
static __global__ void contrib_error_impl(const float * yd, const float * res,
                                          int n, int n_res, int * out) {
    __shared__ float sd[256];
    __shared__ float sr[256];
    const int tid = threadIdx.x;
    float ld = 0.0f, lr = 0.0f;
    for (int i = tid; i < n;     i += blockDim.x) { const float d = yd[i];  ld += d * d; }
    for (int i = tid; i < n_res; i += blockDim.x) { const float r = res[i]; lr += r * r; }
    sd[tid] = ld; sr[tid] = lr;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { sd[tid] += sd[tid + s]; sr[tid] += sr[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) {
        const float delta = sqrtf(sd[0]);
        const float rnorm = sqrtf(sr[0]) + 1e-8f;
        float contrib = 1.0f - delta / rnorm;
        contrib = fminf(1.0f, fmaxf(0.0f, contrib));
        const int S = 1000000;
        const int tp = (int) (contrib * (float) S + 0.5f);
        out[0] = tp;        // TP  -> precision numerator
        out[1] = 0;         // FN
        out[2] = S - tp;    // FP  -> precision = TP/(TP+FP) = contrib
        out[3] = 0;         // TN
    }
}

// src0 = y_delta (MLP-output drift), src1 = residual stream; dst = I32[4,1]
static void ggml_cuda_op_contrib_error(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_set_device(0);
    cudaStream_t stream = ctx.stream(0, 0);
    const int n     = (int) ggml_nelements(src0);
    const int n_res = (int) ggml_nelements(src1);
    contrib_error_impl<<<1, 256, 0, stream>>>((float *)src0->data, (float *)src1->data, n, n_res, (int *)dst->data);
}

// sparse error
static void ggml_cuda_op_sparse_error(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_set_device(0);
    cudaStream_t stream = ctx.stream(0, 0);
    const int threads_per_block = WARP_SIZE * 8;
    const int block_num = (src0->ne[0] + threads_per_block - 1) / threads_per_block;

    initialize<<<1, 1, 0, stream>>>((int *)dst->data);
    sparse_error<<<block_num, threads_per_block, 0, stream>>>((float *)src0->data, (int32_t *)src1->data, (int *)dst->data, src0->ne[0]);
}

static __global__ void eval_async_impl(const float * src, float * dst, int n) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i];
    }
}

// ACAS: non-blocking transfer of the quality payload to the controller worker thread.
struct AsyncTransfer {
    int* cpu_buffer;
    int layer_id;
    size_t size;
};

static std::mutex g_async_transfer_pool_mutex;
static std::vector<int*> g_async_transfer_pool;
static size_t g_async_transfer_pool_size = 0;

// Dedicated stream and pooled pinned buffers for async transfers
static cudaStream_t g_async_eval_stream = nullptr;

static int * acquire_async_transfer_buffer(size_t size) {
    std::lock_guard<std::mutex> lock(g_async_transfer_pool_mutex);

    // Reset pool if size changes
    if (g_async_transfer_pool_size != 0 && g_async_transfer_pool_size != size) {
        ggml_cuda_set_device(0);
        for (int * buf : g_async_transfer_pool) {
            cudaFreeHost(buf);
        }
        g_async_transfer_pool.clear();
        g_async_transfer_pool_size = 0;
    }

    if (!g_async_transfer_pool.empty()) {
        int * buf = g_async_transfer_pool.back();
        g_async_transfer_pool.pop_back();
        return buf;
    }

    int * buf = nullptr;
    CUDA_CHECK(cudaHostAlloc((void**)&buf, size, cudaHostAllocDefault));
    g_async_transfer_pool_size = size;
    return buf;
}

static void release_async_transfer_buffer(int * buffer) {
    if (buffer == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_async_transfer_pool_mutex);
    g_async_transfer_pool.push_back(buffer);
}

static void free_async_transfer_pool() {
    std::lock_guard<std::mutex> lock(g_async_transfer_pool_mutex);
    if (g_async_transfer_pool.empty()) return;
    cudaSetDevice(0);
    for (int * buf : g_async_transfer_pool) {
        cudaFreeHost(buf);
    }
    g_async_transfer_pool.clear();
    g_async_transfer_pool_size = 0;
}

// Host callback: runs after the D2H copy completes; must stay O(1) and CUDA-free.
static void CUDART_CB async_payload_callback(void * user_data) {
    AsyncTransfer * transfer = static_cast<AsyncTransfer *>(user_data);
    if (transfer == nullptr) {
        return;
    }

    if (g_async_controller && transfer->cpu_buffer) {
        AsyncConfusionMatrix cm;
        cm.layer_id = transfer->layer_id;
        cm.TP = transfer->cpu_buffer[0];
        cm.FN = transfer->cpu_buffer[1];
        cm.FP = transfer->cpu_buffer[2];
        cm.TN = transfer->cpu_buffer[3];

        g_async_controller->enqueue_payload(cm);
    }

    if (transfer->cpu_buffer) {
        release_async_transfer_buffer(transfer->cpu_buffer);
    }
    delete transfer;
}

// Payload: I32[4] confusion matrix. src0 passes through to dst on the compute stream.
static void queue_payload_transfer(cudaStream_t compute_stream, int layer_id, const ggml_tensor * src1) {
    cudaStream_t async_stream = g_async_eval_stream ? g_async_eval_stream : compute_stream;

    AsyncTransfer * transfer = new AsyncTransfer();
    transfer->size = sizeof(int) * src1->ne[0];
    transfer->cpu_buffer = acquire_async_transfer_buffer(transfer->size);
    transfer->layer_id = layer_id;

    cudaEvent_t ready_evt;
    CUDA_CHECK(cudaEventCreateWithFlags(&ready_evt, cudaEventDisableTiming));

    CUDA_CHECK(cudaEventRecord(ready_evt, compute_stream));
    CUDA_CHECK(cudaStreamWaitEvent(async_stream, ready_evt, 0));

    CUDA_CHECK(cudaMemcpyAsync(transfer->cpu_buffer, (int*)src1->data, transfer->size, cudaMemcpyDeviceToHost, async_stream));
    CUDA_CHECK(cudaLaunchHostFunc(async_stream, async_payload_callback, transfer));
    CUDA_CHECK(cudaEventDestroy(ready_evt));
}

static void ggml_cuda_op_eval_async(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int layer_id) {
    ggml_cuda_set_device(0);
    cudaStream_t compute_stream = ctx.stream(0, 0);
    queue_payload_transfer(compute_stream, layer_id, src1);

// GPU continues immediately
    const int threads_per_block = WARP_SIZE * 8;
    const int n = ggml_nelements(src0);
    const int block_num = (n + threads_per_block - 1) / threads_per_block;
    eval_async_impl<<<block_num, threads_per_block, 0, compute_stream>>>((float *)src0->data, (float *)dst->data, n);
}

void init_async_transfer_processing() {
    ggml_cuda_set_device(0);
    if (g_async_eval_stream == nullptr) {
        int low = 0, high = 0;
        cudaDeviceGetStreamPriorityRange(&low, &high);
        CUDA_CHECK(cudaStreamCreateWithPriority(&g_async_eval_stream, cudaStreamNonBlocking, high));
    }
}

void cleanup_async_transfer_processing() {
    ggml_cuda_set_device(0);
    if (g_async_eval_stream) {
        CUDA_CHECK(cudaStreamDestroy(g_async_eval_stream));
        g_async_eval_stream = nullptr;
    }
    free_async_transfer_pool();
}

//mul_mat_vec
static void ggml_cuda_op_mul_mat_vec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_set_device(0);
    cudaStream_t stream = ctx.stream(0, 0);
    const int block_num_y = (src0->ne[1] + 16 - 1) / 16;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, 16, 1);

    mul_mat_vec<<<block_nums, block_dims, 0, stream>>>((half *)src0->data, (float *)src1->data, (float *)dst->data, src0->ne[1], src0->ne[0]);
}

//mul_mat_vec_sparse
static void ggml_cuda_op_mul_mat_vec_sparse_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, ggml_tensor * dst) {
    ggml_cuda_set_device(0);
    cudaStream_t stream = ctx.stream(0, 0);
    const int block_num_y = (src0->ne[1] + 16 - 1) / 16;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, 16, 1);


    // 정확도 테스트
    /*
    float *true_vector;
    cudaMalloc((void**)&true_vector, dst->ne[0] * sizeof(float));
    gemv_relu<<<block_nums, block_dims, 0, stream>>>((half *)src0->data, (float *)src1->data, true_vector, src1->ne[0]);

    cudaStreamSynchronize(stream);

    int TP = 0;
    int FP = 0;
    int FN = 0;
    int TN = 0;
    int *tmp_d = (int *) malloc(1 * dst->ne[0] * sizeof(int));
    cudaMemcpy(tmp_d, src2->data, 1 * dst->ne[0] * sizeof(int), cudaMemcpyDeviceToHost);
    float *tmp_y = (float *) malloc(1 * dst->ne[0] * sizeof(float));
    cudaMemcpy(tmp_y, true_vector, 1 * dst->ne[0] * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i=0; i<dst->ne[0]; i++) {
        if (tmp_d[i] == 0) {
            if (tmp_y[i] == 0.0f)
                TP++;
            else
                FP++;
        }
        else {
            if (tmp_y[i] == 0.0f)
                FN++;
            else
                TN++;
        }
    }
    float precision = (float)TP / (TP + FP) * 100;
    float recall = (float)TP / (TP + FN) * 100;
    float accuracy = (float)(TP + TN) / (TP + FP + FN + TN) * 100;
    int il;
    memcpy(&il, dst->op_params, sizeof(int));
    FILE *file = fopen("/ssd/workspace/jiho/prosparse-13B_SizeBitXWgate_Outlier_Regression_gsm8k_alpha0.0-gsm8k-precision-recall.txt", "a");
    fprintf(file, "il : %d [%f, %f, %f]\n", il, precision, recall, accuracy);
    fclose(file);
    cudaFree(true_vector);
    free(tmp_d);
    free(tmp_y);
    */

    //op
    mul_mat_vec_sparse_gate<<<block_nums, block_dims, 0, stream>>>((half *)src0->data, (float *)src1->data, (float *)dst->data, (int *)src2->data, src0->ne[1], src0->ne[0]);
}

//mul_mat_vec_sparse
static void ggml_cuda_op_mul_mat_vec_sparse_up(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, ggml_tensor * dst) {
    ggml_cuda_set_device(0);
    cudaStream_t stream = ctx.stream(0, 0);
    const int block_num_y = (src0->ne[1] + 16 - 1) / 16;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, 16, 1);

    mul_mat_vec_sparse_up<<<block_nums, block_dims, 0, stream>>>((half *)src0->data, (float *)src1->data, (float *)dst->data, (int *)src2->data, src0->ne[1], src0->ne[0]);
}

static void ggml_cuda_mul_mat_vec_p021(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst){
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(ggml_backend_buffer_is_cuda(src0->buffer));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t ne12 = src1->ne[2];

    cudaStream_t main_stream = ctx.stream();

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    ggml_mul_mat_p021_f16_f32_cuda(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_stream);
}

static void ggml_cuda_mul_mat_vec_nc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst){
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(ggml_backend_buffer_is_cuda(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];

    const int64_t ne12 = src1->ne[2];

    cudaStream_t main_stream = ctx.stream();

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    const int64_t row_stride_x = nb01 / sizeof(half);
    const int64_t channel_stride_x = nb02 / sizeof(half);

    ggml_mul_mat_vec_nc_f16_f32_cuda(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, row_stride_x, ne02, ne12, channel_stride_x, main_stream);
}

static __global__ void k_compute_batched_ptrs(
        const half * src0_as_f16, const half * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    int64_t i03 = i13 / r3;
    int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

static void ggml_cuda_mul_mat_batched_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));

    GGML_ASSERT(ggml_backend_buffer_is_cuda(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ne_dst = ggml_nelements(dst);

    cudaStream_t main_stream = ctx.stream();

    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), main_stream));

    void * src0_ddq = src0->data;
    half * src0_f16 = (half *) src0_ddq;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    // convert src1 to fp16
    ggml_cuda_pool_alloc<half> src1_f16_alloc(ctx.pool());
    if (src1->type != GGML_TYPE_F16) {
        const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src1->type);
        const int64_t ne_src1 = ggml_nelements(src1);
        src1_f16_alloc.alloc(ne_src1);
        GGML_ASSERT(to_fp16_cuda != nullptr);
        to_fp16_cuda(src1_ddf, src1_f16_alloc.get(), ne_src1, main_stream);
    }
    half * src1_f16 = src1->type == GGML_TYPE_F16 ? (half *) src1_ddf : src1_f16_alloc.get();

    ggml_cuda_pool_alloc<half> dst_f16(ctx.pool());
    char * dst_t;

    cublasComputeType_t cu_compute_type = CUBLAS_COMPUTE_16F;
    cudaDataType_t      cu_data_type    = CUDA_R_16F;

    // dst strides
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    const half  alpha_f16 = 1.0f;
    const half  beta_f16  = 0.0f;

    const float alpha_f32 = 1.0f;
    const float beta_f32  = 0.0f;

    const void * alpha = &alpha_f16;
    const void * beta  = &beta_f16;

    if (dst->op_params[0] == GGML_PREC_DEFAULT) {
        dst_t = (char *) dst_f16.alloc(ne_dst);

        nbd2 /= sizeof(float) / sizeof(half);
        nbd3 /= sizeof(float) / sizeof(half);
    } else {
        dst_t = (char *) dst_ddf;

        cu_compute_type = CUBLAS_COMPUTE_32F;
        cu_data_type    = CUDA_R_32F;

        alpha = &alpha_f32;
        beta  = &beta_f32;
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

#if 0
    // use cublasGemmEx
    {
        for (int i13 = 0; i13 < ne13; ++i13) {
            for (int i12 = 0; i12 < ne12; ++i12) {
                int i03 = i13 / r3;
                int i02 = i12 / r2;

                CUBLAS_CHECK(
                        cublasGemmEx(g_cublas_handles[g_main_device], CUBLAS_OP_T, CUBLAS_OP_N,
                            ne01, ne11, ne10,
                            alpha, (const char *) src0_as_f16 + i02*src0->nb[2]   + i03*src0->nb[3]  , CUDA_R_16F,   nb01/sizeof(half),
                                   (const char *) src1_as_f16 + i12*src1->nb[2]/2 + i13*src1->nb[3]/2, CUDA_R_16F,   nb11/sizeof(float),
                            beta,  (      char *)       dst_t + i12*nbd2          + i13*nbd3,          cu_data_type, ne01,
                            cu_compute_type,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
        }
    }
#else
    if (r2 == 1 && r3 == 1 && src0->nb[2]*src0->ne[2] == src0->nb[3] && src1->nb[2]*src1->ne[2] == src1->nb[3]) {
        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const char *) src0_f16, CUDA_R_16F,   nb01/nb00, nb02/nb00,  // strideA
                       (const char *) src1_f16, CUDA_R_16F,   nb11/nb10, nb12/nb10,  // strideB
                beta,  (      char *)    dst_t, cu_data_type, ne01,       nb2/nb0,   // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // use cublasGemmBatchedEx
        const int ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        dim3 block_dims(ne13, ne12);
        k_compute_batched_ptrs<<<1, block_dims, 0, main_stream>>>(
                src0_f16, src1_f16, dst_t,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                nb02, nb03,
                src1->type == GGML_TYPE_F16 ? nb12 : nb12/2,
                src1->type == GGML_TYPE_F16 ? nb13 : nb13/2,
                nbd2, nbd3,
                r2, r3);
        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), CUDA_R_16F,   nb01/nb00,
                       (const void **) (ptrs_src.get() + 1*ne23), CUDA_R_16F,   nb11/nb10,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type, ne01,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
#endif

    if (dst->op_params[0] == GGML_PREC_DEFAULT) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
        to_fp32_cuda(dst_f16.get(), dst_ddf, ne_dst, main_stream);
    }
}

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const bool split = ggml_backend_buffer_is_cuda_split(src0->buffer);

    int64_t min_compute_capability = INT_MAX;

    bool any_pascal_with_slow_fp16 = false;
    if (split) {
        ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *) src0->buffer->buft->context;
        auto & tensor_split = buft_ctx->tensor_split;
        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            // skip devices that are not going to do any work:
            if (tensor_split[id] >= (id + 1 < ggml_backend_cuda_get_device_count() ? tensor_split[id + 1] : 1.0f)) {
                continue;
            }

            if (min_compute_capability > ggml_cuda_info().devices[id].cc) {
                min_compute_capability = ggml_cuda_info().devices[id].cc;
            }
            if (ggml_cuda_info().devices[id].cc == 610) {
                any_pascal_with_slow_fp16 = true;
            }
        }
    } else {
        min_compute_capability    = ggml_cuda_info().devices[ctx.device].cc;
        any_pascal_with_slow_fp16 = ggml_cuda_info().devices[ctx.device].cc == 610;
    }

    // check data types and tensor shapes for custom matrix multiplication kernels:
    bool use_dequantize_mul_mat_vec = (ggml_is_quantized(src0->type) || src0->type == GGML_TYPE_F16)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src0->ne[0] % GGML_CUDA_DMMV_X == 0 && src1->ne[1] == 1;

    bool          use_mul_mat_vec_q =  ggml_is_quantized(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;

    bool              use_mul_mat_q =  ggml_cuda_supports_mmq(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)

    const bool fp16_performance_good = min_compute_capability >= CC_RDNA1;

#ifdef CUDA_USE_TENSOR_CORES
    use_mul_mat_q = use_mul_mat_q && min_compute_capability < CC_RDNA3;
#endif // CUDA_USE_TENSOR_CORES

#else

    // fp16 performance is good on Volta or newer and on P100 (compute capability 6.0)
    const bool fp16_performance_good = min_compute_capability >= CC_PASCAL && !any_pascal_with_slow_fp16;

    // mmvq and mmq need the __dp4a instruction which on NVIDIA is only available for CC >= 6.1
    use_mul_mat_vec_q = use_mul_mat_vec_q && min_compute_capability >= MIN_CC_DP4A;
    use_mul_mat_q     = use_mul_mat_q     && min_compute_capability >= MIN_CC_DP4A;

#ifdef CUDA_USE_TENSOR_CORES
    // when tensor cores are available, use them for large batch size
    // ref: https://github.com/ggerganov/llama.cpp/pull/3776
    use_mul_mat_q     = use_mul_mat_q     && (!fp16_performance_good || src1->ne[1] <= MMQ_MAX_BATCH_SIZE);
#endif // CUDA_USE_TENSOR_CORES

#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)

    // if mmvq is available it's a better choice than dmmv:
#ifndef GGML_CUDA_FORCE_DMMV
    use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && !use_mul_mat_vec_q;
#endif // GGML_CUDA_FORCE_DMMV

    // debug helpers
    //printf("src0: %8d %8d %8d %8d\n", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
    //printf("src1: %8d %8d %8d %8d\n", src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
    //printf("src0 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src0), ggml_is_transposed(src0), ggml_type_name(src0->type), src0->name);
    //printf("src1 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src1), ggml_is_transposed(src1), ggml_type_name(src1->type), src1->name);

    if (!split && !fp16_performance_good && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        // KQ single-batch
        ggml_cuda_mul_mat_vec_p021(ctx, src0, src1, dst);
    } else if (!split && !fp16_performance_good && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1) {
        // KQV single-batch
        ggml_cuda_mul_mat_vec_nc(ctx, src0, src1, dst);
    } else if (!split && src0->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || fp16_performance_good) && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2]*src1->ne[3] > 1) {
        // KQ + KQV multi-batch
        ggml_cuda_mul_mat_batched_cublas(ctx, src0, src1, dst);
    } else if (use_dequantize_mul_mat_vec) {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_dequantize_mul_mat_vec, false);
    } else if (use_mul_mat_vec_q) {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_vec_q, true);
    } else if (use_mul_mat_q) {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_q, true);
    } else {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_cublas, false);
    }
}

static __global__ void debug_impl(const float * x, float * dst) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    dst[i] = x[i];
}

static void ggml_cuda_debug(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {

    ggml_cuda_set_device(0);
    cudaStream_t stream = ctx.stream(0, 0);

    // Debug gate Matrix
    cudaStreamSynchronize(stream);
    FILE *file = fopen("/ssd/workspace/jiho/prosparse-13B-Y.txt", "a");

    int il;
    memcpy(&il, dst->op_params, sizeof(int));

    float *tmp = (float *) malloc(1 * 5120 * sizeof(float));
    cudaMemcpy(tmp, (float *)src1->data, 1 * 5120 * sizeof(float), cudaMemcpyDeviceToHost);

    half *tmp2 = (half *) malloc(1 * 5120 * sizeof(half));
    cudaMemcpy(tmp2, (half *)src0->data, 1 * 5120 * sizeof(half), cudaMemcpyDeviceToHost);
    fprintf(file, "[", il);
    for (int i=0; i<5120; i++) {
        fprintf(file, "%f, ", tmp[i] * __half2float(tmp2[i]));
    }
    fprintf(file, "],\n");
    fclose(file);

    debug_impl<<<5120 / 32, 32, 0, stream>>>((float *)src1->data, (float *) dst->data);

}

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

static __global__ void k_copy_src1_to_contiguous(const char * __restrict__ src1_original, char * __restrict__ src1_contiguous,
                                                 int * __restrict__ cur_src1_row, mmid_row_mapping * __restrict__ row_mapping,
                                                 const char * __restrict ids, int64_t i02, size_t ids_nb1, size_t ids_nb0,
                                                 int64_t ne11, int64_t ne10,
                                                 size_t nb11, size_t nb12) {
    int32_t iid1 = blockIdx.x;
    int32_t id = blockIdx.y;

    const int32_t row_id_i = *(const int32_t *) (ids + iid1*ids_nb1 + id*ids_nb0);

    if (row_id_i != i02) {
        return;
    }

    const int64_t i11 = id % ne11;
    const int64_t i12 = iid1;

    __shared__ int src1_row;
    if (threadIdx.x == 0) {
        src1_row = atomicAdd(cur_src1_row, 1);
        row_mapping[src1_row] = {id, iid1};
    }
    __syncthreads();

    const float * src1_row_original = (const float *)(src1_original + i11*nb11 + i12*nb12);
    float * src1_row_contiguous = (float *)(src1_contiguous + src1_row*nb11);

    for (int i = threadIdx.x; i < ne10; i += blockDim.x) {
        src1_row_contiguous[i] = src1_row_original[i];
    }
}

static __global__ void k_copy_dst_from_contiguous(char * __restrict__ dst_original, const char * __restrict__ dst_contiguous,
                                                  const mmid_row_mapping * __restrict__ row_mapping,
                                                  int64_t ne0,
                                                  size_t nb1, size_t nb2) {
    int32_t i = blockIdx.x;

    const int32_t i1 = row_mapping[i].i1;
    const int32_t i2 = row_mapping[i].i2;

    const float * dst_row_contiguous = (const float *)(dst_contiguous + i*nb1);
    float * dst_row_original = (float *)(dst_original + i1*nb1 + i2*nb2);

    for (int j = threadIdx.x; j < ne0; j += blockDim.x) {
        dst_row_original[j] = dst_row_contiguous[j];
    }
}

static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(!ggml_backend_buffer_is_cuda_split(src0->buffer) && "mul_mat_id does not support split buffers");

    cudaStream_t stream = ctx.stream();

    const int64_t n_as = ne02;
    const int64_t n_ids = ids->ne[0];

    std::vector<char> ids_host(ggml_nbytes(ids));
    const char * ids_dev = (const char *) ids->data;
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids_dev, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row  = *dst;

    char * src0_original = (char *) src0->data;
    char * src1_original = (char *) src1->data;
    char * dst_original  = (char *)  dst->data;

    src0_row.ne[2] = 1;
    src0_row.ne[3] = 1;
    src0_row.nb[3] = nb02;

    src1_row.ne[1] = 1;
    src1_row.ne[2] = 1;
    src1_row.ne[3] = 1;
    src1_row.nb[2] = nb11;
    src1_row.nb[3] = nb11;

    dst_row.ne[1] = 1;
    dst_row.ne[2] = 1;
    dst_row.ne[3] = 1;
    dst_row.nb[2] = nb1;
    dst_row.nb[3] = nb1;

    if (ne12 == 1) {
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
            for (int64_t id = 0; id < n_ids; id++) {
                const int32_t i02 = *(const int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);

                GGML_ASSERT(i02 >= 0 && i02 < n_as);

                const int64_t i11 = id % ne11;
                const int64_t i12 = iid1;

                const int64_t i1 = id;
                const int64_t i2 = i12;

                src0_row.data = src0_original + i02*nb02;
                src1_row.data = src1_original + i11*nb11 + i12*nb12;
                dst_row.data  =  dst_original + i1*nb1   + i2*nb2;

                ggml_cuda_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
            }
        }
    } else {
        ggml_cuda_pool_alloc<char> src1_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(src1));
        ggml_cuda_pool_alloc<char>  dst_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(dst));

        src1_row.data = src1_contiguous.get();
        dst_row.data  =  dst_contiguous.get();

        for (int64_t i02 = 0; i02 < n_as; i02++) {
            int64_t num_src1_rows = 0;

            for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
                for (int64_t id = 0; id < n_ids; id++) {
                    const int32_t row_id_i = *(const int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);

                    GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);

                    if (row_id_i != i02) {
                        continue;
                    }

                    num_src1_rows++;
                }
            }

            if (num_src1_rows == 0) {
                continue;
            }

            ggml_cuda_pool_alloc<int> dev_cur_src1_row(ctx.pool(), 1);
            ggml_cuda_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool(), num_src1_rows);
            CUDA_CHECK(cudaMemsetAsync(dev_cur_src1_row.get(), 0, sizeof(int), stream));

            {
                dim3 block_dims(std::min((unsigned int)ne10, 768u));
                dim3 grid_dims(ids->ne[1], n_ids);
                k_copy_src1_to_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                        src1_original, src1_contiguous.get(),
                        dev_cur_src1_row.get(), dev_row_mapping.get(),
                        ids_dev, i02, ids->nb[1], ids->nb[0],
                        ne11, ne10,
                        nb11, nb12);
                CUDA_CHECK(cudaGetLastError());
            }

            src0_row.data = src0_original + i02*nb02;

            GGML_ASSERT(nb11 == sizeof(float)*ne10);
            GGML_ASSERT(nb1 == sizeof(float)*ne0);

            src1_row.ne[1] = num_src1_rows;
            src1_row.nb[1] = nb11;
            src1_row.nb[2] = num_src1_rows*nb11;
            src1_row.nb[3] = num_src1_rows*nb11;

            dst_row.ne[1] = num_src1_rows;
            dst_row.nb[1] = nb1;
            dst_row.nb[2] = num_src1_rows*nb1;
            dst_row.nb[3] = num_src1_rows*nb1;

            ggml_cuda_mul_mat(ctx, &src0_row, &src1_row, &dst_row);

            {
                dim3 block_dims(std::min((unsigned int)ne0, 768u));
                dim3 grid_dims(num_src1_rows);
                k_copy_dst_from_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                        dst_original, dst_contiguous.get(),
                        dev_row_mapping.get(),
                        ne0,
                        nb1, nb2);
                CUDA_CHECK(cudaGetLastError());
            }
        }
    }
}

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    // why is this here instead of mul_mat?
    if (dst->src[0] != nullptr && ggml_backend_buffer_is_cuda_split(dst->src[0]->buffer)) {
        ggml_cuda_set_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    switch (dst->op) {
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            if (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                fprintf(stderr, "%s: cannot compute %s: src0->ne[3] = %" PRId64 ", src1->ne[3] = %" PRId64 " - fallback to CPU\n", __func__, dst->name, dst->src[0]->ne[3], dst->src[1]->ne[3]);
                return false;
            } else {
                ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            }
            break;
        case GGML_OP_DEBUG:
            ggml_cuda_debug(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_FUSTION_MLP:
            if (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                fprintf(stderr, "%s: cannot compute %s: src0->ne[3] = %" PRId64 ", src1->ne[3] = %" PRId64 " - fallback to CPU\n", __func__, dst->name, dst->src[0]->ne[3], dst->src[1]->ne[3]);
                return false;
            } else {
                ggml_cuda_op_fusion_MLP(ctx, dst->src[0], dst->src[1], dst->src[2], dst->src[3], dst);
            }
            break;
        case GGML_OP_VARIANCE:
            ggml_cuda_op_variance(ctx, dst->src[0], dst);
            break;
        case GGML_OP_SIZE_QUANTIZE_Q1:
            ggml_cuda_op_size_quantize_q1(ctx, dst->src[0], dst->src[1], dst->src[2], dst->src[3], dst->src[4], dst);
            break;
        case GGML_OP_QUANTIZE_Q1:
            ggml_cuda_op_quantize_q1(ctx, dst->src[0], dst);
            break;
        case GGML_OP_MUL_MAT_VEC_Q1:
            ggml_cuda_op_mul_mat_vec_q1(ctx, dst->src[0], dst->src[1], dst->src[2], dst->src[3], dst->src[4], dst);
            break;
        case GGML_OP_MUL_MAT_VEC_T:
            ggml_cuda_op_mul_mat_vec_transpose(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_VEC:
            ggml_cuda_op_mul_mat_vec(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_SPARSE_ERROR:
            ggml_cuda_op_sparse_error(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_CONTRIB_ERROR:
            ggml_cuda_op_contrib_error(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_EVAL_ASYNC:
            {
                // Unpack layer_id from extra field
                int64_t packed = *((int64_t*)dst->extra);
                int layer_id = (int)(packed >> 32);
                ggml_cuda_op_eval_async(ctx, dst->src[0], dst->src[1], dst, layer_id);
            }
            break;
        case GGML_OP_MUL_MAT_VEC_S_G:
            ggml_cuda_op_mul_mat_vec_sparse_gate(ctx, dst->src[0], dst->src[1], dst->src[2], dst);
            break;
        case GGML_OP_MUL_MAT_VEC_S_U:
            ggml_cuda_op_mul_mat_vec_sparse_up(ctx, dst->src[0], dst->src[1], dst->src[2], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_ALIBI:
            ggml_cuda_op_alibi(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

GGML_CALL static const char * ggml_backend_cuda_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

GGML_CALL static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_cuda_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return ggml_backend_cuda_buffer_type(cuda_ctx->device);
}

GGML_CALL static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync((char *)tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

GGML_CALL static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *)tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

GGML_CALL static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_backend_is_cuda(backend_src) || ggml_backend_is_cuda(backend_dst));

    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_buffer_is_cuda(src->buffer)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(dst->buffer)) {
        return false;
    }

    // device -> device
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *)backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *)backend_dst->context;

    if (backend_src != backend_dst) {
        ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *)buf_src->context;
        ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *)buf_dst->context;

        GGML_ASSERT(cuda_ctx_src->device == buf_ctx_src->device);
        GGML_ASSERT(cuda_ctx_dst->device == buf_ctx_dst->device);

        // copy on src stream
        if (cuda_ctx_src->device == cuda_ctx_dst->device) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_dst->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, cuda_ctx_dst->device, src->data, cuda_ctx_src->device, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif
        }

        // record event on src stream
        if (!cuda_ctx_src->copy_event) {
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
    } else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_dst->stream()));
    }
    return true;
}

GGML_CALL static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}

GGML_CALL static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }

#ifndef NDEBUG
        assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device));
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j] != nullptr) {
                assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) || ggml_backend_buffer_is_cuda_split(node->src[j]->buffer));
            }
        }
#endif

        bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
        if (!ok) {
            fprintf(stderr, "%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
    }

    return GGML_STATUS_SUCCESS;
}

GGML_CALL static bool ggml_backend_cuda_supports_op(ggml_backend_t backend, const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                    return true;
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a;
                struct ggml_tensor * b;
                if (op->op == GGML_OP_MUL_MAT) {
                    a = op->src[0];
                    b = op->src[1];
                } else {
                    a = op->src[2];
                    b = op->src[1];
                }
                if (a->ne[3] != b->ne[3]) {
                    return false;
                }
                ggml_type a_type = a->type;
                if (a_type == GGML_TYPE_IQ2_XXS || a_type == GGML_TYPE_IQ2_XS || a_type == GGML_TYPE_IQ3_XXS ||
                    a_type == GGML_TYPE_IQ1_S   || a_type == GGML_TYPE_IQ4_NL || a_type == GGML_TYPE_IQ3_S   ||
                    a_type == GGML_TYPE_IQ1_M   || a_type == GGML_TYPE_IQ2_S  || a_type == GGML_TYPE_IQ4_XS) {
                    if (b->ne[1] == 1 && ggml_nrows(b) > 1) {
                        return false;
                    }
                }
                return true;
            } break;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_DUP:
        case GGML_OP_REPEAT:
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
            } break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_NORM:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_CLAMP:
        case GGML_OP_CONT:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ROPE:
        case GGML_OP_ALIBI:
        case GGML_OP_IM2COL:
        case GGML_OP_POOL_2D:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_ARGSORT:
        case GGML_OP_ACC:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
            return true;
        default:
            return false;
    }

    GGML_UNUSED(backend);
}

GGML_CALL static bool ggml_backend_cuda_offload_op(ggml_backend_t backend, const ggml_tensor * op) {
    const int min_batch_size = 32;

    return (op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS) ||
           (op->ne[2] >= min_batch_size && op->op == GGML_OP_MUL_MAT_ID);

    GGML_UNUSED(backend);
}

static ggml_backend_event_t ggml_backend_cuda_event_new(ggml_backend_t backend) {
#ifdef GGML_CUDA_NO_PEER_COPY
    return nullptr;
#else
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .backend = */ backend,
        /* .context = */ event,
    };
#endif
}

static void ggml_backend_cuda_event_free(ggml_backend_event_t event) {
    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));

    delete event;
}

static void ggml_backend_cuda_event_record(ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)event->backend->context;

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (ggml_backend_is_cuda(event->backend)) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ASSERT(false);
    }
}

static void ggml_backend_cuda_event_synchronize(ggml_backend_event_t event) {
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

static ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .get_default_buffer_type = */ ggml_backend_cuda_get_default_buffer_type,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .supports_op             = */ ggml_backend_cuda_supports_op,
    /* .offload_op              = */ ggml_backend_cuda_offload_op,
    /* .event_new               = */ ggml_backend_cuda_event_new,
    /* .event_free              = */ ggml_backend_cuda_event_free,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .event_synchronize       = */ ggml_backend_cuda_event_synchronize,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

GGML_CALL ggml_backend_t ggml_backend_cuda_init(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        fprintf(stderr, "%s: error: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: error: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_cuda_guid(),
        /* .interface = */ ggml_backend_cuda_interface,
        /* .context   = */ ctx
    };

    return cuda_backend;
}

GGML_CALL bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

GGML_CALL int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

GGML_CALL void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    snprintf(description, description_size, "%s", prop.name);
}

GGML_CALL void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));
}

GGML_CALL bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return false;
    }

#if CUDART_VERSION >= 11100
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        cudaGetLastError();

        fprintf(stderr, "%s: warning: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                size/1024.0/1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    return false;
#endif
}

GGML_CALL void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        cudaGetLastError();
    }
}

// backend registry
GGML_CALL static ggml_backend_t ggml_backend_reg_cuda_init(const char * params, void * user_data) {
    ggml_backend_t cuda_backend = ggml_backend_cuda_init((int) (intptr_t) user_data);
    return cuda_backend;

    GGML_UNUSED(params);
}

extern "C" GGML_CALL int ggml_backend_cuda_reg_devices();

GGML_CALL int ggml_backend_cuda_reg_devices() {
    int device_count = ggml_backend_cuda_get_device_count();
    //int device_count = 1; // DEBUG: some tools require delaying CUDA initialization
    for (int i = 0; i < device_count; i++) {
        char name[128];
        snprintf(name, sizeof(name), "%s%d", GGML_CUDA_NAME, i);
        ggml_backend_register(name, ggml_backend_reg_cuda_init, ggml_backend_cuda_buffer_type(i), (void *) (intptr_t) i);
    }
    return device_count;
}

void MemCopy(void * src, void *dst, int64_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

void MemCopy_HTOD(void * src, void *dst, int64_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void MemMalloc(int * d_array, size_t size) {
    cudaMalloc((void **)&d_array, size);
}

int isNegative(void * data) {
    half * tmp = (half *) data;
    if (__half2float(*tmp) < 0.0f)
        return 1;
    else
        return 0;
}

int isBig(float Q1, float Q3, void * data) {
    half * tmp = (half *) data;
    if (__half2float(*tmp) < Q1 || __half2float(*tmp) > Q3)
        return 1;
    else
        return 0;
}

void calQ1Q3(void * data, float * dst, int rows, int cols) {
    half * tmp = (half *) data;
    for (int i=0; i<rows; i++) {
        float mean = 0.0f;
        float std = 0.0f;
        for (int j=0; j<cols; j++) {
            mean += __half2float(tmp[i * cols + j]);
        }
        mean /= cols;
        for (int j=0; j<cols; j++) {
            std += (__half2float(tmp[i * cols + j]) - mean) * (__half2float(tmp[i * cols + j]) - mean);
        }
        std = sqrt(std / cols);
        dst[i * 2 + 0] = mean - 0.6745 * std;
        dst[i * 2 + 1] = mean + 0.6745 * std;
    }
}

//TODO: Hard-coded hyperparameters need to be revised.
void ggml_cuda_set_device_constants() {
    const float std6[] = {
        85.61433623900221,15.523354727672332,3.978918085920551,
        54.89820120106032,15.208394805399521,3.933713673260931,
        38.48385603041713,15.317835577405475,3.930136544874956,
        47.511050333914795,15.250526033480362,3.926523001948811,
        56.08373787353046,15.163743496770074,3.916299967874994,
        75.6678313078744,15.25108835537739,3.9063718277549175,
        72.3729662065994,15.265413329225979,3.905028628195499,
        66.8056798635457,15.277871608173811,3.9067184242023014,
        68.6940251826593,15.227856423237181,3.900154085656198,
        70.54818485143115,15.239834082663904,3.900500863713783,
        71.54021684586368,15.215733959421321,3.896830568810752,
        71.59332910477399,15.25171622157222,3.8995601787626266,
        79.40334678014733,15.286763133819825,3.9024800625012612,
        77.91022468029038,15.287899955271804,3.9048852561390013,
        68.96848784841666,15.327329844756454,3.9100964988581244,
        75.26644584681915,15.303856386404753,3.910150680524277,
        53.68015858267448,15.32483949722231,3.9143610296971887,
        93.13391803307195,15.345897990975686,3.905787026626225,
        110.82023290598704,15.36346124071264,3.9072983106609005,
        77.53367115032218,15.359327179285506,3.909032528588298,
        91.2089062383907,15.324805980890309,3.8975533845212493,
        79.31977256740761,15.37774765629847,3.903711306291901,
        93.60031321931224,15.34684644161128,3.8958009607725734,
        87.87371245837707,15.345874703526498,3.897102644730032,
        82.07775249811819,15.393592377168833,3.8993739147387916,
        80.03722164148171,15.456361533734654,3.9067977778527774,
        80.90649178476853,15.29571319441261,3.8880207918271132,
        83.40322434273706,15.44682671879188,3.907625855460164,
        77.20889976849125,15.572933924686586,3.9264753742795873,
        68.42784433266162,15.62494562148676,3.937982166883932,
        58.35924971869011,15.836915313271207,3.9714310413979503,
        85.7118187578991,15.143494897006256,3.9003656587928552
    };

    cudaMemcpyToSymbol(proportions_s, std6, sizeof(float) * N_LAYER * 3);
}
