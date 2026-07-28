#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

// Global threshold storage for activation-only adaptive sparsity
struct SparsityThresholds {
    std::vector<float> activation_thresholds;
    
    SparsityThresholds() {}
    
    void resize(int n_layers) {
        // Start with small non-zero thresholds to trigger some sparsity
        // This allows L2 errors to be computed and thresholds to adapt
        activation_thresholds.resize(n_layers, 0.01f);
    }
    
    bool load_from_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        std::string line;
        std::vector<float> loaded;
        
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;  // Skip empty lines and comments
            
            std::istringstream iss(line);
            float activation_thresh;
            if (iss >> activation_thresh) {
                loaded.push_back(activation_thresh);
            }
        }
        
        file.close();
        
        // Grow internal storage if needed and copy loaded values
        if (!loaded.empty()) {
            if (activation_thresholds.size() < loaded.size()) {
                activation_thresholds.resize(loaded.size(), 0.01f);
            }
            for (size_t i = 0; i < loaded.size(); ++i) {
                activation_thresholds[i] = loaded[i];
            }
        }
        
        std::cerr << "Loaded thresholds for " << loaded.size() << " layers from " << filename << std::endl;
        return !loaded.empty();
    }
    
    void save_to_file(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open " << filename << " for writing" << std::endl;
            return;
        }
        
        file << "# Activation sparsity thresholds file" << std::endl;
        file << "# Format: activation_threshold (one line per layer)" << std::endl;
        file << std::endl;
        
        for (size_t i = 0; i < activation_thresholds.size(); i++) {
            file << activation_thresholds[i] << std::endl;
        }
        
        file.close();
        const char *log = std::getenv("ACAS_LOG");
        if (log && std::atoi(log) != 0) {
            std::cerr << "Saved thresholds for " << activation_thresholds.size() << " layers to " << filename << std::endl;
        }
    }
    
    float get_activation_threshold(int layer) const {
        if (layer < activation_thresholds.size()) {
            return activation_thresholds[layer];
        }
        return 0.0f;
    }
    
    void set_activation_threshold(int layer, float value) {
        if (layer < activation_thresholds.size()) {
            activation_thresholds[layer] = value;
        }
    }
};

// Global instance
extern SparsityThresholds g_sparsity_thresholds;
