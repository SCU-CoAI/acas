#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

// Global threshold storage for activation-only adaptive sparsity
struct SparsityThresholds {
    std::vector<float> activation_thresholds;
    // Per-layer predictor threshold - direct magnitude cutoff for proxy values
    std::vector<float> predictor_thresholds;
    
    SparsityThresholds() {}
    
    void resize(int n_layers) {
        // Start with small non-zero thresholds to trigger some sparsity
        // This allows L2 errors to be computed and thresholds to adapt
        activation_thresholds.resize(n_layers, 0.01f);
        // Default predictor threshold to 100.0
        predictor_thresholds.resize(n_layers, 200.0f);
    }
    
    bool load_from_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        std::string line;
        std::vector<float> loaded_activation;
        std::vector<float> loaded_predictor;
        
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;  // Skip empty lines and comments
            
            std::istringstream iss(line);
            float activation_thresh, predictor_thresh;
            
            if (iss >> activation_thresh) {
                loaded_activation.push_back(activation_thresh);
                
                // Try to read optional second column for predictor threshold
                if (iss >> predictor_thresh) {
                    loaded_predictor.push_back(predictor_thresh);
                } else {
                    // If no second column, use default predictor threshold
                    loaded_predictor.push_back(200.0f);
                }
            }
        }
        
        file.close();
        
        // Grow internal storage if needed and copy loaded values
        if (!loaded_activation.empty()) {
            if (activation_thresholds.size() < loaded_activation.size()) {
                activation_thresholds.resize(loaded_activation.size(), 0.01f);
                predictor_thresholds.resize(loaded_activation.size(), 50.0f);
            }
            for (size_t i = 0; i < loaded_activation.size(); ++i) {
                activation_thresholds[i] = loaded_activation[i];
                if (i < loaded_predictor.size()) {
                    predictor_thresholds[i] = loaded_predictor[i];
                }
            }
        }
        
        std::cerr << "Loaded thresholds for " << loaded_activation.size() << " layers from " << filename << std::endl;
        return !loaded_activation.empty();
    }
    
    void save_to_file(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open " << filename << " for writing" << std::endl;
            return;
        }
        
        file << "# Sparsity thresholds file" << std::endl;
        file << "# Format: activation_threshold predictor_threshold (one line per layer)" << std::endl;
        file << std::endl;
        
        for (size_t i = 0; i < activation_thresholds.size(); i++) {
            file << activation_thresholds[i] << " " << predictor_thresholds[i] << std::endl;
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

    // Predictor threshold (per-layer)
    float get_predictor_threshold(int layer) const {
        if (layer < predictor_thresholds.size()) {
            return predictor_thresholds[layer];
        }
        return 50.0f;
    }

    void set_predictor_threshold(int layer, float value) {
        if (layer < predictor_thresholds.size()) {
            predictor_thresholds[layer] = value;
        }
    }
};

// Global instance
extern SparsityThresholds g_sparsity_thresholds;
