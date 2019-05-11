#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include "model.cpp"
#include "../render/leaf.cpp"
#include "../render/sep.cpp"

const std::string base_path(std::string("..") + sep + std::string("models"));


struct StanfordModel : Model {
  public:
    uint16_t** data;
    StanfordModel(std::string name, 
            unsigned int width, unsigned int height, unsigned int depth) 
            : Model(name, width, height, depth) {
        data = read_model(name, depth);
    }

    Leaf get(unsigned int x, unsigned int y, unsigned int z) const override {
        uint16_t value = data[z][y * width + x];
        // quantize values
        value /= 256 * 32;
        value *= 32;
        unsigned int color_x = x * 256 / width;
        unsigned int color_y = y * 256 / height;
        return Leaf(value, value/4 + color_x / 2, value/8 + color_y / 2, value);
    }

    ~StanfordModel() override {
        for (int i = 0; i < depth; i++) {
            delete[] data[i];
        }
        delete[] data;
    }
        
  private:
    uint16_t* read_slice(std::string model_name, unsigned int slice) {
        std::string path = base_path + sep + model_name + 
                           sep + std::to_string(slice);
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        char* buffer = new char[size];
        if (file.read((char*) buffer, size)) {
            return (uint16_t*) buffer;
        }
        throw std::runtime_error("Error reading file " + path);
    }

    uint16_t** read_model
            (std::string model_name, unsigned int num_slices) {
        uint16_t** buffer = new uint16_t*[num_slices];
        for (int i = 0; i < num_slices; i++) {
            buffer[i] = read_slice(model_name, i+1);
        }
        return buffer;
    }
};
