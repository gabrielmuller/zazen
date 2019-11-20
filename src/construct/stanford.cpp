#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include "model.cpp"
#include "../render/leaf.cpp"
#include "sep.cpp"

const std::string base_path(std::string("..") + sep + std::string("models"));


struct StanfordModel : Model {
  public:
    uint16_t** data;
    StanfordModel(std::string name, std::string prefix,
            unsigned int width, unsigned int height, unsigned int depth) 
            : Model(name, width, height, depth) {
        data = read_model(name, prefix, height);
    }

    Leaf at(int3 pos) const override {
        pos.y = height - pos.y - 1;

        if (pos.x >= width || pos.y >= height || pos.z >= depth) {
            return Leaf();
        }

        uint16_t value = data[pos.y][pos.z * depth + pos.x];

        // quantize values
        value /= 256 * 16;
        value *= 16;

        if (value < 0xa0) value = 0;
        return Leaf(value, value, value, value);
    }

    ~StanfordModel() override {
        for (unsigned int i = 0; i < height; i++) {
            delete[] data[i];
        }
        delete[] data;
    }
        
  private:
    uint16_t* read_slice(std::string model_name,
                         std::string prefix,
                         unsigned int slice) {

        std::string path = base_path + sep + model_name + 
                           sep + prefix + std::to_string(slice);
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        std::streamsize size = file.tellg();
        if (size < 0) {
            std::cerr << "File '" << path << "' does not exist!\n";
            throw std::runtime_error("File does not exist.");
        }
        file.seekg(0, std::ios::beg);
        char* buffer = new char[size];
        if (file.read((char*) buffer, size)) {
            return (uint16_t*) buffer;
        }
        throw std::runtime_error("Error reading file " + path);
    }

    uint16_t** read_model
            (std::string model_name,
             std::string prefix,
             unsigned int num_slices) {
        uint16_t** buffer = new uint16_t*[num_slices];
        for (unsigned int i = 0; i < num_slices; i++) {
            buffer[i] = read_slice(model_name, prefix, i+1);
        }
        return buffer;
    }
};
