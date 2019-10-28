#pragma once

#include <fstream>
#include <iostream>
#include "../render/block.cpp"

class BlockWriter {
    std::string filename;
    std::ofstream stream;
    size_t byte_count = 0;
    unsigned int element_count = 0;

  public:
    BlockWriter(std::string filename) : filename(filename) {
        stream = std::ofstream(filename, std::ios::binary);
    }

    ~BlockWriter() {
        std::cout << "File '" << filename << "' saved to disk.\n"
        << byte_count << " bytes.\n"
        << element_count << " elements.\n";
    }

    unsigned int pos() const {
        return element_count;
    }

    template <typename T>
    void operator<<(const T& node) {
        stream.write((char*) &node, sizeof(T));
        byte_count += sizeof(T);
        element_count += sizeof(T) / Block::ELEMENT_SIZE;
    }
};
