#pragma once
#include <iostream>
#include <fstream>

struct Block {
  private:
    static const std::size_t ELEMENT_SIZE = 8;
    const size_t element_count;
    char* data = nullptr;
    char* front = nullptr;
    size_t front_index = 0;

  public:
    static const std::string EXTENSION;

    explicit Block(size_t element_count) :
            element_count(element_count) {
        data = new char[element_count * ELEMENT_SIZE];
        front = data;
    }

    ~Block() {
        delete[] data;
    }

    inline void read_from_stream(std::ifstream& stream, size_t count) {
        stream.read(front, count * ELEMENT_SIZE);
    }

    void to_file(std::string filename) {
        filename += EXTENSION;
        std::ofstream stream(filename, std::ios::binary);
        stream.write((char*) &front_index, sizeof(size_t));
        stream.write(data, size() * ELEMENT_SIZE);
        std::cout << "File \"" << filename << "\" saved to disk.\n";
    }

    template <class T>
    T& get(const int32_t index) const {
        return ((T*) data)[index];
    }

    char* slot() {
        char* front_slot = front;
        front += ELEMENT_SIZE;
        front_index++;
        return front_slot;
    }

    inline size_t size() const {
        return front_index;
    }

    inline size_t capacity() const {
        return element_count;
    }

};

const std::string Block::EXTENSION = ".zaz";

Block* from_file(std::string filename) {
    std::ifstream stream(filename, std::ios::binary);

    char length_buffer[sizeof(size_t)];
    stream.read(length_buffer, sizeof(size_t));

    size_t count = *((size_t*) length_buffer);
    Block* block = new Block(count);
    block->read_from_stream(stream, count);

    std::cout << "File \"" << filename << "\" loaded from disk.\n";
    return block;
}
