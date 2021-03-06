#pragma once
#include <iostream>
#include <fstream>

struct Block {
  private:
    const size_t element_count;
    char* _data = nullptr;
    char* front = nullptr;
    size_t front_index;

  public:
    static const std::size_t ELEMENT_SIZE = 4;
    static const std::string EXTENSION;

    Block(size_t element_count, bool full = false) :
            element_count(element_count) {
        const size_t size = element_count * ELEMENT_SIZE;
        _data = new char[size];
        if (full) { 
            front = _data + size;
            front_index = element_count;
        } else {
            front = _data;
            front_index = 0;
        }
    }

    ~Block() {
        delete[] _data;
    }

    inline void read_from_stream(std::ifstream& stream, std::streamsize size) {
        stream.read(_data, size);
    }

    void to_file(std::string filename) {
        filename += EXTENSION;
        std::ofstream stream(filename, std::ios::binary);
        if (stream.good()) {
            stream.write(_data, size() * ELEMENT_SIZE);
            stream.write((char*) &front_index, sizeof(size_t));
            std::cout << "File '" << filename << "' saved to disk.\n";
        } else {
            std::cout << "Could not open file '" << filename << ".\n";
        }
    }

    template <typename T>
    T& at(const int32_t index) const {
        return *((T*) (_data + index * ELEMENT_SIZE));
    }

    template <typename T>
    inline T& back() const {
        return at<T>(size() - 1);
    }

    inline size_t size() const {
        return front_index;
    }

    inline size_t byte_size() const {
        return front_index * ELEMENT_SIZE;
    }

    inline size_t capacity() const {
        return element_count;
    }

    inline char* data() const {
        return _data;
    }

};

const std::string Block::EXTENSION = ".zaz";

Block* from_file(std::string filename) {
    std::ifstream stream(filename, std::ios::binary | std::ios::ate);

    std::streamsize size = stream.tellg();
    stream.seekg(0, std::ios::beg);

    Block* block = new Block(size / Block::ELEMENT_SIZE, true);
    block->read_from_stream(stream, size);

    std::cout << "File \"" << filename << "\" loaded from disk.\n";
    return block;
}
