#pragma once
#include <iostream>

struct Block {
  private:
    static const std::size_t element_size = 8;
    const size_t element_count;
    char* data = nullptr;
    char* front = nullptr;
    size_t front_index = 0;

  public:
    explicit Block(size_t element_count, char* data) :
        element_count(element_count),
        data(data) {
            front = data;
        }

    explicit Block(size_t element_count) : element_count(element_count) {
        data = new char[element_count * element_size];
        front = data;
    }

    ~Block() {
        delete[] data;
    }

    Block(Block&) = delete; // No copy constructor.
    Block& operator=(Block&) = delete; // No assigning.
    Block(Block&& rhs) = delete; // No move constructor.
    Block& operator=(Block&& rhs) = delete; // No move assignment operator.

    template <class T>
    T& get(const int32_t index) const {
        if (index > front_index || index < 0) {
            std::cout << "index " << index << "\n";
            throw "Out of bounds access in block";
        }
        return ((T*) data)[index];
    }

    char* slot() {
        char* front_slot = front;
        front += element_size;
        front_index++;
        return front_slot;
    }

    size_t size() {
        return front_index;
    }

    size_t capacity() {
        return element_count;
    }
};
