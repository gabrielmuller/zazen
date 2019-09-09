    

#include "int3.cpp"
#include "indexedleaf.cpp"
#include "model.cpp"
#include <stack>
#include <cstdint>
#include <algorithm>
#include <cmath>

void f(int size, int3 offset) {
    if (size == 1) {
        std::cout << offset.x << " " << offset.y << " " << offset.z << "\n";
        return;
    }
    size /= 2;
    for (uint8_t i = 0; i < 8; i++) {
        int3 o(offset);
        if (i & 4) o.x += size;
        if (i & 2) o.y += size;
        if (i & 1) o.z += size;
        f(size, o);
    }
}

struct ZFrame {
    const unsigned int size;
    const int3 offset;
    uint8_t i;

    ZFrame(const unsigned int size, const int3 offset, uint8_t i) 
            : size(size), offset(offset), i(i) {}
};

class ZStream {
    const Model* model;
    unsigned int index;
    bool _open;
    std::stack<ZFrame> stack;

    int3 next_coords() {
        if (stack.top().size == 1) {
            int3 ret(stack.top().offset);
            stack.pop();
            while (!stack.empty() && stack.top().i >= 8) stack.pop();
            if (stack.empty()) _open = false;
            index++;
            return ret;
        }
        const unsigned int size = stack.top().size / 2;
        int3 offset(stack.top().offset);
        if (stack.top().i & 4) offset.x += size;
        if (stack.top().i & 2) offset.y += size;
        if (stack.top().i & 1) offset.z += size;
        stack.top().i++;
        stack.push(ZFrame(size, offset, 0));
        return next_coords();
    }

  public:
    const unsigned int power;
    const unsigned int stream_size;
    ZStream(Model* model) : model(model), index(0), _open(true), 
            power(std::log2(std::max({model->width, model->height, model->depth}))),
            stream_size(std::pow(2, power)) {
        stack.push(ZFrame(1 << power, int3(0, 0, 0), 0));
    }

    inline bool is_open() { return _open; }

    inline IndexedLeaf next() {
        Leaf leaf;
        do leaf = model->at(next_coords());
        while (!leaf.valid() && is_open());
        return IndexedLeaf(leaf, index);
    }
};
