    

#include "int3.cpp"
#include <stack>
#include <cstdint>
#include <algorithm>

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

struct IndexedLeaf {
    const Leaf leaf;
    const unsigned int index;
    IndexedLeaf(const Leaf leaf, const unsigned int index) : leaf(leaf), index(index) {}
};

class ZStream {
    const Model* model;
    std::stack<ZFrame> stack;
    bool _open;
    unsigned int index;

    int3 next_coords() {
        if (stack.top().size == 1) {
            int3 ret(stack.top().offset);
            stack.pop();
            
            while (!stack.empty() && stack.top().i >= 8) {
                stack.pop();
            }

            if (stack.empty()) {
                _open = false;
                return ret;
            }

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
    ZStream(Model* model) : model(model), index(0), _open(true) {
        const unsigned int power = std::log2(std::max({model->width, model->height, model->depth}));
        stack.push(ZFrame(1 << power, int3(0, 0, 0), 0));
    }

    inline bool is_open() { return _open; }

    inline IndexedLeaf next() {
        Leaf leaf;
        do leaf = model->at(next_coords()); while (!leaf.valid());
        return IndexedLeaf(leaf, index);
    }
};
