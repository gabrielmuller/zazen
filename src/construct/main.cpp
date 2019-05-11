#include "construct.cpp"
#include "stanford.cpp"

int main() {
    Block* block = new Block(4700000);
    Model* model = bunny();
    construct(model, block);
    block->to_file(model->name);
    return 0;
}
