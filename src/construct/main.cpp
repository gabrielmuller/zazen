#include "construct.cpp"
#include "zstream.cpp"
#include "stanford.cpp"
#include "generate.cpp"
#include "../render/block.cpp"

StanfordModel* bunny() {
    return new StanfordModel("bunny", "", 512, 512, 316);
}

StanfordModel* brain() {
    return new StanfordModel("brain", "MRbrain.", 256, 256, 109);
}

GenerateModel* generated() {
    const unsigned int size = 729;
    return new GenerateModel("generated", size, size, size);
}

Block* example() {
    Block* block = new Block(5);

    auto e_child = block->size();

    new (block->slot()) Leaf(0xff, 0xff, 0xff, 0xff);
    new (block->slot()) Leaf(0xff, 0xaa, 0xaa, 0xff);
    new (block->slot()) Leaf(0xff, 0x33, 0x33, 0xff);

    auto p_child = block->size();
    Voxel* e = new (block->slot()) Voxel();
    e->child = e_child;
    e->valid = e->leaf = 0xa2;

    Voxel* p = new (block->slot()) Voxel();
    p->child = p_child;
    p->valid = 0x80;
    p->leaf = 0x00;

    return block;
}

void save_model(Model* model) {
    ZStream stream(model);
    BlockWriter writer(model->name + ".zaz");
    Builder builder(stream.power, writer);
    unsigned int counter = 0;
    while(stream.is_open()) {
        builder.add_leaf(stream.next());
        if (!(counter % 10000)) {
            std::cout << "\r" << (int) (stream.progress() * 100) << "%"
            << std::flush;
        }
        counter++;
    }
    std::cout << "\r";
}

int main() {
    Block* example_block = example();
    GenerateModel* gen = generated();
    //StanfordModel* bunny_model = bunny();
    save_model(gen);
    //save_model(bunny_model);
    example_block->to_file("example");
    delete example_block;
    delete gen;
    //delete bunny_model;
    return 0;
}
