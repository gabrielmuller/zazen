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
    return new GenerateModel("generated", 256, 256, 256);
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
    Block block(100000000); // TODO: dynamic allocation
    ZStream stream(model);
    Builder builder(&block, stream.power);
    while(!builder.is_done() && stream.is_open()) builder.add_leaf(stream.next());
    block.to_file(model->name);
}

int main() {
    //save_model(bunny());
    //save_model(brain());
    save_model(generated());
    example()->to_file("example");
    return 0;
}
