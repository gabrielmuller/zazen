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
    Block* block = new Block(11);
    Voxel* p = new (block->slot()) Voxel();

    p->child = block->size();
    new (block->slot()) Leaf(0x33, 0x22, 0x00, 0xff);
    Voxel* c = new (block->slot()) Voxel();
    Voxel* d = new (block->slot()) Voxel();

    c->child = block->size();
    new (block->slot()) Leaf(0xff, 0x88, 0x11, 0xff);
    new (block->slot()) Leaf(0xff, 0xff, 0x00, 0xff);


    d->child = block->size();
    new (block->slot()) Leaf(0x01, 0xff, 0x80, 0xff);
    Voxel* e = new (block->slot()) Voxel();

    e->child = block->size();
    new (block->slot()) Leaf(0xff, 0x33, 0x33, 0xff);
    new (block->slot()) Leaf(0xff, 0xaa, 0xaa, 0xff);
    new (block->slot()) Leaf(0xff, 0xff, 0xff, 0xff);

    p->valid = 0x8a;
    p->leaf = 0x02;

    c->valid = 0x82;
    c->leaf = 0x82;

    d->valid = 0x82;
    d->leaf = 0x02;

    e->valid = 0xa2;
    e->leaf = 0xa2;

    return block;
}

void save_model(Model* model) {
    Block block(100000000);
    ZStream stream(model);
    Builder builder(&block, stream.power);
    while(!builder.is_done()) builder.add_leaf(stream.next());
    block.to_file(model->name);
}

int main() {
    //save_model(bunny());
    //save_model(brain());
    save_model(generated());
    example()->to_file("example");
    return 0;
}
