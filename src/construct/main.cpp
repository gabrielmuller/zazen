#include "construct.cpp"
#include "zstream.cpp"
#include "stanford.cpp"
#include "generate.cpp"
#include "../render/block.cpp"

StanfordModel* bunny() {
    return new StanfordModel("bunny", "", 512, 316, 512);
}

StanfordModel* brain() {
    return new StanfordModel("brain", "MRbrain.", 256, 109, 256);
}

GenerateModel* generated() {
    const unsigned int size = 512;
    return new GenerateModel("generated", size, size, size);
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
    //GenerateModel* gen = generated();
    StanfordModel* bunny_model = bunny();
    //save_model(gen);
    save_model(bunny_model);
    //delete gen;
    delete bunny_model;
    return 0;
}
