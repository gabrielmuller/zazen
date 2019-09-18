#version 150 core
out vec4 outColor;

void main() {
    for (int i = 0; i < 5; i++) {
        outColor = vec4(1.0, 0.7, 1.0, 1.0);
    }
}
