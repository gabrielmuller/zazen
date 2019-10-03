#version 430 core

uniform uvec2 viewportSize;
uniform float time;
uniform sampler1D svoData;
in vec4 gl_FragCoord;

out vec4 outColor;

layout (binding = 2, std430) buffer svo {
    uint[] data;
};

void main() {
    uint i = uint(gl_FragCoord.y + time*time*100) * viewportSize.x + uint(gl_FragCoord.x);
    uint icol = data[i];
    vec4 color = vec4(
        icol & 0xff,
        (icol >> 8) & 0xff,
        (icol >> 16) & 0xff,
        (icol >> 24) & 0xff
    );
    color /= 0xff;
    outColor = color;
}
