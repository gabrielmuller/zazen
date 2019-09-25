#version 150 core

uniform vec2 viewportSize;
uniform float time;
uniform sampler1D svoData;
in vec4 gl_FragCoord;

out vec4 outColor;

void main() {
    vec2 coords = gl_FragCoord.xy / viewportSize;
    outColor = vec4(texelFetch(svoData, 0, 0));
}
