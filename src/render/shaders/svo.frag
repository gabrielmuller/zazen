#version 150 core

uniform vec2 viewportSize;
uniform float time;
in vec4 gl_FragCoord;

out vec4 outColor;

void main() {
    vec2 coords = gl_FragCoord.xy / viewportSize;
    outColor = vec4(coords.xy, abs(sin(time)), 1.0);
}
