#version 430 core

out vec4 outColor;
in vec4 gl_FragCoord;
uniform uvec2 viewportSize;
uniform uint upscale;
uniform sampler2D colorTexture;

void main() {
    outColor = vec4(vec3( 
            texture(
                colorTexture,
                gl_FragCoord.xy / viewportSize
            ) + 
            texture(
                colorTexture,
                (gl_FragCoord.xy + vec2(10.)) / viewportSize
            )) * 0.5, 1.0);
}
