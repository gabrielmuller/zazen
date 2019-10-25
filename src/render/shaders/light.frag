#version 430 core

out vec4 outColor;
in vec4 gl_FragCoord;
uniform uvec2 viewportSize;
uniform uint upscale;
layout (binding = 0) uniform sampler2D colorTexture;
layout (binding = 3) uniform sampler2D positionTexture;

void main() {
    vec2 verts[3];
    verts[0] = gl_FragCoord.xy / float(upscale);


    bool lowerTri = fract(verts[0]).x > fract(verts[0]).y;
    vec2 triOrientation = lowerTri ? vec2(1., 0.) : vec2(0., 1.);

    verts[lowerTri?1:2] = verts[0] + vec2(1.);
    verts[lowerTri?2:1] = verts[0] + triOrientation;

    vec4 colors[3];
    vec4 pos[3];

    for (uint i = 0; i < 3; i++) {
        verts[i] = (verts[i] * float(upscale)) / viewportSize;
        colors[i] = texture(colorTexture, verts[i]);
        pos[i] = texture(positionTexture, verts[i]);
        if (pos[i].w == 0.0) {
            outColor = vec4(0.8, 0.9, 1.0, 1.0);
            return;
        }
    }

    vec3 normal = cross(
        pos[2].xyz - pos[0].xyz,
        pos[1].xyz - pos[0].xyz
        );
    normal = normalize(normal);

    vec3 sun = normalize(pos[0].xyz);
    float dist = length(pos[0].xyz);
    float lightness = dot(sun, normal) * 0.8 * 1./(dist*dist) + 0.2;

    outColor = colors[0] * 0.5 + colors[1] * 0.25 + colors[2] * 0.25 * lightness;
}
