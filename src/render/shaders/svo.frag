#version 430 core

uniform uvec2 viewportSize;
uniform vec3 camPos;
uniform float time;
uniform uint modelSize;
in vec4 gl_FragCoord;

out vec4 outColor;

layout (binding = 2, std430) buffer svo {
    uint[] data;
};


uint whichOctant(in vec3 pos, in vec3 corner, in float size) {
    /* Which of the eight stack_octs does pos reside in the box (corner, size)? */
    uint oct = 0;
    float octSize = size / 2.;

    if (pos.x > corner.x + octSize) oct ^= 4;
    if (pos.y > corner.y + octSize) oct ^= 2;
    if (pos.z > corner.z + octSize) oct ^= 1;
    return oct;
}

vec3 adjustCorner(in vec3 corner, float size, uint octant) {
    vec3 adjusted = corner;
    if (bool(octant & 4)) adjusted.x += size;
    if (bool(octant & 2)) adjusted.y += size;
    if (bool(octant & 1)) adjusted.z += size;
    return adjusted;
}

void voxel(in uint block_i, out uint child, out uint leaf, out uint valid) {
    uint i = block_i * 2;
    child = data[i++];
    leaf = data[i] >> 8;
    valid = data[i] & 0xff;
}

vec4 getLeaf(in uint block_i) {
    uint rgba = data[block_i * 2];
    return vec4(
        float(rgba & 0xff) / 256.,
        float((rgba >> 8) & 0xff) / 256.,
        float((rgba >> 16) & 0xff) / 256.,
        float((rgba >> 24) & 0xff) / 256.
    );
}

uint addressOf(in uint child, in uint valid, in uint octant) {
    /* Get address of a specific child inside a voxel. */
    uint mask = ~(0xffffffff << octant);
    return child + bitCount(mask & valid);
}
    
void main() {
    /* Initialize ray. */
    float fov = 0.6;
    vec2 uv = (gl_FragCoord.xy * 2. / viewportSize.y)  - vec2(1.);
    vec3 direction = vec3(uv * fov, -1.);
    vec3 position = camPos;

    /* Set up stack_ */
    uint stack_size = 1;
    float stack_boxSize = 2.0;
    uint child[10];
    uint leaf[10];
    uint valid[10];
    vec3 stack_corner[10];
    uint stack_octs[10];
    voxel(modelSize - 1, child[0], leaf[0], valid[0]);
    stack_corner[0] = vec3(-1);
    stack_octs[0] = whichOctant(position, vec3(-1), stack_boxSize);



    /* Assume ray direction does not change (no refraction / reflection) */
    vec3 mirror = vec3(1.);
    uint mask = 0;
    if (direction.x >= 0) mirror.x = -1, mask ^= 4;
    if (direction.y >= 0) mirror.y = -1, mask ^= 2;
    if (direction.z >= 0) mirror.z = -1, mask ^= 1;

    float dist = 0.;
    vec3 color = vec3(0.);

    while (true) {
        uint i = stack_size - 1;
        uint oct = stack_octs[i];
        bool isValid = bool((valid[i] >> oct) & 1);
        bool isLeaf = bool((leaf[i] >> oct) & 1);

        if (isLeaf) {
            /* Ray origin is inside leaf voxel, render leaf. */
            color = getLeaf(addressOf(child[i], valid[i], oct)).xyz / (dist*dist+1.);
            break;
        } 

        if (isValid) {
            /* Go a level deeper. */
            voxel(addressOf(child[i], valid[i], oct), child[stack_size], leaf[stack_size], valid[stack_size]);
            stack_boxSize *= 0.5;
            stack_corner[stack_size] = adjustCorner(
                stack_corner[i],
                stack_boxSize,
                oct
            );
            stack_octs[stack_size] = whichOctant(
                position,
                stack_corner[stack_size],
                stack_boxSize
            );

            // PUSH
            stack_size++;

            color.z += (1. - color.z) / 20.;
        } else {
            /* Ray origin is in invalid voxel, cast ray until it hits the next
             * voxel. 
             */
            vec3 childCorner = stack_corner[stack_size - 1];
            float childSize = stack_boxSize * 0.5;
            childCorner = adjustCorner(childCorner, childSize, oct);

            vec3 mirrorPos = position * mirror;
            vec3 mirrorDir = direction * mirror;
            vec3 mirrorCorner = adjustCorner(
                childCorner * mirror,
                -childSize,
                mask
            );

            vec3 t = (mirrorCorner - mirrorPos) / mirrorDir;
            float amount = 99999999999.; // Distance ray will traverse

            /* Detect which face hit. */
            uint hitFace = 0;

            /* amount will be the minimum positive component of t. */
            if (t.x > 0.) {
                amount = t.x;
                hitFace = 4;
            }
            if (t.y > 0. && t.y < amount) {
                amount = t.y;
                hitFace = 2;
            }
            if (t.z > 0. && t.z < amount) {
                amount = t.z;
                hitFace = 1;
            }

            /* Ray will start next step at the point of this intersection. */
            position += direction * amount;
            dist += amount;

            while (
                bool(hitFace & ~(stack_octs[stack_size-1] ^ mask)) && 
                stack_size > 0
            ) {
                /* Hit face is at this voxel's boundary, search parent */

                // POP
                stack_size--;
                stack_boxSize *= 2.;
            }

            if (stack_size == 0) {
                /* Ray is outside root octree. */
                color.xy += vec2(0.8) + position.xy * 0.2;
                break;
            }
            /* Loop end: found ancestral voxel with space on the hit axis.
             * Transfer to sibling voxel, changing on the axis of the face
             * that was hit.
             */
            stack_octs[stack_size - 1] ^= hitFace;
        }
    }
    outColor = vec4(color, 1.0);
}
