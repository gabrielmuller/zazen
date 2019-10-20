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


struct Voxel {
    uint child;
    uint leaf;
    uint valid;
};

struct StackEntry {
    Voxel voxel;
    uint octant;
    vec3 corner;
};

struct Stack {
    StackEntry data[20];
    uint size;
    float boxSize;
};

StackEntry top(in Stack stack) {
    return stack.data[stack.size - 1];
}

uint whichOctant(in vec3 pos, in vec3 corner, in float size) {
    /* Which of the eight octants does pos reside in the box (corner, size)? */
    uint oct = 0;
    float octSize = size / 2.;

    if (pos.x > corner.x + octSize) oct ^= 4;
    if (pos.y > corner.y + octSize) oct ^= 2;
    if (pos.z > corner.z + octSize) oct ^= 1;
    return oct;
}

vec3 adjustCorner(in vec3 corner, float size, uint octant) {
    vec3 adjusted = corner;
    if (octant & 4) adjusted.x += size;
    if (octant & 2) adjusted.y += size;
    if (octant & 1) adjusted.z += size;
    return adjusted;
}

Voxel voxel(in uint block_i) {
    uint i = block_i * 2;
    return Voxel(data[i++], data[i] >> 8, data[i] & 0xff);
}

vec4 leaf(in uint block_i) {
    uint rgba = data[block_i * 2];
    return vec4(
        (rgba & 0xff) / 256.,
        ((rgba >> 8) & 0xff) / 256.,
        ((rgba >> 16) & 0xff) / 256.,
        ((rgba >> 24) & 0xff) / 256.
    );
}

uint address_of(in Voxel voxel, uint octant) {
    /* Get address of a specific child inside a voxel. */
    uint mask = ~(0xffffffff << octant);
    return voxel.child + bitCount(mask & voxel.valid);
}
    
void main() {
    /* Set up stack. */
    Stack stack;
    stack.size = 1;
    stack.boxSize = 2.0;
    stack.data[0] = StackEntry(
        voxel(modelSize - 1),
        0,
        vec3(-1)
    );

    /* Initialize ray. */
    float fov = 1.2;
    vec2 uv = (gl_FragCoord.xy * fov / viewportSize) - vec2(0.5);
    vec3 direction = vec3(uv.x * 1.2, -1.0, uv.y * 1.2);
    vec3 position = camPos;

    /* Assume ray direction does not change (no refraction / reflection) */
    vec3 mirror;
    uint mask = 0;
    if (direction.x >= 0) mirror.x = -1, mask ^= 4;
    if (direction.y >= 0) mirror.y = -1, mask ^= 2;
    if (direction.z >= 0) mirror.z = -1, mask ^= 1;

    float dist = 0.;
    vec3 color = vec3(0.3, 0., 0.);
    uint i = 0;

    for (; i < 100; i++) { // prevent infinite loop
        StackEntry entry = top(stack);
        uint oct = entry.octant;
        bool isValid = bool((entry.voxel.valid >> oct) & 1);
        bool isLeaf = bool((entry.voxel.leaf >> oct) & 1);

        if (isLeaf) {
            /* Ray origin is inside leaf voxel, render leaf. */
            color = leaf(address_of(entry.voxel, oct)).xyz / (dist*dist+1.);
            break;
        } 

        if (isValid) {
            /* Go a level deeper. */
            stack.data[stack.size].voxel = voxel(address_of(entry.voxel, oct));
            stack.boxSize *= 0.5;
            stack.data[stack.size].corner = adjustCorner(
                entry.corner,
                stack.boxSize,
                entry.octant
            );
            stack.data[stack.size].octant = whichOctant(
                position,
                stack.data[stack.size].corner, 
                stack.boxSize
            );

            // PUSH
            stack.size++;

            color.y += (1 - color.y) / 60.;
        } else {
            /* Ray origin is in invalid voxel, cast ray until it hits next
             * voxel. 
             */
            vec3 childCorner = top(stack).corner;
            float childSize = stack.boxSize * 0.5;
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

            if (hitFace == 0) {
                color = vec3(uv, 1.0);
                break;
            }

            uint counter = 0;
            while (
                bool(hitFace & ~(top(stack).octant ^ mask)) && 
                stack.size > 0
            ) {
                counter++;
                /* Hit face is at this voxel's boundary, search parent */

                // POP
                stack.size--;
                stack.boxSize *= 2.;
            }

            if (stack.size == 0) {
                /* Ray is outside root octree. */

                // EMPTY STACK AFTER LOOP - BLUE
                color.z = 0.8;
                break;
            }
            /* Loop end: found ancestral voxel with space on the hit axis.
             * Transfer to sibling voxel, changing on the axis of the face
             * that was hit.
             */
            stack.data[stack.size].octant ^= hitFace;
        }
    }
    outColor = vec4(color, 1.0);
}
