#version 430 core

uniform uvec2 viewportSize;
uniform vec3 camPos;
uniform float time;
uniform uint modelSize;
uniform mat3 camRot;
uniform float xSection;
in vec4 gl_FragCoord;

layout (location = 0) out vec4 outColor;
layout (location = 3) out vec4 outPosition;

layout (binding = 2, std430) buffer svo {
    uint[] data;
};

uint whichOctant(in vec3 pos, in vec3 corner, in float size) {
    /* Which of the eight octants does pos reside in the box (corner, size)? */
    uint oct = 0;
    float octSize = size / 2.;

    if (pos.x > corner.x + octSize) oct ^= 4;
    if (pos.y > corner.y + octSize) oct ^= 2;
    if (pos.z > corner.z + octSize) oct ^= 1;
    return oct;
}

vec3 octVec(in uint octant) {
    /* Each resulting vector component is either one or zero depending on 
     * octant's bits. */
    vec3 vec = vec3(0.);
    if (bool(octant & 4)) vec.x = 1.;
    if (bool(octant & 2)) vec.y = 1.;
    if (bool(octant & 1)) vec.z = 1.;
    return vec;
}

void voxel(uint block_i, out uint child, out uint leaf, out uint valid) {
    child = data[block_i];
    leaf = data[block_i + 1] >> 8;
    valid = data[block_i + 1] & 0xff;
}

vec4 leafColor(in uint block_i) {
    uint rgba = data[block_i];
    return vec4(
        float(rgba & 0xff) / 256.,
        float((rgba >> 8) & 0xff) / 256.,
        float((rgba >> 16) & 0xff) / 256.,
        float((rgba >> 24) & 0xff) / 256.
    );
}

uint addressOf(in uint child, in uint leaf, in uint valid, in uint octant) {
    /* Get address of a specific child inside a voxel. */
    uint mask = ~(~0 << octant);
    return child + bitCount(mask & valid) * 2 - bitCount(mask & leaf);
}
    
void main() {
    /* Initialize ray. */
    float fov = 0.6;
    vec2 uv = (gl_FragCoord.xy * 2. / viewportSize.y)  - vec2(1.);
    vec3 direction = camRot * vec3(uv * fov, 1.);
    vec3 position = camPos;

    /* Set up stack. */
    uint size = 1; // size of stack
    float boxSize = 2.0; // size of box ray is currently in

    /* Stack attributes */
    uint child[10];
    uint leaf[10];
    uint valid[10];
    vec3 corner[10];
    uint octant[10];

    /* Set root stack entry */
    voxel(modelSize - 2, child[0], leaf[0], valid[0]);
    corner[0] = vec3(-1);

    /* Assume ray direction does not change during execution 
     * (no refraction / reflection) 
     */
    uint mask = 0;
    vec3 invdir = 1. / direction;
    if (invdir.x >= 0) mask ^= 4;
    if (invdir.y >= 0) mask ^= 2;
    if (invdir.z >= 0) mask ^= 1;
    vec3 maskVec = octVec(mask);
    vec3 mirror = vec3(1.) - maskVec * 2.;

    float cutSize = boxSize * xSection;
    if (any(lessThan(position, corner[0]))
     || any(greaterThan(position, corner[0] + vec3(cutSize)))) {
        /* Before the first iteration, if the camera is outside the scenes's
         * bounding box, the traversal starts at the intersection between the
         * ray and the box.
         */
        vec3 lowCorner = corner[0] + cutSize * (vec3(1.) - maskVec);
        vec3 highCorner = corner[0] + cutSize * maskVec;
        vec3 tMin = (lowCorner  - position) * invdir;
        vec3 tMax = (highCorner - position) * invdir;

        if (any(greaterThan(tMin, tMax.zxy)) 
         || any(greaterThan(tMin, tMax.yzx))) {
            /* No intersection */
            outColor = vec4(0.);
            outPosition = vec4(0.);
            return;
        }

        float t = max(max(tMin.x, tMin.y), tMin.z);
        position += direction * t;

        if (t < 0.) {
            outColor = vec4(0.);
            outPosition = vec4(0.);
            return;
        }
    }

    octant[0] = whichOctant(position, vec3(-1), boxSize);

    while (true) {
        uint oct = octant[size-1];
        if (bool((leaf[size-1] >> oct) & 1)) {
            /* Ray origin is inside leaf voxel, render leaf. */
            outColor = leafColor(
                addressOf(
                    child[size-1],
                    leaf [size-1],
                    valid[size-1],
                    oct
                )
            );
            outPosition = vec4(position, 1.0);
            return;
        } 

        if (bool((valid[size-1] >> oct) & 1)) {
            /* Go a level deeper. */

            // PUSH
            voxel(
                addressOf(
                    child[size-1],
                    leaf [size-1],
                    valid[size-1],
                    oct
                ),
                child[size],
                leaf[size],
                valid[size]
            );

            boxSize *= 0.5;
            corner[size] = corner[size-1] + boxSize * octVec(oct);
            octant[size] = whichOctant(
                position,
                corner[size],
                boxSize
            );

            size++;

        } else {
            /* Ray origin is in invalid voxel, cast ray until it hits the next
             * voxel. 
             */

            float childSize = boxSize * 0.5;
            vec3 childCorner = corner[size-1] + childSize * octVec(oct);

            vec3 adjustedCorner = childCorner - maskVec * childSize * mirror;

            vec3 t = (adjustedCorner - position) * invdir;
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

            while (
                bool(hitFace & ~(octant[size-1] ^ mask)) && 
                size > 0
            ) {
                /* Hit face is at this voxel's boundary, search parent */

                // POP
                size--;
                boxSize *= 2.;
            }

            if (size == 0) {
                /* Ray is outside root octree. */
                outColor = vec4(0.0);
                outPosition = vec4(direction, 0.);
                return;
            }
            /* Loop end: found ancestral voxel with space on the hit axis.
             * Transfer to sibling voxel, changing on the axis of the face
             * that was hit.
             */
            octant[size-1] ^= hitFace;
        }
    }
}
