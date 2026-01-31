#version 450
layout(vertices = 4) out;

layout(location = 0) in vec2 in_uv[];
layout(location = 1) in flat uint in_layer[];

layout(location = 0) out vec2 out_uv[];
layout(location = 1) out flat uint out_layer[];

layout(set = 0, binding = 0) uniform Camera {
    mat4 view;
    mat4 proj;
    vec3 pos;
} ubo;

void main() {
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    out_uv[gl_InvocationID] = in_uv[gl_InvocationID];
    out_layer[gl_InvocationID] = in_layer[gl_InvocationID];

    if (gl_InvocationID == 0) {
        // SIMPLIFIED: gl_in is already in World Space!
        // Just average the 4 corners to find the patch center
        vec4 center = (gl_in[0].gl_Position + gl_in[1].gl_Position + 
                gl_in[2].gl_Position + gl_in[3].gl_Position) / 4.0;

        float dist = distance(ubo.pos, center.xyz);

        // LOD Calculation (Same logic)
        float level = mix(64.0, 2.0, clamp((dist - 200.0) / 1000.0, 0.0, 1.0));
        level = 64.; // cpu does enough LODing, this should work too

        gl_TessLevelOuter[0] = level;
        gl_TessLevelOuter[1] = level;
        gl_TessLevelOuter[2] = level;
        gl_TessLevelOuter[3] = level;

        gl_TessLevelInner[0] = level;
        gl_TessLevelInner[1] = level;
    }
}
