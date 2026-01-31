#version 450
// ... inputs ...
layout(location = 0) in float height_factor;
layout(location = 1) in vec3 v_world_pos;
layout(location = 2) in vec3 v_normal;
layout(location = 3) in flat uint in_layer; // <--- You need to pass the layer from TES to FS!
layout(location = 4) in vec2 in_uv;

layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 2) uniform sampler2DArray colorMap;

const vec3 LIGHT_DIR = normalize(vec3(0.5, 0.8, 0.5));

void main() {
    // ... normal calc ...
    vec3 normal = normalize(v_normal);
    float NdotL = max(dot(normal, LIGHT_DIR), 0.0);
    float light = NdotL + 0.8; 

    // SAMPLING
    // We need UVs here. You might need to pass `in_uv` from TES to FS if you haven't already.
    // Assuming you add `layout(location = 4) in vec2 in_uv;`
    vec3 albedo = texture(colorMap, vec3(in_uv, float(in_layer))).rgb;

    f_color = vec4(albedo * light, 1.0);
}
