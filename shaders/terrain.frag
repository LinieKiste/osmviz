#version 450
layout(location = 0) in float height_factor;
layout(location = 1) in vec3 v_world_pos;
layout(location = 2) in vec3 v_normal;
layout(location = 3) in flat uint in_layer;
layout(location = 4) in vec2 in_uv;

layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 1) uniform sampler2DArray heightMap;
layout(set = 0, binding = 2) uniform sampler2DArray colorMap;

const vec3 LIGHT_DIR = normalize(vec3(0.5, 0.8, 0.5));

float get_height(vec2 uv, uint layer) {
    vec3 rgb = texture(heightMap, vec3(uv, layer)).rgb;

    float r = rgb.r * 255.0;
    float g = rgb.g * 255.0;
    float b = rgb.b * 255.0;

    float height_meters = -10000.0 + ((r * 65536.0 + g * 256.0 + b) * 0.1);

    return height_meters / 1000.0;
}

void main() {
    // ... normal calc ...
    vec3 normal = normalize(v_normal);
    float NdotL = max(dot(normal, LIGHT_DIR), 0.0);
    float light = NdotL + 0.8; 

    vec3 h_grayscale = vec3(get_height(in_uv, in_layer)/2.);
    vec3 albedo = texture(colorMap, vec3(in_uv, float(in_layer))).rgb;

    f_color = vec4(albedo * light, 1.0);
    // f_color = vec4(h_grayscale * light, 1.0);
}
