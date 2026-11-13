#version 450

layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_world_normal;

layout(location = 0) out vec4 final_color;

layout(set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_proj;
    vec4 camera_pos;
    vec4 ambient;
    vec4 dir_direction;
    vec4 dir_color;
    ivec4 counts;
} scene;

layout(set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    mat4 normal_matrix;
    vec4 albedo;
    vec4 specular;
} object;

struct PointLight {
    vec4 position;   // xyz
    vec4 color;      // rgb * intensity
};

struct SpotLight {
    vec4 position;   // xyz
    vec4 direction;  // xyz — куда светит, w = cos(внутренний угол)
    vec4 color;      // rgb * intensity, w = cos(внешний угол)
};

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight spot_lights[];
};

// Модель Блинна-Фонга для одного источника.
// L — единичный вектор от точки к источнику, radiance — уже с учётом затухания.
vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance) {
    float ndl = max(dot(N, L), 0.0);
    vec3 H = normalize(L + V);
    float spec = ndl > 0.0 ? pow(max(dot(N, H), 0.0), object.specular.w) : 0.0;
    return radiance * (object.albedo.rgb * ndl + object.specular.rgb * spec);
}

void main() {
    vec3 N = normalize(v_world_normal);
    vec3 V = normalize(scene.camera_pos.xyz - v_world_pos);

    // Рассеянный свет
    vec3 color = scene.ambient.rgb * object.albedo.rgb;

    // Направленный источник
    vec3 Ld = -normalize(scene.dir_direction.xyz);
    color += blinnPhong(N, V, Ld, scene.dir_color.rgb);

    // Точечные источники: затухание по закону обратных квадратов
    for (int i = 0; i < scene.counts.x; ++i) {
        vec3 to_light = point_lights[i].position.xyz - v_world_pos;
        float dist2 = max(dot(to_light, to_light), 1e-4);
        vec3 L = to_light * inversesqrt(dist2);
        color += blinnPhong(N, V, L, point_lights[i].color.rgb / dist2);
    }

    // Прожекторные источники: обратные квадраты + мягкий край конуса
    for (int i = 0; i < scene.counts.y; ++i) {
        SpotLight s = spot_lights[i];
        vec3 to_light = s.position.xyz - v_world_pos;
        float dist2 = max(dot(to_light, to_light), 1e-4);
        vec3 L = to_light * inversesqrt(dist2);

        float cos_theta = dot(-L, normalize(s.direction.xyz));
        float cos_inner = s.direction.w;
        float cos_outer = s.color.w;
        float edge = clamp((cos_theta - cos_outer) / max(cos_inner - cos_outer, 1e-4), 0.0, 1.0);
        edge = smoothstep(0.0, 1.0, edge);

        color += blinnPhong(N, V, L, s.color.rgb / dist2 * edge);
    }

    final_color = vec4(color, 1.0);
}