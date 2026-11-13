#include "application.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <vector>
#include <imgui.h>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace application {

	namespace {

		struct Vertex {
			glm::vec3 position;
			glm::vec3 normal;
		};

		void generateCube(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices) {
			const glm::vec3 normals[6] = {
				{ 1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}
			};
			const float sx[4] = { -1,  1, 1, -1 };
			const float sy[4] = { -1, -1, 1,  1 };

			for (const glm::vec3& n : normals) {
				const glm::vec3 helper = (std::abs(n.y) > 0.5f) ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);
				const glm::vec3 u = glm::normalize(glm::cross(helper, n));
				const glm::vec3 w = glm::cross(n, u);

				const uint16_t base = uint16_t(vertices.size());
				for (int i = 0; i < 4; ++i) {
					vertices.push_back({ 0.5f * (n + sx[i] * u + sy[i] * w), n });
				}
				const uint16_t quad[6] = { 0, 1, 2, 0, 2, 3 };
				for (uint16_t q : quad) indices.push_back(uint16_t(base + q));
			}
		}

		std::vector<Vertex> cube_vertices;
		std::vector<uint16_t> cube_indices;

		VkBuffer vertex_buffer = VK_NULL_HANDLE;
		VmaAllocation vertex_allocation = VK_NULL_HANDLE;
		VkBuffer index_buffer = VK_NULL_HANDLE;
		VmaAllocation index_allocation = VK_NULL_HANDLE;

		constexpr uint32_t MAX_LIGHTS = 256; 

		struct SceneUniforms {
			glm::mat4  view_proj;
			glm::vec4  camera_pos;
			glm::vec4  ambient;        
			glm::vec4  dir_direction;  
			glm::vec4  dir_color;      
			glm::ivec4 counts;         
		};

		struct ModelUniforms {
			glm::mat4 model;
			glm::mat4 normal_matrix;
			glm::vec4 albedo;
			glm::vec4 specular;        
		};

		struct PointLightGPU {
			glm::vec4 position;
			glm::vec4 color;
		};

		struct SpotLightGPU {
			glm::vec4 position;
			glm::vec4 direction;       
			glm::vec4 color;           
		};

		struct PointLightCPU {
			glm::vec3 position{ 0.0f, 2.0f, 2.0f };
			glm::vec3 color{ 1.0f, 1.0f, 1.0f };
			float intensity = 10.0f;
		};

		struct SpotLightCPU {
			glm::vec3 position{ 0.0f, 3.0f, 0.0f };
			glm::vec3 direction{ 0.0f, -1.0f, 0.0f };
			glm::vec3 color{ 1.0f, 1.0f, 1.0f };
			float intensity = 20.0f;
			float inner_deg = 15.0f;
			float outer_deg = 25.0f;
		};

		struct LightingState {
			// камера 
			glm::vec3 camera_position{ 0.0f, 0.0f, 3.0f };
			float yaw_deg = 0.0f;     
			float pitch_deg = 0.0f;
			// рассеянный и направленный свет
			glm::vec3 ambient_color{ 1.0f, 1.0f, 1.0f };
			float ambient_intensity = 0.1f;
			glm::vec3 dir_direction{ -0.5f, -1.0f, -0.3f };
			glm::vec3 dir_color{ 1.0f, 1.0f, 1.0f };
			float dir_intensity = 1.0f;
			// материал
			glm::vec3 albedo{ 0.8f, 0.3f, 0.2f };
			glm::vec3 specular_color{ 1.0f, 1.0f, 1.0f };
			float shininess = 32.0f;
		
			std::vector<PointLightCPU> point_lights;
			std::vector<SpotLightCPU> spot_lights;
		};

		struct GpuBuffer {
			VkBuffer buffer = VK_NULL_HANDLE;
			VmaAllocation allocation = VK_NULL_HANDLE;
			void* mapped = nullptr;
		};

		struct SceneState {
			//1
			int projection = 0;            
			float fov_deg = 60.0f;
			float ortho_half_height = 1.5f;

			//2
			glm::vec3 position = glm::vec3(0.0f);
			glm::vec3 rotation_deg = glm::vec3(0.0f);
			glm::vec3 scale = glm::vec3(1.0f);

			//3
			bool animate = true;
			bool playing = true;
			float speed = 1.0f;
			float radius = 1.0f;
			float vertical_amplitude = 0.5f;
			int vertical_frequency = 2;
			float spin_speed = 1.0f;

		};

		SceneState scene;
		double animation_time = 0.0;
		double last_time = 0.0;
		bool show_demo = false;

		VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
		VkPipeline pipeline = VK_NULL_HANDLE;

		LightingState lighting;

		SceneUniforms scene_uniforms{};
		ModelUniforms model_uniforms{};
		std::vector<PointLightGPU> point_lights_gpu;
		std::vector<SpotLightGPU> spot_lights_gpu;

		GpuBuffer scene_ubo, model_ubo, point_ssbo, spot_ssbo;

		VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;
		VkDescriptorSetLayout model_set_layout = VK_NULL_HANDLE;
		VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
		VkDescriptorSet scene_set = VK_NULL_HANDLE;
		VkDescriptorSet model_set = VK_NULL_HANDLE;

		VkShaderModule loadShaderModule(const char* path) {
			std::ifstream file(path, std::ios::binary | std::ios::ate);
			if (!file) {
				std::cerr << "Failed to open shader file: " << path << '\n';
				return VK_NULL_HANDLE;
			}

			const size_t size = size_t(file.tellg());
			std::vector<uint32_t> code(size / sizeof(uint32_t));
			file.seekg(0);
			file.read(reinterpret_cast<char*>(code.data()), std::streamsize(size));

			const VkShaderModuleCreateInfo info = {
				.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
				.codeSize = size,
				.pCode = code.data(),
			};

			VkShaderModule module = VK_NULL_HANDLE;
			if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module) != VK_SUCCESS) {
				std::cerr << "Failed to create shader module: " << path << '\n';
				return VK_NULL_HANDLE;
			}
			return module;
		}

		bool createPipeline() {
			auto& context = graphics::internal::context;

			VkShaderModule vert = loadShaderModule("shaders/cube.vert.spv");
			VkShaderModule frag = loadShaderModule("shaders/cube.frag.spv");
			if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) {
				if (vert != VK_NULL_HANDLE) vkDestroyShaderModule(context.device, vert, nullptr);
				if (frag != VK_NULL_HANDLE) vkDestroyShaderModule(context.device, frag, nullptr);
				return false;
			}

			const VkDescriptorSetLayout set_layouts[] = { scene_set_layout, model_set_layout };
			const VkPipelineLayoutCreateInfo layout_info = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.setLayoutCount = 2,
				.pSetLayouts = set_layouts,
			};

			if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create pipeline layout\n";
				return false;
			}

			const VkPipelineShaderStageCreateInfo stages[] = {
				{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
					.stage = VK_SHADER_STAGE_VERTEX_BIT,
					.module = vert,
					.pName = "main",
				},
				{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
					.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
					.module = frag,
					.pName = "main",
				},
			};

			const VkVertexInputBindingDescription binding = {
				.binding = 0,
				.stride = sizeof(Vertex),
				.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
			};

			const VkVertexInputAttributeDescription attributes[] = {
				{.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position) },
				{.location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal) },
			};

			const VkPipelineVertexInputStateCreateInfo vertex_input = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
				.vertexBindingDescriptionCount = 1,
				.pVertexBindingDescriptions = &binding,
				.vertexAttributeDescriptionCount = 2,
				.pVertexAttributeDescriptions = attributes,
			};

			const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
				.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
			};

			const VkPipelineViewportStateCreateInfo viewport_state = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
				.viewportCount = 1,
				.scissorCount = 1,
			};

			const VkPipelineRasterizationStateCreateInfo rasterization = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
				.polygonMode = VK_POLYGON_MODE_FILL,
				.cullMode = VK_CULL_MODE_NONE,
				.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
				.lineWidth = 1.0f,
			};

			const VkPipelineMultisampleStateCreateInfo multisample = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
				.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			};

			const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
				.depthTestEnable = VK_TRUE,
				.depthWriteEnable = VK_TRUE,
				.depthCompareOp = VK_COMPARE_OP_LESS,
			};

			const VkPipelineColorBlendAttachmentState blend_attachment = {
				.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
								  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
			};

			const VkPipelineColorBlendStateCreateInfo color_blend = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
				.attachmentCount = 1,
				.pAttachments = &blend_attachment,
			};

			const VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

			const VkPipelineDynamicStateCreateInfo dynamic_state = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
				.dynamicStateCount = 2,
				.pDynamicStates = dynamic_states,
			};

			const VkGraphicsPipelineCreateInfo pipeline_info = {
				.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
				.stageCount = 2,
				.pStages = stages,
				.pVertexInputState = &vertex_input,
				.pInputAssemblyState = &input_assembly,
				.pViewportState = &viewport_state,
				.pRasterizationState = &rasterization,
				.pMultisampleState = &multisample,
				.pDepthStencilState = &depth_stencil,
				.pColorBlendState = &color_blend,
				.pDynamicState = &dynamic_state,
				.layout = pipeline_layout,
				.renderPass = context.render_pass,
				.subpass = 0,
			};

			const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline);

			vkDestroyShaderModule(context.device, vert, nullptr);
			vkDestroyShaderModule(context.device, frag, nullptr);

			if (result != VK_SUCCESS) {
				std::cerr << "Failed to create graphics pipeline\n";
				return false;
			}
			return true;
		}

		bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, const void* data,
			VkBuffer& buffer, VmaAllocation& allocation, void** mapped = nullptr) {
			auto& context = graphics::internal::context;

			const VkBufferCreateInfo buffer_info = {
				.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
				.size = size,
				.usage = usage,
				.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			};

			const VmaAllocationCreateInfo alloc_info = {
				.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
						 VMA_ALLOCATION_CREATE_MAPPED_BIT,
				.usage = VMA_MEMORY_USAGE_AUTO,
			};

			VmaAllocationInfo info;
			if (vmaCreateBuffer(context.allocator, &buffer_info, &alloc_info,
				&buffer, &allocation, &info) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan buffer\n";
				return false;
			}

			std::memcpy(info.pMappedData, data, size);
			vmaFlushAllocation(context.allocator, allocation, 0, VK_WHOLE_SIZE);
			if (mapped) *mapped = info.pMappedData;
			return true;
		}

		void uploadBuffer(const GpuBuffer& b, const void* data, VkDeviceSize size) {
			auto& context = graphics::internal::context;
			std::memcpy(b.mapped, data, size);
			vmaFlushAllocation(context.allocator, b.allocation, 0, size);
		}

		bool createDescriptors() {
			auto& context = graphics::internal::context;

			const VkShaderStageFlags vf = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
			const VkDescriptorSetLayoutBinding scene_bindings[] = {
				{.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = vf },
				{.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
				{.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
			};
			const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = 3,
				.pBindings = scene_bindings,
			};
			if (vkCreateDescriptorSetLayout(context.device, &scene_layout_info, nullptr,
				&scene_set_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create scene descriptor set layout\n";
				return false;
			}

			const VkDescriptorSetLayoutBinding model_binding = {
				.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = vf,
			};
			const VkDescriptorSetLayoutCreateInfo model_layout_info = {
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = 1,
				.pBindings = &model_binding,
			};
			if (vkCreateDescriptorSetLayout(context.device, &model_layout_info, nullptr,
				&model_set_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create model descriptor set layout\n";
				return false;
			}

			const VkDescriptorPoolSize pool_sizes[] = {
				{.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 2 },
				{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 },
			};
			const VkDescriptorPoolCreateInfo pool_info = {
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.maxSets = 2,
				.poolSizeCount = 2,
				.pPoolSizes = pool_sizes,
			};
			if (vkCreateDescriptorPool(context.device, &pool_info, nullptr,
				&descriptor_pool) != VK_SUCCESS) {
				std::cerr << "Failed to create descriptor pool\n";
				return false;
			}

			const VkDescriptorSetLayout layouts[] = { scene_set_layout, model_set_layout };
			const VkDescriptorSetAllocateInfo set_alloc_info = {
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 2,
				.pSetLayouts = layouts,
			};
			VkDescriptorSet sets[2];
			if (vkAllocateDescriptorSets(context.device, &set_alloc_info, sets) != VK_SUCCESS) {
				std::cerr << "Failed to allocate descriptor sets\n";
				return false;
			}
			scene_set = sets[0];
			model_set = sets[1];

			const VkDescriptorBufferInfo scene_info = { scene_ubo.buffer, 0, sizeof(SceneUniforms) };
			const VkDescriptorBufferInfo point_info = { point_ssbo.buffer, 0, sizeof(PointLightGPU) * MAX_LIGHTS };
			const VkDescriptorBufferInfo spot_info = { spot_ssbo.buffer,  0, sizeof(SpotLightGPU) * MAX_LIGHTS };
			const VkDescriptorBufferInfo model_info = { model_ubo.buffer,  0, sizeof(ModelUniforms) };

			const VkWriteDescriptorSet writes[] = {
				{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = scene_set, .dstBinding = 0,
				  .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &scene_info },
				{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = scene_set, .dstBinding = 1,
				  .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &point_info },
				{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = scene_set, .dstBinding = 2,
				  .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &spot_info },
				{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = model_set, .dstBinding = 0,
				  .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &model_info },
			};
			vkUpdateDescriptorSets(context.device, 4, writes, 0, nullptr);
			return true;
		}

		void drawLab2UI(LightingState& s) {
			ImGui::Begin("Lab 2: Camera & Lighting");

			if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
				ImGui::DragFloat3("Position##cam", &s.camera_position.x, 0.05f);
				ImGui::SliderFloat("Yaw", &s.yaw_deg, -180.0f, 180.0f);
				ImGui::SliderFloat("Pitch", &s.pitch_deg, -89.0f, 89.0f);
				ImGui::TextDisabled("FOV: в окне Cube controls -> Projection");
			}

			if (ImGui::CollapsingHeader("Ambient & directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
				ImGui::ColorEdit3("Ambient color", &s.ambient_color.x);
				ImGui::SliderFloat("Ambient intensity", &s.ambient_intensity, 0.0f, 1.0f);
				ImGui::DragFloat3("Direction", &s.dir_direction.x, 0.01f);
				ImGui::ColorEdit3("Color##dir", &s.dir_color.x);
				ImGui::SliderFloat("Intensity##dir", &s.dir_intensity, 0.0f, 5.0f);
			}

			if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
				ImGui::ColorEdit3("Albedo", &s.albedo.x);
				ImGui::ColorEdit3("Specular color", &s.specular_color.x);
				ImGui::SliderFloat("Shininess", &s.shininess, 1.0f, 256.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
			}

			if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
				ImGui::BeginDisabled(s.point_lights.size() >= MAX_LIGHTS);
				if (ImGui::Button("Add point light")) s.point_lights.emplace_back();
				ImGui::EndDisabled();

				int remove_idx = -1;
				for (int i = 0; i < int(s.point_lights.size()); ++i) {
					PointLightCPU& l = s.point_lights[size_t(i)];
					ImGui::PushID(i);
					if (ImGui::TreeNode("node", "Point light %d", i)) {
						ImGui::DragFloat3("Position", &l.position.x, 0.05f);
						ImGui::ColorEdit3("Color", &l.color.x);
						ImGui::DragFloat("Intensity", &l.intensity, 0.1f, 0.0f, 1000.0f);
						if (ImGui::Button("Remove")) remove_idx = i;
						ImGui::TreePop();
					}
					ImGui::PopID();
				}
				if (remove_idx >= 0) s.point_lights.erase(s.point_lights.begin() + remove_idx);
			}

			if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
				ImGui::BeginDisabled(s.spot_lights.size() >= MAX_LIGHTS);
				if (ImGui::Button("Add spot light")) s.spot_lights.emplace_back();
				ImGui::EndDisabled();

				int remove_idx = -1;
				for (int i = 0; i < int(s.spot_lights.size()); ++i) {
					SpotLightCPU& l = s.spot_lights[size_t(i)];
					ImGui::PushID(1000 + i);
					if (ImGui::TreeNode("node", "Spot light %d", i)) {
						ImGui::DragFloat3("Position", &l.position.x, 0.05f);
						ImGui::DragFloat3("Direction", &l.direction.x, 0.01f);
						ImGui::ColorEdit3("Color", &l.color.x);
						ImGui::DragFloat("Intensity", &l.intensity, 0.1f, 0.0f, 1000.0f);
						ImGui::SliderFloat("Inner angle", &l.inner_deg, 0.0f, 89.0f);
						ImGui::SliderFloat("Outer angle", &l.outer_deg, 0.0f, 90.0f);
						if (l.inner_deg > l.outer_deg) l.inner_deg = l.outer_deg;
						if (ImGui::Button("Remove")) remove_idx = i;
						ImGui::TreePop();
					}
					ImGui::PopID();
				}
				if (remove_idx >= 0) s.spot_lights.erase(s.spot_lights.begin() + remove_idx);
			}

			ImGui::End();
		}

	} // namespace

	bool initialize() {
		generateCube(cube_vertices, cube_indices);

		if (!createBuffer(cube_vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			cube_vertices.data(), vertex_buffer, vertex_allocation)) {
			return false;
		}

		if (!createBuffer(cube_indices.size() * sizeof(uint16_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			cube_indices.data(), index_buffer, index_allocation)) {
			return false;
		}

		if (!createBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			&scene_uniforms, scene_ubo.buffer, scene_ubo.allocation, &scene_ubo.mapped)) {
			return false;
		}
		if (!createBuffer(sizeof(ModelUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			&model_uniforms, model_ubo.buffer, model_ubo.allocation, &model_ubo.mapped)) {
			return false;
		}

		
		const std::vector<PointLightGPU> zero_points(MAX_LIGHTS);
		const std::vector<SpotLightGPU> zero_spots(MAX_LIGHTS);
		if (!createBuffer(sizeof(PointLightGPU) * MAX_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			zero_points.data(), point_ssbo.buffer, point_ssbo.allocation, &point_ssbo.mapped)) {
			return false;
		}
		if (!createBuffer(sizeof(SpotLightGPU) * MAX_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			zero_spots.data(), spot_ssbo.buffer, spot_ssbo.allocation, &spot_ssbo.mapped)) {
			return false;
		}

		if (!createDescriptors()) {
			return false;
		}

		if (!createPipeline()) {
			return false;
		}
		return true;
	}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);

	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, scene_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, model_set_layout, nullptr);
	vmaDestroyBuffer(context.allocator, scene_ubo.buffer, scene_ubo.allocation);
	vmaDestroyBuffer(context.allocator, model_ubo.buffer, model_ubo.allocation);
	vmaDestroyBuffer(context.allocator, point_ssbo.buffer, point_ssbo.allocation);
	vmaDestroyBuffer(context.allocator, spot_ssbo.buffer, spot_ssbo.allocation);

	vmaDestroyBuffer(context.allocator, index_buffer, index_allocation);
	vmaDestroyBuffer(context.allocator, vertex_buffer, vertex_allocation);
}

void update(double time) {
	const float dt = float(time - last_time);
	last_time = time;

	ImGui::Begin("Cube controls");

	if (ImGui::CollapsingHeader("Projection", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::RadioButton("Perspective", &scene.projection, 0);
		ImGui::SameLine();
		ImGui::RadioButton("Orthographic", &scene.projection, 1);
		if (scene.projection == 0) {
			ImGui::SliderFloat("FOV (deg)", &scene.fov_deg, 20.0f, 120.0f);
		}
		else {
			ImGui::SliderFloat("Ortho half-height", &scene.ortho_half_height, 0.5f, 5.0f);
		}
	}

	if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Position", &scene.position.x, 0.01f);
		ImGui::SliderFloat3("Rotation (deg)", &scene.rotation_deg.x, -180.0f, 180.0f);
		ImGui::DragFloat3("Scale", &scene.scale.x, 0.01f, 0.05f, 5.0f);
	}

	if (ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Checkbox("Enable trajectory", &scene.animate);
		if (ImGui::Button(scene.playing ? "Pause" : "Play")) {
			scene.playing = !scene.playing;
		}
		ImGui::SameLine();
		if (ImGui::Button("Reset time")) {
			animation_time = 0.0;
		}
		ImGui::SliderFloat("Speed", &scene.speed, 0.0f, 5.0f);
		ImGui::SliderFloat("Radius", &scene.radius, 0.0f, 3.0f);
		ImGui::SliderFloat("Vertical amplitude", &scene.vertical_amplitude, 0.0f, 2.0f);
		ImGui::SliderInt("Vertical frequency", &scene.vertical_frequency, 1, 6);
		ImGui::SliderFloat("Spin speed", &scene.spin_speed, 0.0f, 5.0f);
	}

	

	if (ImGui::Button("Reset all")) {
		scene = SceneState();
		animation_time = 0.0;
	}
	ImGui::Checkbox("Show ImGui demo", &show_demo);
	ImGui::End();
	drawLab2UI(lighting);

	if (show_demo) {
		ImGui::ShowDemoWindow(&show_demo);
	}

	if (scene.animate && scene.playing) {
		animation_time += double(dt) * double(scene.speed);
	}
	const float t = float(animation_time);

	glm::vec3 offset(0.0f);
	float spin = 0.0f;
	if (scene.animate) {
		offset = glm::vec3(scene.radius * std::cos(t),
			scene.vertical_amplitude * std::sin(float(scene.vertical_frequency) * t),
			scene.radius * std::sin(t));
		spin = scene.spin_speed * t;
	}

	glm::mat4 model = glm::translate(glm::mat4(1.0f), scene.position + offset);
	model = glm::rotate(model, glm::radians(scene.rotation_deg.z), glm::vec3(0.0f, 0.0f, 1.0f));
	model = glm::rotate(model, glm::radians(scene.rotation_deg.y), glm::vec3(0.0f, 1.0f, 0.0f));
	model = glm::rotate(model, glm::radians(scene.rotation_deg.x), glm::vec3(1.0f, 0.0f, 0.0f));
	model = glm::rotate(model, spin, glm::vec3(0.5f, 1.0f, 0.0f));
	model = glm::scale(model, scene.scale);

	const float yaw = glm::radians(lighting.yaw_deg);
	const float pitch = glm::radians(lighting.pitch_deg);
	const glm::vec3 forward(std::cos(pitch) * std::sin(yaw),
		std::sin(pitch),
		-std::cos(pitch) * std::cos(yaw));
	const glm::mat4 view = glm::lookAt(lighting.camera_position,
		lighting.camera_position + forward,
		glm::vec3(0.0f, 1.0f, 0.0f));

	const auto extent = graphics::internal::context.swapchain_extent;
	const float aspect = float(extent.width) / float(std::max(extent.height, 1u));

	glm::mat4 proj;
	if (scene.projection == 0) {
		proj = glm::perspective(glm::radians(scene.fov_deg), aspect, 0.1f, 100.0f);
	}
	else {
		const float h = scene.ortho_half_height;
		proj = glm::ortho(-h * aspect, h * aspect, -h, h, 0.1f, 100.0f);
	}
	proj[1][1] *= -1.0f;

	// --- сцена ---
	glm::vec3 dir = lighting.dir_direction;
	if (glm::dot(dir, dir) < 1e-8f) dir = glm::vec3(0.0f, -1.0f, 0.0f);

	const uint32_t point_count = uint32_t(std::min<size_t>(lighting.point_lights.size(), MAX_LIGHTS));
	const uint32_t spot_count = uint32_t(std::min<size_t>(lighting.spot_lights.size(), MAX_LIGHTS));

	scene_uniforms.view_proj = proj * view;
	scene_uniforms.camera_pos = glm::vec4(lighting.camera_position, 1.0f);
	scene_uniforms.ambient = glm::vec4(lighting.ambient_color * lighting.ambient_intensity, 1.0f);
	scene_uniforms.dir_direction = glm::vec4(glm::normalize(dir), 0.0f);
	scene_uniforms.dir_color = glm::vec4(lighting.dir_color * lighting.dir_intensity, 1.0f);
	scene_uniforms.counts = glm::ivec4(int(point_count), int(spot_count), 0, 0);

	// --- модель ---
	model_uniforms.model = model;
	model_uniforms.normal_matrix = glm::transpose(glm::inverse(model));
	model_uniforms.albedo = glm::vec4(lighting.albedo, 1.0f);
	model_uniforms.specular = glm::vec4(lighting.specular_color, lighting.shininess);

	// --- источники ---
	point_lights_gpu.clear();
	for (uint32_t i = 0; i < point_count; ++i) {
		const PointLightCPU& l = lighting.point_lights[i];
		point_lights_gpu.push_back({ glm::vec4(l.position, 1.0f),
									 glm::vec4(l.color * l.intensity, 0.0f) });
	}

	spot_lights_gpu.clear();
	for (uint32_t i = 0; i < spot_count; ++i) {
		const SpotLightCPU& l = lighting.spot_lights[i];
		glm::vec3 d = l.direction;
		if (glm::dot(d, d) < 1e-8f) d = glm::vec3(0.0f, -1.0f, 0.0f);
		spot_lights_gpu.push_back({ glm::vec4(l.position, 1.0f),
									glm::vec4(glm::normalize(d), std::cos(glm::radians(l.inner_deg))),
									glm::vec4(l.color * l.intensity, std::cos(glm::radians(l.outer_deg))) });
	}
}
void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;

	uploadBuffer(scene_ubo, &scene_uniforms, sizeof(SceneUniforms));
	uploadBuffer(model_ubo, &model_uniforms, sizeof(ModelUniforms));
	if (!point_lights_gpu.empty())
		uploadBuffer(point_ssbo, point_lights_gpu.data(), point_lights_gpu.size() * sizeof(PointLightGPU));
	if (!spot_lights_gpu.empty())
		uploadBuffer(spot_ssbo, spot_lights_gpu.data(), spot_lights_gpu.size() * sizeof(SpotLightGPU));

	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(fd.command_buffer, &begin);

	VkClearValue clear_values[2] = {};
	clear_values[0].color = { { 0.1f, 0.1f, 0.1f, 1.0f } };
	clear_values[1].depthStencil = { 1.0f, 0 };         

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = {.extent = context.swapchain_extent },
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

	const VkViewport viewport = {
		0.0f, 0.0f,
		float(context.swapchain_extent.width), float(context.swapchain_extent.height),
		0.0f, 1.0f,
	};
	const VkRect2D scissor = { {0, 0}, context.swapchain_extent };
	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer, &offset);
	vkCmdBindIndexBuffer(fd.command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT16);
	const VkDescriptorSet sets[] = { scene_set, model_set };
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		pipeline_layout, 0, 2, sets, 0, nullptr);

	vkCmdDrawIndexed(fd.command_buffer, uint32_t(cube_indices.size()), 1, 0, 0, 0);

	vkCmdEndRenderPass(fd.command_buffer);
	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application