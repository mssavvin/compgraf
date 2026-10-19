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
			glm::vec3 color;
		};

		constexpr float half = 0.5f;

		Vertex makeVertex(float x, float y, float z) {
			return { {x, y, z}, {x + half, y + half, z + half} };
		}

		const std::array<Vertex, 8> cube_vertices = {
			makeVertex(-half, -half, -half),
			makeVertex(half, -half, -half),
			makeVertex(half,  half, -half),
			makeVertex(-half,  half, -half), 
			makeVertex(-half, -half,  half), 
			makeVertex(half, -half,  half), 
			makeVertex(half,  half,  half), 
			makeVertex(-half,  half,  half),
		};

		const std::array<uint16_t, 36> cube_indices = {
			4, 5, 6,  4, 6, 7, 
			1, 0, 3,  1, 3, 2, 
			5, 1, 2,  5, 2, 6, 
			0, 4, 7,  0, 7, 3, 
			7, 6, 2,  7, 2, 3, 
			0, 1, 5,  0, 5, 4, 
		};

		VkBuffer vertex_buffer = VK_NULL_HANDLE;
		VmaAllocation vertex_allocation = VK_NULL_HANDLE;
		VkBuffer index_buffer = VK_NULL_HANDLE;
		VmaAllocation index_allocation = VK_NULL_HANDLE;

		struct PushConstants {
			glm::mat4 mvp;
			glm::vec4 tint;
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

			//4
			glm::vec4 tint = glm::vec4(1.0f);
		};

		SceneState scene;
		double animation_time = 0.0;
		double last_time = 0.0;
		bool show_demo = false;

		VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
		VkPipeline pipeline = VK_NULL_HANDLE;
		PushConstants push = { glm::mat4(1.0f), glm::vec4(1.0f) };

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

			const VkPushConstantRange push_range = {
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
				.offset = 0,
				.size = sizeof(PushConstants),
			};

			const VkPipelineLayoutCreateInfo layout_info = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.pushConstantRangeCount = 1,
				.pPushConstantRanges = &push_range,
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
				{.location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, color) },
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
			VkBuffer& buffer, VmaAllocation& allocation) {
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
			return true;
		}

	} // namespace

	bool initialize() {
		if (!createBuffer(sizeof(cube_vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			cube_vertices.data(), vertex_buffer, vertex_allocation)) {
			return false;
		}

		if (!createBuffer(sizeof(cube_indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			cube_indices.data(), index_buffer, index_allocation)) {
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

	if (ImGui::CollapsingHeader("Color", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Tint", &scene.tint.x);
	}

	if (ImGui::Button("Reset all")) {
		scene = SceneState();
		animation_time = 0.0;
	}
	ImGui::Checkbox("Show ImGui demo", &show_demo);
	ImGui::End();

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

	const glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

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

	push.mvp = proj * view * model;
	push.tint = scene.tint;
}
void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;

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
	vkCmdPushConstants(fd.command_buffer, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
		0, sizeof(PushConstants), &push);

	vkCmdDrawIndexed(fd.command_buffer, uint32_t(cube_indices.size()), 1, 0, 0, 0);

	vkCmdEndRenderPass(fd.command_buffer);
	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application