/*-------------------------------------------------------------------------
 * Vulkan Conformance Tests
 * ------------------------
 *
 * Copyright (c) 2017-2019 The Khronos Group Inc.
 * Copyright (c) 2018-2019 NVIDIA Corporation
 * Copyright (c) 2026 Huawei Technologies Co. Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *//*!
 * \file
 * \brief Memory Model Test Case
 *//*--------------------------------------------------------------------*/

#include "vktMemoryModelTestCase.hpp"

#include "deDefs.h"
#include "deMath.h"
#include "tcuDefs.hpp"
#include "tcuTestLog.hpp"
#include "vkBarrierUtil.hpp"
#include "vkBufferWithMemory.hpp"
#include "vkBuilderUtil.hpp"
#include "vkCmdUtil.hpp"
#include "vkImageWithMemory.hpp"
#include "vkMemUtil.hpp"
#include "vkObjUtil.hpp"
#include "vkPrograms.hpp"
#include "vkQueryUtil.hpp"
#include "vkRefUtil.hpp"
#include "vkTypeUtil.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <string_view>

namespace vkt
{
namespace MemoryModel
{
using namespace vk;

namespace
{

constexpr uint32_t GL_SEM1 = SEM_BUFFER | SEM_IMAGE | SEM_SHARED;
constexpr uint32_t GL_SEM2 = SEM_ACQUIRE | SEM_RELEASE | SEM_ACQUIRE_RELEASE | SEM_AVAILABLE | SEM_VISIBLE;

constexpr uint32_t DEVICE_ADDRESS_SIZE = static_cast<uint32_t>(sizeof(VkDeviceAddress));

template <std::size_t N>
std::string formatSemantics(uint32_t value, const std::pair<uint32_t, const char *> (&mappings)[N])
{
    std::string result;
    for (const auto &mapping : mappings)
    {
        if (value & mapping.first)
        {
            if (!result.empty())
                result += " | ";
            result += mapping.second;
        }
    }
    return result.empty() ? "0" : result;
}

std::string gl_semantics1(uint32_t value)
{
    static constexpr std::pair<uint32_t, const char *> kMappings[] = {
        {SEM_BUFFER, "gl_StorageSemanticsBuffer"},
        {SEM_IMAGE, "gl_StorageSemanticsImage"},
        {SEM_SHARED, "gl_StorageSemanticsShared"},
    };
    return formatSemantics(value & GL_SEM1, kMappings);
}

std::string gl_semantics2(uint32_t value)
{
    static constexpr std::pair<uint32_t, const char *> kMappings[] = {
        {SEM_ACQUIRE, "gl_SemanticsAcquire"},
        {SEM_RELEASE, "gl_SemanticsRelease"},
        {SEM_ACQUIRE_RELEASE, "gl_SemanticsAcquireRelease"},
        {SEM_AVAILABLE, "gl_SemanticsMakeAvailable"},
        {SEM_VISIBLE, "gl_SemanticsMakeVisible"},
    };
    return formatSemantics(value & GL_SEM2, kMappings);
}

const char *gl_qualifier(const MemQualifier qual)
{
    switch (qual)
    {
    case MEM_NONE:
        return "";
    case MEM_NONPRIVATE:
        return "nonprivate";
    case MEM_SUBGROUP:
        return "subgroupcoherent";
    case MEM_WORKGROUP:
        return "workgroupcoherent";
    case MEM_QUEUEFAMILY:
        return "queuefamilycoherent";
    case MEM_DEVICE:
        return "devicecoherent";
    case MEM_COHERENT:
        return "coherent";
    default:
        DE_ASSERT(0);
        return "";
    }
}

std::string declareCoordinate(const BufferDef &buffer, const std::string &var, const std::string &pos)
{
    switch (buffer.sc)
    {
    case SC_BUFFER:
    case SC_PHYSBUFFER:
        return "   uint " + var + " = getBufferCoord(" + pos + ");\n";
    case SC_IMAGE:
        return "   ivec2 " + var + " = " + pos + ";\n";
    case SC_WORKGROUP:
        return "   uint " + var + " = getSharedCoord(" + pos + ");\n";
    default:
        DE_ASSERT(0);
        return {};
    }
}

MemoryRequirement getBufferMemoryRequirement(const BufferDef &buffer, bool isDeviceAddressSupported)
{
    const bool deviceAddress = isDeviceAddressSupported && buffer.sc == SC_PHYSBUFFER;
    return (deviceAddress ? MemoryRequirement::DeviceAddress : MemoryRequirement::Any) |
           (buffer.local ? MemoryRequirement::Local : MemoryRequirement::NonLocal);
}

VkImageCreateInfo makeImageCreateInfo(const VkFormat format, const uint32_t width, const uint32_t height,
                                      const VkImageUsageFlags usage)
{
    return {
        VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, // VkStructureType sType;
        nullptr,                             // const void* pNext;
        (VkImageCreateFlags)0u,              // VkImageCreateFlags flags;
        VK_IMAGE_TYPE_2D,                    // VkImageType imageType;
        format,                              // VkFormat format;
        makeExtent3D(width, height, 1u),     // VkExtent3D extent;
        1u,                                  // uint32_t mipLevels;
        1u,                                  // uint32_t arrayLayers;
        VK_SAMPLE_COUNT_1_BIT,               // VkSampleCountFlagBits samples;
        VK_IMAGE_TILING_OPTIMAL,             // VkImageTiling tiling;
        usage,                               // VkImageUsageFlags usage;
        VK_SHARING_MODE_EXCLUSIVE,           // VkSharingMode sharingMode;
        0u,                                  // uint32_t queueFamilyIndexCount;
        nullptr,                             // const uint32_t* pQueueFamilyIndices;
        VK_IMAGE_LAYOUT_UNDEFINED            // VkImageLayout initialLayout;
    };
}

// The fragment stage is optional: it is left out when fragmentModule is VK_NULL_HANDLE
Move<VkPipeline> makeGraphicsPipeline(const DeviceInterface &vk, const VkDevice device,
                                      const VkPipelineLayout pipelineLayout, const VkShaderModule vertexModule,
                                      const VkShaderModule fragmentModule, const VkRenderPass renderPass,
                                      const VkViewport *viewport, const VkRect2D *scissor,
                                      const VkPrimitiveTopology topology, const VkBool32 rasterizerDiscardEnable,
                                      const VkSpecializationInfo *specInfo)
{
    const VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, // VkStructureType sType;
        nullptr,                                                   // const void* pNext;
        (VkPipelineVertexInputStateCreateFlags)0,                  // VkPipelineVertexInputStateCreateFlags flags;
        0u,                                                        // uint32_t vertexBindingDescriptionCount;
        nullptr, // const VkVertexInputBindingDescription* pVertexBindingDescriptions;
        0u,      // uint32_t vertexAttributeDescriptionCount;
        nullptr  // const VkVertexInputAttributeDescription* pVertexAttributeDescriptions;
    };

    const VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, // VkStructureType sType;
        nullptr,                                                     // const void* pNext;
        (VkPipelineInputAssemblyStateCreateFlags)0,                  // VkPipelineInputAssemblyStateCreateFlags flags;
        topology,                                                    // VkPrimitiveTopology topology;
        VK_FALSE                                                     // VkBool32 primitiveRestartEnable;
    };

    const VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, // VkStructureType sType;
        nullptr,                                                    // const void* pNext;
        (VkPipelineRasterizationStateCreateFlags)0,                 // VkPipelineRasterizationStateCreateFlags flags;
        VK_FALSE,                                                   // VkBool32 depthClampEnable;
        rasterizerDiscardEnable,                                    // VkBool32 rasterizerDiscardEnable;
        VK_POLYGON_MODE_FILL,                                       // VkPolygonMode polygonMode;
        VK_CULL_MODE_NONE,                                          // VkCullModeFlags cullMode;
        VK_FRONT_FACE_CLOCKWISE,                                    // VkFrontFace frontFace;
        VK_FALSE,                                                   // VkBool32 depthBiasEnable;
        0.0f,                                                       // float depthBiasConstantFactor;
        0.0f,                                                       // float depthBiasClamp;
        0.0f,                                                       // float depthBiasSlopeFactor;
        1.0f                                                        // float lineWidth;
    };

    const VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, // VkStructureType sType
        nullptr,                                                  // const void* pNext
        0u,                                                       // VkPipelineMultisampleStateCreateFlags flags
        VK_SAMPLE_COUNT_1_BIT,                                    // VkSampleCountFlagBits rasterizationSamples
        VK_FALSE,                                                 // VkBool32 sampleShadingEnable
        1.0f,                                                     // float minSampleShading
        nullptr,                                                  // const VkSampleMask* pSampleMask
        VK_FALSE,                                                 // VkBool32 alphaToCoverageEnable
        VK_FALSE                                                  // VkBool32 alphaToOneEnable
    };

    const VkPipelineViewportStateCreateInfo viewportStateCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, // VkStructureType sType
        nullptr,                                               // const void* pNext
        (VkPipelineViewportStateCreateFlags)0,                 // VkPipelineViewportStateCreateFlags flags
        1u,                                                    // uint32_t viewportCount
        viewport,                                              // const VkViewport* pViewports
        1u,                                                    // uint32_t scissorCount
        scissor                                                // const VkRect2D* pScissors
    };

    std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
    shaderStages.push_back({
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, (VkPipelineShaderStageCreateFlags)0,
        VK_SHADER_STAGE_VERTEX_BIT, // stage
        vertexModule,               // shader
        "main",
        specInfo, // pSpecializationInfo
    });
    if (fragmentModule != VK_NULL_HANDLE)
    {
        shaderStages.push_back({
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, (VkPipelineShaderStageCreateFlags)0,
            VK_SHADER_STAGE_FRAGMENT_BIT, // stage
            fragmentModule,               // shader
            "main",
            specInfo, // pSpecializationInfo
        });
    }

    const VkGraphicsPipelineCreateInfo graphicsPipelineCreateInfo = {
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, // VkStructureType sType;
        nullptr,                                         // const void* pNext;
        (VkPipelineCreateFlags)0,                        // VkPipelineCreateFlags flags;
        static_cast<uint32_t>(shaderStages.size()),      // uint32_t stageCount;
        shaderStages.data(),                             // const VkPipelineShaderStageCreateInfo* pStages;
        &vertexInputStateCreateInfo,   // const VkPipelineVertexInputStateCreateInfo* pVertexInputState;
        &inputAssemblyStateCreateInfo, // const VkPipelineInputAssemblyStateCreateInfo* pInputAssemblyState;
        nullptr,                       // const VkPipelineTessellationStateCreateInfo* pTessellationState;
        &viewportStateCreateInfo,      // const VkPipelineViewportStateCreateInfo* pViewportState;
        &rasterizationStateCreateInfo, // const VkPipelineRasterizationStateCreateInfo* pRasterizationState;
        &multisampleStateCreateInfo,   // const VkPipelineMultisampleStateCreateInfo* pMultisampleState;
        nullptr,                       // const VkPipelineDepthStencilStateCreateInfo* pDepthStencilState;
        nullptr,                       // const VkPipelineColorBlendStateCreateInfo* pColorBlendState;
        nullptr,                       // const VkPipelineDynamicStateCreateInfo* pDynamicState;
        pipelineLayout,                // VkPipelineLayout layout;
        renderPass,                    // VkRenderPass renderPass;
        0u,                            // uint32_t subpass;
        VK_NULL_HANDLE,                // VkPipeline basePipelineHandle;
        0                              // int basePipelineIndex;
    };

    return createGraphicsPipeline(vk, device, VK_NULL_HANDLE, &graphicsPipelineCreateInfo);
}

} // namespace

void MemoryModelTestCase::initPrograms(SourceCollections &programCollection) const
{
    std::ostringstream css;

    css << "#version 450 core\n";
    if (!m_params.core11)
    {
        css << "#pragma use_vulkan_memory_model\n";
    }
    css << "#extension GL_EXT_shader_atomic_float : enable\n"
           "#extension GL_EXT_split_barrier : enable\n"
           "#extension GL_KHR_shader_subgroup_basic : enable\n"
           "#extension GL_KHR_shader_subgroup_shuffle : enable\n"
           "#extension GL_KHR_shader_subgroup_ballot : enable\n"
           "#extension GL_KHR_memory_scope_semantics : enable\n"
           "#extension GL_ARB_gpu_shader_int64 : enable\n"
           "#extension GL_EXT_buffer_reference : enable\n";

    // Can be overridden by a child class, empty by default
    css << declareSpirvIntrinsics();

    css << "// DIM/NUM_WORKGROUP_EACH_DIM overriden by spec constants\n"
           "layout(constant_id = 0) const int DIM = 1;\n"
           "layout(constant_id = 1) const int NUM_WORKGROUP_EACH_DIM = 1;\n";

    // Can be overridden by a child class, empty by default
    css << declareSharedVariables();

    if (m_params.stage == STAGE_COMPUTE)
    {
        css << "layout(local_size_x_id = 0, local_size_y_id = 0, local_size_z = 1) in;\n";
    }

    for (size_t i = 0; i < m_buffers.size(); i++)
    {
        const auto &buffer = m_buffers[i];
        switch (buffer.sc)
        {
        case SC_PHYSBUFFER:
        {
            css << "layout(buffer_reference) buffer T" << buffer.name << "Ref { " << gl_type(buffer.type)
                << " x[]; };\n";
            break;
        }
        case SC_BUFFER:
        {
            css << "layout(set=0, binding=" << i << ") " << gl_qualifier(buffer.qual) << " buffer T" << buffer.name
                << " { " << gl_type(buffer.type) << " x[]; } " << buffer.name << ";\n";
            break;
        }
        case SC_IMAGE:
        {
            const char *type1 = isIntType(buffer.type) ? "r32ui" : "r32f";
            const char *type2 = isIntType(buffer.type) ? "uimage2D" : "image2D";
            css << "layout(set=0, binding=" << i << ", " << type1 << ") uniform " << gl_qualifier(buffer.qual) << " "
                << type2 << " " << buffer.name << ";\n";
            break;
        }
        case SC_WORKGROUP:
        {
            css << "struct S" << buffer.name << " { " << gl_type(buffer.type) << " x[DIM*DIM]; };\n"
                << "shared S" << buffer.name << " " << buffer.name << ";\n";
            break;
        }
        default:
            DE_ASSERT(0);
        }
    }

    if (hasStorageClass(SC_PHYSBUFFER))
    {
        css << "layout (push_constant, std430) uniform PC {\n";
        for (size_t i = 0; i < m_buffers.size(); i++)
        {
            if (m_buffers[i].sc == SC_PHYSBUFFER)
            {
                const char *suffix = m_buffers[i].qual == MEM_NONE ? "" : "ref";
                css << "layout(offset = " << (i * DEVICE_ADDRESS_SIZE) << ") T" << m_buffers[i].name << "Ref "
                    << m_buffers[i].name << suffix << ";\n";
            }
        }
        css << "};\n";
    }

    // Declare functions to compute invocation ID and coordinates
    // Incremented invocationIds by one to avoid collisions with initial zero
    css << "uint getInvocationId(ivec2 pos) { return uint(pos.y * DIM * NUM_WORKGROUP_EACH_DIM + pos.x) + 1u; }\n"
        << "uint getSharedCoord(ivec2 pos) { return uint((pos.y % DIM) * DIM + pos.x % DIM); }\n"
        << "uint getBufferCoord(ivec2 pos) { return uint((pos.y / DIM * NUM_WORKGROUP_EACH_DIM + pos.x / DIM) * "
           "DIM * DIM) + getSharedCoord(pos); }\n";

    css << "void main()\n"
           "{\n";

    if (m_params.stage == STAGE_FRAGMENT)
    {
        // Kill helper invocations so they don't load outside the bounds of the SSBO.
        // Helper pixels are also initially "active" and if a thread gets one as its
        // partner in SCOPE_SUBGROUP mode, it can't run the test.
        css << "   if (gl_HelperInvocation) { return; }\n";
    }

    for (const auto &buffer : m_buffers)
    {
        if (buffer.sc == SC_PHYSBUFFER && buffer.qual != MEM_NONE)
        {
            css << "   " << gl_qualifier(buffer.qual) << " T" << buffer.name << "Ref " << buffer.name << " = "
                << buffer.name << "ref;\n";
        }
    }

    css << initLayoutCoordinates();

    for (const auto &buffer : m_buffers)
    {
        if (buffer.sc == SC_WORKGROUP)
        {
            css << "   " << buffer.name << ".x[" << buffer.name << "Coord] = 0;\n";
        }
    }

    if (hasStorageClass(SC_WORKGROUP))
    {
        switch (m_params.mappingScope)
        {
        case SCOPE_SUBGROUP:
            css << "   subgroupBarrier();\n";
            break;
        case SCOPE_WORKGROUP:
            css << "   barrier();\n";
            break;
        default:
            DE_ASSERT(0);
        }
    }

    // Implemented by a child class
    css << initProgram();

    css << "}\n";

    if (m_params.stage == STAGE_FRAGMENT)
    {
        // Draw a fullscreen triangle strip based on gl_VertexIndex
        std::ostringstream vss;
        vss << "#version 450 core\n"
               "vec2 coords[4] = {ivec2(-1,-1), ivec2(-1, 1), ivec2(1, -1), ivec2(1, 1)};\n"
               "void main() { gl_Position = vec4(coords[gl_VertexIndex], 0, 1); }\n";
        programCollection.glslSources.add("vert") << glu::VertexSource(vss.str());
    }

    const vk::ShaderBuildOptions buildOptions(programCollection.usedVulkanVersion, vk::SPIRV_VERSION_1_3, 0u);

    switch (m_params.stage)
    {
    case STAGE_COMPUTE:
        programCollection.glslSources.add("test") << glu::ComputeSource(css.str()) << buildOptions;
        break;
    case STAGE_VERTEX:
        programCollection.glslSources.add("test") << glu::VertexSource(css.str()) << buildOptions;
        break;
    case STAGE_FRAGMENT:
        programCollection.glslSources.add("test") << glu::FragmentSource(css.str()) << buildOptions;
        break;
    default:
        DE_ASSERT(0);
    }
}

std::string MemoryModelTestCase::initLayoutCoordinates() const
{
    DE_ASSERT(m_params.numPartners >= 1);

    std::ostringstream css;
    css << "   uint gridDim = DIM * NUM_WORKGROUP_EACH_DIM;\n";

    // Compute thread's 2D position (global ID).
    switch (m_params.stage)
    {
    case STAGE_COMPUTE:
        css << "   ivec2 pos = ivec2(gl_GlobalInvocationID.xy);\n";
        break;
    case STAGE_VERTEX:
        css << "   ivec2 pos = ivec2(gl_VertexIndex % gridDim, gl_VertexIndex / gridDim);\n";
        break;
    case STAGE_FRAGMENT:
        css << "   ivec2 pos = ivec2(gl_FragCoord.xy);\n";
        break;
    default:
        DE_ASSERT(0);
    }

    // Compute mapping block size.
    // Compute thread's index in the mapping block.
    switch (m_params.mappingScope)
    {
    case SCOPE_SUBGROUP:
        css << "   uint size = gl_SubgroupSize;\n"
               "   uint index = gl_SubgroupInvocationID;\n";
        break;
    case SCOPE_WORKGROUP:
        css << "   uint size = DIM * DIM;\n"
               "   uint index = uint((pos.y % DIM) * DIM + pos.x % DIM);\n";
        break;
    default:
        css << "   uint size = gridDim * gridDim;\n"
               "   uint index = pos.y * gridDim + pos.x;\n";
    }

    // Compute partner thread indexes.
    // For a single partner, preserve the original logic.
    // For multiple partners, increment own index by stride for each partner.
    if (m_params.numPartners == 1)
    {
        css << "   uint indexPartner1 = size - 1u - index;\n"
               "   bool idle = false;\n";
    }
    else
    {
        // Assumes subgroupSize >= groupSize
        const uint32_t groupSize = m_params.numPartners + 1;
        css << "   uint stride = size / " << groupSize << "u;\n";
        for (uint32_t i = 1; i <= m_params.numPartners; i++)
        {
            css << "   uint indexPartner" << i << " = (index + " << i << "u * stride) % (" << groupSize
                << "u * stride);\n";
        }
        css << "   bool idle = index >= " << groupSize << "u * stride;\n";
    }

    // An invocation is non-idle only if all its partners are active
    if (m_params.mappingScope == SCOPE_SUBGROUP)
    {
        css << "   uvec4 sgBallot = subgroupBallot(true);\n";
        for (uint32_t i = 1; i <= m_params.numPartners; i++)
            css << "   idle = idle || !subgroupBallotBitExtract(sgBallot, indexPartner" << i << ");\n";
    }

    // Instead of returning from idle invocation, which could cause barrier divergence,
    // idle invocations are mapped to themselves
    if (m_params.numPartners > 1 || m_params.mappingScope == SCOPE_SUBGROUP)
    {
        css << "   if (idle) {\n";
        for (uint32_t i = 1; i <= m_params.numPartners; i++)
        {
            css << "      indexPartner" << i << " = index;\n";
        }
        css << "   }\n";
    }

    // Compute 2D positions of each partner
    for (uint32_t i = 1; i <= m_params.numPartners; i++)
    {
        const std::string index = "indexPartner" + std::to_string(i);
        css << "   ivec2 posPartner" << i << " = ";
        switch (m_params.mappingScope)
        {
        case SCOPE_SUBGROUP:
            css << "subgroupShuffle(pos, " << index << ");\n";
            break;
        case SCOPE_WORKGROUP:
            css << "pos - pos % DIM + ivec2(" << index << " % DIM, " << index << " / DIM);\n";
            break;
        default:
            css << "ivec2(" << index << " % gridDim, " << index << " / gridDim);\n";
        }
    }

    // Compute invocationId and coordinates for self and each partner
    for (uint32_t i = 0; i <= m_params.numPartners; i++)
    {
        const std::string suffix = (i == 0) ? "" : "Partner" + std::to_string(i);
        const std::string pos    = "pos" + suffix;

        css << "   uint invocationId" << suffix << " = getInvocationId(" << pos << ");\n";
        for (const auto &buffer : m_buffers)
            css << declareCoordinate(buffer, buffer.name + ("Coord" + suffix), pos);
    }

    if (m_params.stage == STAGE_VERTEX)
    {
        css << "   gl_PointSize = 1.0f;\n"
               "   gl_Position = vec4(0.0f, 0.0f, 0.0f, 1.0f);\n";
    }

    return css.str();
}

std::string MemoryModelTestCase::initLayoutWorkgroupCoordinates(const char *name) const
{
    DE_ASSERT((m_params.mappingScope == SCOPE_QUEUEFAMILY || m_params.mappingScope == SCOPE_DEVICE) &&
              m_params.stage == STAGE_COMPUTE && m_params.numPartners == 1);

    const BufferDef &buffer = getBuffer(name);
    DE_ASSERT(buffer.sc != SC_WORKGROUP);

    std::ostringstream css;

    for (uint32_t i = 1; i <= m_params.numPartners; i++)
    {
        const std::string suffix = "Partner" + std::to_string(i);
        const std::string pos    = "groupPos" + suffix;

        // Position of the first invocation in the partner's workgroup
        css << "   ivec2 " << pos << " = pos" << suffix << " - pos" << suffix << " % DIM;\n";
        css << declareCoordinate(buffer, buffer.name + ("GroupCoord" + suffix), pos);
    }

    return css.str();
}

void MemoryModelTestCase::checkSupportBuffers(const Context &context) const
{
    for (const auto &buffer : m_buffers)
    {
        if (buffer.type == DATA_TYPE_FLOAT64 && buffer.sc == SC_IMAGE)
            TCU_THROW(NotSupportedError,
                      "VkShaderAtomicFloat64: 64-bit floating point image atomic operations not supported");
    }

    if (hasDataType(DATA_TYPE_UINT64) && !context.getDeviceFeatures().shaderInt64)
        TCU_THROW(NotSupportedError, "64-bit integer in shaders not supported");

    if (hasStorageClass(SC_PHYSBUFFER) && !context.isBufferDeviceAddressSupported())
        TCU_THROW(NotSupportedError, "Physical storage buffer pointers not supported");

    const auto &vki                     = context.getInstanceInterface();
    const auto physDevice               = context.getPhysicalDevice();
    const bool isDeviceAddressSupported = context.isDeviceFunctionalitySupported("VK_KHR_buffer_device_address");

    const VkPhysicalDeviceMemoryProperties properties = getPhysicalDeviceMemoryProperties(vki, physDevice);

    for (const auto &buffer : m_buffers)
    {
        if (buffer.sc != SC_BUFFER && buffer.sc != SC_PHYSBUFFER)
            continue;

        const MemoryRequirement memReq = getBufferMemoryRequirement(buffer, isDeviceAddressSupported);
        if (getCompatibleMemoryTypes(properties, memReq) == 0)
            TCU_THROW(NotSupportedError, "Test variant uses memory type which is not supported");
    }
}

void MemoryModelTestCase::checkSupportBufferAtomics(const Context &context, const char *name) const
{
    const BufferDef &buffer = getBuffer(name);

    switch (buffer.type)
    {
    case DATA_TYPE_UINT64:
        if ((buffer.sc == SC_BUFFER || buffer.sc == SC_PHYSBUFFER) &&
            !context.getShaderAtomicInt64Features().shaderBufferInt64Atomics)
            TCU_THROW(NotSupportedError, "64-bit integer buffer atomics not supported");
        if (buffer.sc == SC_WORKGROUP && !context.getShaderAtomicInt64Features().shaderSharedInt64Atomics)
            TCU_THROW(NotSupportedError, "64-bit integer shared atomics not supported");
        break;

    case DATA_TYPE_FLOAT32:
        if (!context.isDeviceFunctionalitySupported("VK_EXT_shader_atomic_float"))
            TCU_THROW(NotSupportedError, "Missing extension: VK_EXT_shader_atomic_float");
        if ((buffer.sc == SC_BUFFER || buffer.sc == SC_PHYSBUFFER) &&
            !context.getShaderAtomicFloatFeaturesEXT().shaderBufferFloat32Atomics)
            TCU_THROW(NotSupportedError, "VkShaderAtomicFloat32: 32-bit floating point buffer atomics not supported");
        if (buffer.sc == SC_IMAGE && !context.getShaderAtomicFloatFeaturesEXT().shaderImageFloat32Atomics)
            TCU_THROW(NotSupportedError, "VkShaderAtomicFloat32: 32-bit floating point image atomics not supported");
        if (buffer.sc == SC_WORKGROUP && !context.getShaderAtomicFloatFeaturesEXT().shaderSharedFloat32Atomics)
            TCU_THROW(NotSupportedError, "VkShaderAtomicFloat32: 32-bit floating point shared atomics not supported");
        break;

    case DATA_TYPE_FLOAT64:
        if (!context.isDeviceFunctionalitySupported("VK_EXT_shader_atomic_float"))
            TCU_THROW(NotSupportedError, "Missing extension: VK_EXT_shader_atomic_float");
        if ((buffer.sc == SC_BUFFER || buffer.sc == SC_PHYSBUFFER) &&
            !context.getShaderAtomicFloatFeaturesEXT().shaderBufferFloat64Atomics)
            TCU_THROW(NotSupportedError, "VkShaderAtomicFloat64: 64-bit floating point buffer atomics not supported");
        // DATA_TYPE_FLOAT64 with SC_IMAGE is rejected in checkSupportBuffers
        if (buffer.sc == SC_WORKGROUP && !context.getShaderAtomicFloatFeaturesEXT().shaderSharedFloat64Atomics)
            TCU_THROW(NotSupportedError, "VkShaderAtomicFloat64: 64-bit floating point shared atomics not supported");
        break;

    default:
        break;
    }
}

void MemoryModelTestCase::checkSupportStageAtomics(const Context &context) const
{
    if (m_params.stage == STAGE_VERTEX && !context.getDeviceFeatures().vertexPipelineStoresAndAtomics)
        TCU_THROW(NotSupportedError, "vertexPipelineStoresAndAtomics not supported");
    if (m_params.stage == STAGE_FRAGMENT && !context.getDeviceFeatures().fragmentStoresAndAtomics)
        TCU_THROW(NotSupportedError, "fragmentStoresAndAtomics not supported");
}

void MemoryModelTestCase::checkSupportSubgroupFeatures(const Context &context) const
{
    const VkPhysicalDeviceSubgroupProperties &subgroupProperties = context.getSubgroupProperties();
    if (subgroupProperties.subgroupSize <= m_params.numPartners)
        TCU_THROW(NotSupportedError, "Device does not support required subgroup size");

    const VkSubgroupFeatureFlags requiredOperations =
        VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;

    if ((subgroupProperties.supportedOperations & requiredOperations) != requiredOperations)
        TCU_THROW(NotSupportedError, "Subgroup features not supported");

    VkShaderStageFlags flags = 0;
    if (m_params.stage == STAGE_COMPUTE)
        flags = VK_SHADER_STAGE_COMPUTE_BIT;
    else if (m_params.stage == STAGE_VERTEX)
        flags = VK_SHADER_STAGE_VERTEX_BIT;
    else if (m_params.stage == STAGE_FRAGMENT)
        flags = VK_SHADER_STAGE_FRAGMENT_BIT;

    if ((subgroupProperties.supportedStages & flags) == 0)
        TCU_THROW(NotSupportedError, "Device does not support subgroup operations for this stage");
}

void MemoryModelTestCase::checkSupportSplitBarriers(const Context &context) const
{
    if (m_params.stage != STAGE_COMPUTE)
        TCU_THROW(NotSupportedError, "VK_EXT_shader_split_barrier is only supported in compute shaders");

#ifndef CTS_USES_VULKANSC
    if (!context.getShaderSplitBarrierFeaturesEXT().shaderSplitBarrier)
        TCU_THROW(NotSupportedError, "shaderSplitBarrier feature not supported");
#else
    DE_UNREF(context);
    TCU_THROW(NotSupportedError, "shaderSplitBarrier feature not supported");
#endif
}

const char *MemoryModelTestCase::gl_barrier()
{
    return "barrier()";
}

std::string MemoryModelTestCase::gl_controlBarrier(const Scope scope1, const Scope scope2, const uint32_t sem)
{
    std::ostringstream css;
    css << "controlBarrier(" << gl_scope(scope1) << ", " << gl_scope(scope2) << ", " << gl_semantics1(sem) << ", "
        << gl_semantics2(sem) << ")";
    return css.str();
}

std::string MemoryModelTestCase::gl_controlBarrierArrive(const Scope scope1, const Scope scope2, const uint32_t sem)
{
    std::ostringstream css;
    css << "controlBarrierArrive(" << gl_scope(scope1) << ", " << gl_scope(scope2) << ", " << gl_semantics1(sem) << ", "
        << gl_semantics2(sem) << ")";
    return css.str();
}

std::string MemoryModelTestCase::gl_controlBarrierWait(const Scope scope1, const Scope scope2, const uint32_t sem)
{
    std::ostringstream css;
    css << "controlBarrierWait(" << gl_scope(scope1) << ", " << gl_scope(scope2) << ", " << gl_semantics1(sem) << ", "
        << gl_semantics2(sem) << ")";
    return css.str();
}

std::string MemoryModelTestCase::gl_memoryBarrier(const Scope scope, const uint32_t sem)
{
    std::ostringstream css;
    css << "memoryBarrier(" << gl_scope(scope) << ", " << gl_semantics1(sem) << ", " << gl_semantics2(sem) << ")";
    return css.str();
}

std::string MemoryModelTestCase::gl_atomicLoad(const char *name, const char *idx, const Scope scope,
                                               const uint32_t sem) const
{
    const BufferDef &buffer = getBuffer(name);
    std::ostringstream css;
    if (buffer.sc == SC_IMAGE)
        css << "imageAtomicLoad(" << buffer.name << ", " << idx << ", ";
    else
        css << "atomicLoad(" << buffer.name << ".x[" << idx << "], ";
    css << gl_scope(scope) << ", " << gl_semantics1(sem) << ", " << gl_semantics2(sem) << ")";
    return css.str();
}

std::string MemoryModelTestCase::gl_atomicStore(const char *name, const char *idx, const char *value, const Scope scope,
                                                const uint32_t sem) const
{
    const BufferDef &buffer = getBuffer(name);
    const bool noCast       = is64BitType(buffer.type) && buffer.sc == SC_IMAGE;
    const char *typeStr     = noCast ? "" : gl_type(buffer.type);

    std::ostringstream css;
    if (buffer.sc == SC_IMAGE)
        css << "imageAtomicStore(" << buffer.name << ", " << idx << ", ";
    else
        css << "atomicStore(" << buffer.name << ".x[" << idx << "], ";
    css << typeStr << "(" << value << "), " << gl_scope(scope) << ", " << gl_semantics1(sem) << ", "
        << gl_semantics2(sem) << ")";
    return css.str();
}

std::string MemoryModelTestCase::gl_atomicExchange(const char *name, const char *idx, const char *value,
                                                   const Scope scope, const uint32_t sem) const
{
    const BufferDef &buffer = getBuffer(name);
    const bool noCast       = is64BitType(buffer.type) && buffer.sc == SC_IMAGE;
    const char *typeStr     = noCast ? "" : gl_type(buffer.type);

    std::ostringstream css;
    if (buffer.sc == SC_IMAGE)
        css << "imageAtomicExchange(" << buffer.name << ", " << idx << ", ";
    else
        css << "atomicExchange(" << buffer.name << ".x[" << idx << "], ";
    css << typeStr << "(" << value << "), " << gl_scope(scope) << ", " << gl_semantics1(sem) << ", "
        << gl_semantics2(sem) << ")";
    return css.str();
}

std::string MemoryModelTestCase::gl_load(const char *name, const char *idx) const
{
    const BufferDef &buffer = getBuffer(name);
    std::ostringstream css;
    if (buffer.sc == SC_IMAGE)
        css << "imageLoad(" << buffer.name << ", " << idx << ").x";
    else
        css << buffer.name << ".x[" << idx << "]";
    return css.str();
}

std::string MemoryModelTestCase::gl_store(const char *name, const char *idx, const char *value) const
{
    const BufferDef &buffer = getBuffer(name);
    std::ostringstream css;
    if (buffer.sc != SC_IMAGE)
        css << buffer.name << ".x[" << idx << "] = " << gl_type(buffer.type) << "(" << value << ")";
    else if (isIntType(buffer.type))
        css << "imageStore(" << buffer.name << ", " << idx << ", uvec4(" << value << ", 0, 0, 0))";
    else
        css << "imageStore(" << buffer.name << ", " << idx << ", vec4(" << gl_type(buffer.type) << "(" << value
            << "), 0, 0, 0))";
    return css.str();
}

const char *MemoryModelTestCase::gl_type(const DataType type)
{
    switch (type)
    {
    case DATA_TYPE_UINT:
        return "uint";
    case DATA_TYPE_UINT64:
        return "uint64_t";
    case DATA_TYPE_FLOAT32:
        return "float";
    case DATA_TYPE_FLOAT64:
        return "double";
    default:
        DE_ASSERT(0);
        return "";
    }
}

const char *MemoryModelTestCase::gl_scope(const Scope scope)
{
    switch (scope)
    {
    case SCOPE_DEVICE:
        return "gl_ScopeDevice";
    case SCOPE_QUEUEFAMILY:
        return "gl_ScopeQueueFamily";
    case SCOPE_WORKGROUP:
        return "gl_ScopeWorkgroup";
    case SCOPE_SUBGROUP:
        return "gl_ScopeSubgroup";
    case SCOPE_INVOCATION:
        return "gl_ScopeInvocation";
    default:
        DE_ASSERT(0);
        return "";
    }
}

std::string MemoryModelTestCase::gl_toUint(const DataType type, const std::string &expr)
{
    switch (type)
    {
    case DATA_TYPE_UINT:
        return expr;
    case DATA_TYPE_UINT64:
        return "uint(" + expr + ")";
    case DATA_TYPE_FLOAT32:
        return "floatBitsToUint(" + expr + ")";
    case DATA_TYPE_FLOAT64:
        return "floatBitsToUint(float(" + expr + "))";
    default:
        DE_ASSERT(0);
        return expr;
    }
}

const BufferDef &MemoryModelTestCase::getBuffer(const char *name) const
{
    for (const auto &buffer : m_buffers)
    {
        if (name == std::string_view(buffer.name))
            return buffer;
    }
    TCU_THROW(InternalError, std::string("Unknown buffer: ") + name);
}

bool MemoryModelTestCase::hasStorageClass(const StorageClass sc) const
{
    return std::any_of(m_buffers.begin(), m_buffers.end(), [sc](const BufferDef &b) { return b.sc == sc; });
}

bool MemoryModelTestCase::hasDataType(const DataType type) const
{
    return std::any_of(m_buffers.begin(), m_buffers.end(), [type](const BufferDef &b) { return b.type == type; });
}

uint32_t MemoryModelTestCase::getBufferSemantics(const char *name) const
{
    return scToSem(getBuffer(name).sc);
}

namespace
{

class MemoryModelTestInstance : public vkt::MultiQueueRunnerTestInstance
{
public:
    MemoryModelTestInstance(Context &context, Stage stage, const std::vector<BufferDef> &buffers)
        : vkt::MultiQueueRunnerTestInstance(context, stage == STAGE_COMPUTE ? vkt::COMPUTE_QUEUE : vkt::GRAPHICS_QUEUE)
        , m_stage(stage)
        , m_buffers(buffers)
    {
    }

    tcu::TestStatus queuePass(const vkt::QueueData &queueData) override;

private:
    const Stage m_stage;
    const std::vector<BufferDef> m_buffers;
};

tcu::TestStatus MemoryModelTestInstance::queuePass(const vkt::QueueData &queueData)
{
    const DeviceInterface &vk = m_context.getDeviceInterface();
    const VkDevice device     = m_context.getDevice();
    Allocator &allocator      = m_context.getDefaultAllocator();

    const VkShaderStageFlags allShaderStages =
        VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    const VkPipelineStageFlags allPipelineStages =
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
        (m_stage == STAGE_COMPUTE ? 0 : (VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT));

    uint32_t DIM                         = 31;
    const VkPhysicalDeviceLimits &limits = m_context.getDeviceProperties().limits;
    // If necessary, shrink workgroup size to fit HW limits
    if (DIM * DIM > limits.maxComputeWorkGroupInvocations)
    {
        DIM = (uint32_t)deFloatSqrt((float)limits.maxComputeWorkGroupInvocations);
    }
    const uint32_t NUM_WORKGROUP_EACH_DIM = 8;
    const uint32_t GRID_DIM               = DIM * NUM_WORKGROUP_EACH_DIM;
    const uint32_t NUM_INVOCATIONS        = GRID_DIM * GRID_DIM;

    // Last buffer is the result buffer
    const uint32_t numBuffers      = static_cast<uint32_t>(m_buffers.size());
    const uint32_t idxResultBuffer = numBuffers - 1;
    DE_ASSERT(!m_buffers.empty() && m_buffers[idxResultBuffer].sc == SC_BUFFER &&
              m_buffers[idxResultBuffer].type == DATA_TYPE_UINT);

    std::vector<VkDeviceSize> bufferSizes(numBuffers);
    std::vector<de::MovePtr<BufferWithMemory>> buffers(numBuffers);
    std::vector<VkDeviceAddress> bufferAddresses(numBuffers);
    de::MovePtr<BufferWithMemory> copyBuffer;

    const bool isDeviceAddressSupported = m_context.isDeviceFunctionalitySupported("VK_KHR_buffer_device_address");
    for (uint32_t i = 0; i < numBuffers; ++i)
    {
        const BufferDef &buffer  = m_buffers[i];
        const size_t elementSize = is64BitType(buffer.type) ? sizeof(uint64_t) : sizeof(uint32_t);
        bufferSizes[i]           = NUM_INVOCATIONS * elementSize;
        if (buffer.sc != SC_BUFFER && buffer.sc != SC_PHYSBUFFER)
            continue;

        VkBufferUsageFlags usageFlags =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (buffer.sc == SC_PHYSBUFFER)
            usageFlags |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

        const MemoryRequirement memReq = getBufferMemoryRequirement(buffer, isDeviceAddressSupported);
        buffers[i]                     = de::MovePtr<BufferWithMemory>(
            new BufferWithMemory(vk, device, allocator, makeBufferCreateInfo(bufferSizes[i], usageFlags), memReq));
        if (buffer.sc == SC_PHYSBUFFER)
            bufferAddresses[i] = getBufferDeviceAddress(vk, device, **buffers[i]);
    }

    // Try to use cached host memory for the buffer the CPU will read from, else fallback to host visible.
    try
    {
        copyBuffer = de::MovePtr<BufferWithMemory>(new BufferWithMemory(
            vk, device, allocator, makeBufferCreateInfo(bufferSizes[idxResultBuffer], VK_BUFFER_USAGE_TRANSFER_DST_BIT),
            MemoryRequirement::HostVisible | MemoryRequirement::Cached));
    }
    catch (const tcu::NotSupportedError &)
    {
        copyBuffer = de::MovePtr<BufferWithMemory>(new BufferWithMemory(
            vk, device, allocator, makeBufferCreateInfo(bufferSizes[idxResultBuffer], VK_BUFFER_USAGE_TRANSFER_DST_BIT),
            MemoryRequirement::HostVisible));
    }

    const VkImageSubresourceRange range = makeImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u);
    std::vector<de::MovePtr<ImageWithMemory>> images(numBuffers);
    std::vector<Move<VkImageView>> imageViews(numBuffers);

    for (uint32_t i = 0; i < numBuffers; ++i)
    {
        if (m_buffers[i].sc != SC_IMAGE)
            continue;

        const VkFormat imageFormat = isIntType(m_buffers[i].type) ? VK_FORMAT_R32_UINT : VK_FORMAT_R32_SFLOAT;
        const VkImageUsageFlags usageFlags =
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        const MemoryRequirement memReq = m_buffers[i].local ? MemoryRequirement::Local : MemoryRequirement::NonLocal;

        try
        {
            images[i] = de::MovePtr<ImageWithMemory>(new ImageWithMemory(
                vk, device, allocator, makeImageCreateInfo(imageFormat, GRID_DIM, GRID_DIM, usageFlags), memReq));
        }
        catch (const tcu::NotSupportedError &)
        {
            if (!m_buffers[i].local)
            {
                TCU_THROW(NotSupportedError, "Test variant uses non-device-local memory, which is not supported");
            }
            throw;
        }

        imageViews[i] = makeImageView(vk, device, **images[i], VK_IMAGE_VIEW_TYPE_2D, imageFormat, range);
    }

    DescriptorSetLayoutBuilder layoutBuilder;
    for (uint32_t i = 0; i < numBuffers; ++i)
    {
        const VkDescriptorType descType =
            (m_buffers[i].sc == SC_IMAGE) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBuilder.addSingleBinding(descType, allShaderStages);
    }

    const Unique<VkDescriptorSetLayout> descriptorSetLayout(layoutBuilder.build(vk, device));

    const Unique<VkDescriptorPool> descriptorPool(
        DescriptorPoolBuilder()
            .addType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, numBuffers)
            .addType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, numBuffers)
            .build(vk, device, VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, 1u));

    const Unique<VkDescriptorSet> descriptorSet(makeDescriptorSet(vk, device, *descriptorPool, *descriptorSetLayout));

    DescriptorSetUpdateBuilder setUpdateBuilder;
    for (uint32_t i = 0; i < numBuffers; ++i)
    {
        if (m_buffers[i].sc == SC_BUFFER)
        {
            const VkDescriptorBufferInfo bufferInfo = makeDescriptorBufferInfo(**buffers[i], 0, bufferSizes[i]);
            setUpdateBuilder.writeSingle(*descriptorSet, DescriptorSetUpdateBuilder::Location::binding(i),
                                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &bufferInfo);
        }
        else if (m_buffers[i].sc == SC_IMAGE)
        {
            const VkDescriptorImageInfo imageInfo =
                makeDescriptorImageInfo(VK_NULL_HANDLE, *imageViews[i], VK_IMAGE_LAYOUT_GENERAL);
            setUpdateBuilder.writeSingle(*descriptorSet, DescriptorSetUpdateBuilder::Location::binding(i),
                                         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageInfo);
        }
    }
    setUpdateBuilder.update(vk, device);

    const VkPushConstantRange pushConstRange = {
        allShaderStages,                       // VkShaderStageFlags stageFlags
        0,                                     // uint32_t offset
        idxResultBuffer * DEVICE_ADDRESS_SIZE, // uint32_t size
    };

    const Unique<VkPipelineLayout> pipelineLayout(
        makePipelineLayout(vk, device, *descriptorSetLayout, idxResultBuffer > 0 ? &pushConstRange : nullptr));

    Move<VkPipeline> pipeline;
    Move<VkRenderPass> renderPass;
    Move<VkFramebuffer> framebuffer;

    const VkPipelineBindPoint bindPoint =
        m_stage == STAGE_COMPUTE ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS;

    const uint32_t specData[2]                = {DIM, NUM_WORKGROUP_EACH_DIM};
    const VkSpecializationMapEntry entries[2] = {
        makeSpecializationMapEntry(0u, 0u, sizeof(uint32_t)),
        makeSpecializationMapEntry(1u, sizeof(uint32_t), sizeof(uint32_t)),
    };

    const VkSpecializationInfo specInfo = makeSpecializationInfo(2u, entries, sizeof(specData), specData);

    if (m_stage == STAGE_COMPUTE)
    {
        const Unique<VkShaderModule> shader(
            createShaderModule(vk, device, m_context.getBinaryCollection().get("test"), 0));
        pipeline = makeComputePipeline(vk, device, *pipelineLayout, 0u, nullptr, *shader, 0u, &specInfo);
    }
    else
    {
        // Vertex tests run "test" as the only shader stage, fragment tests pair it with the fullscreen "vert" shader
        const bool isVertexStage = (m_stage == STAGE_VERTEX);
        const Unique<VkShaderModule> vertexShader(
            createShaderModule(vk, device, m_context.getBinaryCollection().get(isVertexStage ? "test" : "vert"), 0));
        const Unique<VkShaderModule> fragmentShader(
            isVertexStage ? Move<VkShaderModule>() :
                            createShaderModule(vk, device, m_context.getBinaryCollection().get("test"), 0));

        const VkViewport viewport = makeViewport(GRID_DIM, GRID_DIM);
        const VkRect2D scissor    = makeRect2D(GRID_DIM, GRID_DIM);
        const VkPrimitiveTopology topology =
            isVertexStage ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        const VkBool32 rasterizerDiscardEnable = isVertexStage ? VK_TRUE : VK_FALSE;

        renderPass  = makeRenderPass(vk, device);
        framebuffer = makeFramebuffer(vk, device, *renderPass, 0, nullptr, GRID_DIM, GRID_DIM);
        pipeline    = makeGraphicsPipeline(vk, device, *pipelineLayout, *vertexShader, *fragmentShader, *renderPass,
                                           &viewport, &scissor, topology, rasterizerDiscardEnable, &specInfo);
    }

    const Unique<VkCommandPool> cmdPool(makeCommandPool(vk, device, queueData.familyIndex));
    const Unique<VkCommandBuffer> cmdBuffer(
        allocateCommandBuffer(vk, device, *cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY));
    const VkClearValue clearColor = makeClearValueColorU32(0, 0, 0, 0);
    const VkBufferCopy copyParams = makeBufferCopy(0u, 0u, bufferSizes[idxResultBuffer]);

    constexpr uint32_t NUM_SUBMITS    = 4;
    constexpr uint32_t NUM_ITERATIONS = 50;

    for (uint32_t x = 0; x < NUM_SUBMITS; ++x)
    {
        beginCommandBuffer(vk, *cmdBuffer, 0u);

        if (x == 0)
            vk.cmdFillBuffer(*cmdBuffer, **buffers[idxResultBuffer], 0, bufferSizes[idxResultBuffer], 0);

        for (uint32_t i = 0; i < numBuffers; ++i)
        {
            if (m_buffers[i].sc != SC_IMAGE)
                continue;

            const VkImageMemoryBarrier imageBarrier =
                makeImageMemoryBarrier(0u, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                                       VK_IMAGE_LAYOUT_GENERAL, **images[i], range);
            cmdPipelineImageMemoryBarrier(vk, *cmdBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                          VK_PIPELINE_STAGE_TRANSFER_BIT, &imageBarrier);
        }

        vk.cmdBindDescriptorSets(*cmdBuffer, bindPoint, *pipelineLayout, 0u, 1, &*descriptorSet, 0u, nullptr);
        vk.cmdBindPipeline(*cmdBuffer, bindPoint, *pipeline);

        for (uint32_t i = 0; i < numBuffers; ++i)
        {
            if (m_buffers[i].sc != SC_PHYSBUFFER)
                continue;

            vk.cmdPushConstants(*cmdBuffer, *pipelineLayout, allShaderStages, DEVICE_ADDRESS_SIZE * i,
                                DEVICE_ADDRESS_SIZE, &bufferAddresses[i]);
        }

        for (uint32_t iters = 0; iters < NUM_ITERATIONS; ++iters)
        {
            for (uint32_t i = 0; i < idxResultBuffer; ++i)
            {
                if (buffers[i])
                    vk.cmdFillBuffer(*cmdBuffer, **buffers[i], 0, bufferSizes[i], 0);
                if (images[i])
                    vk.cmdClearColorImage(*cmdBuffer, **images[i], VK_IMAGE_LAYOUT_GENERAL, &clearColor.color, 1,
                                          &range);
            }

            const VkMemoryBarrier transferToShader =
                makeMemoryBarrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            cmdPipelineMemoryBarrier(vk, *cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, allPipelineStages,
                                     &transferToShader);

            if (m_stage == STAGE_COMPUTE)
            {
                vk.cmdDispatch(*cmdBuffer, NUM_WORKGROUP_EACH_DIM, NUM_WORKGROUP_EACH_DIM, 1);
            }
            else
            {
                beginRenderPass(vk, *cmdBuffer, *renderPass, *framebuffer, makeRect2D(GRID_DIM, GRID_DIM), 0, nullptr,
                                VK_SUBPASS_CONTENTS_INLINE);
                // Draw a point cloud for vertex shader testing, and a single quad for fragment shader testing
                if (m_stage == STAGE_VERTEX)
                {
                    vk.cmdDraw(*cmdBuffer, NUM_INVOCATIONS, 1u, 0u, 0u);
                }
                else
                {
                    vk.cmdDraw(*cmdBuffer, 4u, 1u, 0u, 0u);
                }
                endRenderPass(vk, *cmdBuffer);
            }

            const VkMemoryBarrier shaderToTransfer =
                makeMemoryBarrier(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                  VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            cmdPipelineMemoryBarrier(vk, *cmdBuffer, allPipelineStages, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     &shaderToTransfer);
        }

        if (x == NUM_SUBMITS - 1)
        {
            vk.cmdCopyBuffer(*cmdBuffer, **buffers[idxResultBuffer], **copyBuffer, 1, &copyParams);
            const VkMemoryBarrier transferToHost =
                makeMemoryBarrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            cmdPipelineMemoryBarrier(vk, *cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                     &transferToHost);
        }

        endCommandBuffer(vk, *cmdBuffer);
        submitCommandsAndWait(vk, device, queueData.handle, cmdBuffer.get());
        m_context.resetCommandPoolForVKSC(device, *cmdPool);
    }

    tcu::TestLog &log = m_context.getTestContext().getLog();

    invalidateAlloc(vk, device, copyBuffer->getAllocation());
    const uint32_t *ptr = static_cast<const uint32_t *>(copyBuffer->getAllocation().getHostPtr());
    qpTestResult res    = QP_TEST_RESULT_PASS;

    constexpr uint32_t MAX_LOGGED_ERRORS = 256;
    uint32_t numErrors                   = 0;
    for (uint32_t i = 0; i < NUM_INVOCATIONS; ++i)
    {
        if (ptr[i] != 0)
        {
            if (numErrors < MAX_LOGGED_ERRORS)
            {
                log << tcu::TestLog::Message << "Failed invocation: " << i << tcu::TestLog::EndMessage;
            }
            numErrors++;
            res = QP_TEST_RESULT_FAIL;
        }
    }

    if (numErrors)
    {
        log << tcu::TestLog::Message << "Total Errors: " << numErrors << tcu::TestLog::EndMessage;
    }

    return tcu::TestStatus(res, qpGetTestResultName(res));
}

} // namespace

TestInstance *MemoryModelTestCase::createInstance(Context &context) const
{
    return new MemoryModelTestInstance(context, m_params.stage, m_buffers);
}

} // namespace MemoryModel
} // namespace vkt
