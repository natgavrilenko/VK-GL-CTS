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
 * \brief Memory Model Pattern Test Instance
 *//*--------------------------------------------------------------------*/

#include "vktMemoryModelPatternTestInstance.hpp"

#include "vkBufferWithMemory.hpp"
#include "vkImageWithMemory.hpp"
#include "vkQueryUtil.hpp"
#include "vkBuilderUtil.hpp"
#include "vkCmdUtil.hpp"
#include "vkTypeUtil.hpp"
#include "vkObjUtil.hpp"

#include "deDefs.h"
#include "deMath.h"
#include "deSharedPtr.hpp"
#include "deString.h"

#include "tcuDefs.hpp"
#include "tcuTestCase.hpp"
#include "tcuTestLog.hpp"

namespace vkt
{
namespace MemoryModel
{
using namespace vk;

MemoryModelTestInstance::MemoryModelTestInstance(Context &context, Stage stage, const std::vector<BufferDef> &buffers)
    : vkt::MultiQueueRunnerTestInstance(context, stage == STAGE_COMPUTE ? vkt::COMPUTE_QUEUE : vkt::GRAPHICS_QUEUE)
    , m_stage(stage)
    , m_buffers(buffers)
{
}

tcu::TestStatus MemoryModelTestInstance::queuePass(const vkt::QueueData &queueData)
{
    const DeviceInterface &vk = m_context.getDeviceInterface();
    const VkDevice device     = m_context.getDevice();
    Allocator &allocator      = m_context.getDefaultAllocator();

    const VkFlags allShaderStages   = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT |
                                      VK_SHADER_STAGE_FRAGMENT_BIT;
    const VkFlags allPipelineStages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                      (m_stage == STAGE_COMPUTE ?
                                           0 :
                                           (VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT));

    VkPhysicalDeviceProperties2 properties;
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = nullptr;

    m_context.getInstanceInterface().getPhysicalDeviceProperties2(m_context.getPhysicalDevice(), &properties);

    uint32_t DIM                    = 31;
    uint32_t NUM_WORKGROUP_EACH_DIM = 8;
    // If necessary, shrink workgroup size to fit HW limits
    if (DIM * DIM > properties.properties.limits.maxComputeWorkGroupInvocations)
    {
        DIM = (uint32_t)deFloatSqrt((float)properties.properties.limits.maxComputeWorkGroupInvocations);
    }
    uint32_t NUM_INVOCATIONS = (DIM * DIM * NUM_WORKGROUP_EACH_DIM * NUM_WORKGROUP_EACH_DIM);

    // The last buffer is result buffer
    const uint32_t numBuffers = static_cast<uint32_t>(m_buffers.size());
    const uint32_t idxResultBuffer = numBuffers - 1;

    std::vector<VkDeviceSize> bufferSizes(numBuffers);
    std::vector<de::MovePtr<BufferWithMemory>> buffers(numBuffers);
    std::vector<VkDescriptorBufferInfo> bufferDescriptors(numBuffers);
    de::MovePtr<BufferWithMemory> copyBuffer;

    const bool isDeviceAddressSupported = m_context.isDeviceFunctionalitySupported("VK_KHR_buffer_device_address");
    for (uint32_t i = 0; i < numBuffers; ++i)
    {
        const BufferDef &buffer = m_buffers[i];
        size_t elementSize = (buffer.type == DATA_TYPE_UINT64 || buffer.type == DATA_TYPE_FLOAT64) ?
                                 sizeof(uint64_t) :
                                 sizeof(uint32_t);

        bufferSizes[i] = NUM_INVOCATIONS * elementSize;
        if (buffer.sc != SC_BUFFER && buffer.sc != SC_PHYSBUFFER)
            continue;

        VkFlags usageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bool memoryDeviceAddress = false;
        if (buffer.sc == SC_PHYSBUFFER)
        {
            usageFlags |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            memoryDeviceAddress = isDeviceAddressSupported;
        }
        buffers[i] = de::MovePtr<BufferWithMemory>(
            new BufferWithMemory(vk, device, allocator, makeBufferCreateInfo(bufferSizes[i], usageFlags),
                                 (memoryDeviceAddress ? MemoryRequirement::DeviceAddress : MemoryRequirement::Any) |
                                 (buffer.local ? MemoryRequirement::Local : MemoryRequirement::NonLocal)));
        bufferDescriptors[i] = makeDescriptorBufferInfo(**buffers[i], 0, bufferSizes[i]);
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

    std::vector<de::MovePtr<ImageWithMemory>> images(numBuffers);
    std::vector<Move<VkImageView>> imageViews(numBuffers);
    std::vector<VkDescriptorImageInfo> imageDescriptors(numBuffers);

    for (uint32_t i = 0; i < numBuffers; ++i)
    {
        if (m_buffers[i].sc != SC_IMAGE)
            continue;

        const bool local = m_buffers[i].local;
        const VkFormat imageFormat = (m_buffers[i].type == DATA_TYPE_UINT || m_buffers[i].type == DATA_TYPE_UINT64) ? VK_FORMAT_R32_UINT : VK_FORMAT_R32_SFLOAT;

        const VkImageCreateInfo imageCreateInfo = {
            VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, // VkStructureType sType;
            nullptr,                             // const void* pNext;
            (VkImageCreateFlags)0u,              // VkImageCreateFlags flags;
            VK_IMAGE_TYPE_2D,                    // VkImageType imageType;
            imageFormat,                         // VkFormat format;
            {
                DIM * NUM_WORKGROUP_EACH_DIM, // uint32_t width;
                DIM * NUM_WORKGROUP_EACH_DIM, // uint32_t height;
                1u                            // uint32_t depth;
            },                                // VkExtent3D    extent;
            1u,                               // uint32_t  mipLevels;
            1u,                               // uint32_t  arrayLayers;
            VK_SAMPLE_COUNT_1_BIT,            // VkSampleCountFlagBits samples;
            VK_IMAGE_TILING_OPTIMAL,          // VkImageTiling tiling;
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT, // VkImageUsageFlags usage;
            VK_SHARING_MODE_EXCLUSIVE,           // VkSharingMode sharingMode;
            0u,                                  // uint32_t  queueFamilyIndexCount;
            nullptr,                             // const uint32_t*   pQueueFamilyIndices;
            VK_IMAGE_LAYOUT_UNDEFINED            // VkImageLayout initialLayout;
        };
        VkImageViewCreateInfo imageViewCreateInfo = {
            VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, // VkStructureType sType;
            nullptr,                                  // const void* pNext;
            (VkImageViewCreateFlags)0u,               // VkImageViewCreateFlags  flags;
            VK_NULL_HANDLE,                           // VkImage image;
            VK_IMAGE_VIEW_TYPE_2D,                    // VkImageViewType viewType;
            imageFormat,                              // VkFormat format;
            {
                VK_COMPONENT_SWIZZLE_R, // VkComponentSwizzle r;
                VK_COMPONENT_SWIZZLE_G, // VkComponentSwizzle g;
                VK_COMPONENT_SWIZZLE_B, // VkComponentSwizzle b;
                VK_COMPONENT_SWIZZLE_A  // VkComponentSwizzle a;
            },                          // VkComponentMapping  components;
            {
                VK_IMAGE_ASPECT_COLOR_BIT, // VkImageAspectFlags aspectMask;
                0u,                        // uint32_t   baseMipLevel;
                1u,                        // uint32_t   levelCount;
                0u,                        // uint32_t   baseArrayLayer;
                1u                         // uint32_t   layerCount;
            }                              // VkImageSubresourceRange subresourceRange;
        };

        try
        {
            images[i] = de::MovePtr<ImageWithMemory>(
                new ImageWithMemory(vk, device, allocator, imageCreateInfo,
                                    local ? MemoryRequirement::Local : MemoryRequirement::NonLocal));
        }
        catch (const tcu::NotSupportedError &)
        {
            if (!local)
            {
                TCU_THROW(NotSupportedError, "Test variant uses non-device-local memory, which is not supported");
            }
            throw;
        }
        imageViewCreateInfo.image = **images[i];
        imageViews[i] = createImageView(vk, device, &imageViewCreateInfo, nullptr);
        imageDescriptors[i] = makeDescriptorImageInfo(VK_NULL_HANDLE, *imageViews[i], VK_IMAGE_LAYOUT_GENERAL);
    }

    DescriptorSetLayoutBuilder layoutBuilder;

    for (uint32_t i = 0; i < numBuffers; ++i)
    {
        const VkDescriptorType descType = (m_buffers[i].sc == SC_IMAGE) ?
                                              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE :
                                              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBuilder.addSingleBinding(descType, allShaderStages);
    }
    Unique<VkDescriptorSetLayout> descriptorSetLayout(layoutBuilder.build(vk, device));

    Unique<VkDescriptorPool> descriptorPool(
        DescriptorPoolBuilder()
            .addType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, numBuffers)
            .addType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, numBuffers)
            .build(vk, device, VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, 1u));
    Unique<VkDescriptorSet> descriptorSet(makeDescriptorSet(vk, device, *descriptorPool, *descriptorSetLayout));

    DescriptorSetUpdateBuilder setUpdateBuilder;
    for (uint32_t i = 0; i < numBuffers; ++i)
    {
        if (m_buffers[i].sc == SC_BUFFER)
        {
            setUpdateBuilder.writeSingle(*descriptorSet, DescriptorSetUpdateBuilder::Location::binding(i),
                                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &bufferDescriptors[i]);
        }
        else if (m_buffers[i].sc == SC_IMAGE)
        {
            setUpdateBuilder.writeSingle(*descriptorSet, DescriptorSetUpdateBuilder::Location::binding(i),
                                         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageDescriptors[i]);
        }
    }
    setUpdateBuilder.update(vk, device);

    const VkPushConstantRange pushConstRange = {
        allShaderStages,                                         // VkShaderStageFlags stageFlags
        0,                                                       // uint32_t           offset
        static_cast<uint32_t>(idxResultBuffer * sizeof(VkDeviceSize)) // uint32_t           size
    };

    const VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,      // sType
        nullptr,                                            // pNext
        (VkPipelineLayoutCreateFlags)0,
        1,                                                  // setLayoutCount
        &descriptorSetLayout.get(),                         // pSetLayouts
        idxResultBuffer > 0 ? 1u : 0u,                      // pushConstantRangeCount
        idxResultBuffer > 0 ? &pushConstRange : nullptr,    // pPushConstantRanges
    };

    Move<VkPipelineLayout> pipelineLayout = createPipelineLayout(vk, device, &pipelineLayoutCreateInfo, nullptr);

    Move<VkPipeline> pipeline;
    Move<VkRenderPass> renderPass;
    Move<VkFramebuffer> framebuffer;

    VkPipelineBindPoint bindPoint =
        m_stage == STAGE_COMPUTE ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS;

    const uint32_t specData[2] = {DIM, NUM_WORKGROUP_EACH_DIM};

    const VkSpecializationMapEntry entries[2] = {
        {0, sizeof(uint32_t) * 0, sizeof(uint32_t)},
        {1, sizeof(uint32_t) * 1, sizeof(uint32_t)},
    };

    const VkSpecializationInfo specInfo = {
        2,                // mapEntryCount
        entries,          // pMapEntries
        sizeof(specData), // dataSize
        specData          // pData
    };

    if (m_stage == STAGE_COMPUTE)
    {
        const Unique<VkShaderModule> shader(
            createShaderModule(vk, device, m_context.getBinaryCollection().get("test"), 0));

        const VkPipelineShaderStageCreateInfo shaderCreateInfo = {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            (VkPipelineShaderStageCreateFlags)0,
            VK_SHADER_STAGE_COMPUTE_BIT, // stage
            *shader,                     // shader
            "main",
            &specInfo, // pSpecializationInfo
        };

        const VkComputePipelineCreateInfo pipelineCreateInfo = {
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            nullptr,
            0u,               // flags
            shaderCreateInfo, // cs
            *pipelineLayout,  // layout
            VK_NULL_HANDLE,   // basePipelineHandle
            0u,               // basePipelineIndex
        };
        pipeline = createComputePipeline(vk, device, VK_NULL_HANDLE, &pipelineCreateInfo, nullptr);
    }
    else
    {

        const VkSubpassDescription subpassDesc = {
            (VkSubpassDescriptionFlags)0,
            VK_PIPELINE_BIND_POINT_GRAPHICS, // pipelineBindPoint
            0u,                                  // inputCount
            nullptr,                             // pInputAttachments
            0u,                                  // colorCount
            nullptr,                             // pColorAttachments
            nullptr,                             // pResolveAttachments
            nullptr,                             // depthStencilAttachment
            0u,                                  // preserveCount
            nullptr,                             // pPreserveAttachments
        };
        const VkRenderPassCreateInfo renderPassParams = {
            VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, // sType
            nullptr,                                       // pNext
            (VkRenderPassCreateFlags)0,
            0u,           // attachmentCount
            nullptr,      // pAttachments
            1u,           // subpassCount
            &subpassDesc, // pSubpasses
            0u,           // dependencyCount
            nullptr,      // pDependencies
        };

        renderPass = createRenderPass(vk, device, &renderPassParams);

        const VkFramebufferCreateInfo framebufferParams = {
            VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, // sType
            nullptr,                                       // pNext
            (VkFramebufferCreateFlags)0,
            *renderPass,                  // renderPass
            0u,                           // attachmentCount
            nullptr,                      // pAttachments
            DIM * NUM_WORKGROUP_EACH_DIM, // width
            DIM * NUM_WORKGROUP_EACH_DIM, // height
            1u,                           // layers
        };

        framebuffer = createFramebuffer(vk, device, &framebufferParams);

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
            (VkPipelineInputAssemblyStateCreateFlags)0, // VkPipelineInputAssemblyStateCreateFlags flags;
            (m_stage == STAGE_VERTEX) ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST :
                                        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, // VkPrimitiveTopology topology;
            VK_FALSE                                                          // VkBool32 primitiveRestartEnable;
        };

        const VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo = {
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, // VkStructureType sType;
            nullptr,                                                    // const void* pNext;
            (VkPipelineRasterizationStateCreateFlags)0,     // VkPipelineRasterizationStateCreateFlags flags;
            VK_FALSE,                                       // VkBool32 depthClampEnable;
            (m_stage == STAGE_VERTEX) ? VK_TRUE : VK_FALSE, // VkBool32 rasterizerDiscardEnable;
            VK_POLYGON_MODE_FILL,                           // VkPolygonMode polygonMode;
            VK_CULL_MODE_NONE,                              // VkCullModeFlags cullMode;
            VK_FRONT_FACE_CLOCKWISE,                        // VkFrontFace frontFace;
            VK_FALSE,                                       // VkBool32 depthBiasEnable;
            0.0f,                                           // float depthBiasConstantFactor;
            0.0f,                                           // float depthBiasClamp;
            0.0f,                                           // float depthBiasSlopeFactor;
            1.0f                                            // float lineWidth;
        };

        const VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo = {
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, // VkStructureType                          sType
            nullptr,                                                  // const void*                              pNext
            0u,                                                       // VkPipelineMultisampleStateCreateFlags    flags
            VK_SAMPLE_COUNT_1_BIT, // VkSampleCountFlagBits                    rasterizationSamples
            VK_FALSE,              // VkBool32                                 sampleShadingEnable
            1.0f,                  // float                                    minSampleShading
            nullptr,               // const VkSampleMask*                      pSampleMask
            VK_FALSE,              // VkBool32                                 alphaToCoverageEnable
            VK_FALSE               // VkBool32                                 alphaToOneEnable
        };

        VkViewport viewport = makeViewport(DIM * NUM_WORKGROUP_EACH_DIM, DIM * NUM_WORKGROUP_EACH_DIM);
        VkRect2D scissor    = makeRect2D(DIM * NUM_WORKGROUP_EACH_DIM, DIM * NUM_WORKGROUP_EACH_DIM);

        const VkPipelineViewportStateCreateInfo viewportStateCreateInfo = {
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, // VkStructureType                             sType
            nullptr,                                               // const void*                                 pNext
            (VkPipelineViewportStateCreateFlags)0,                 // VkPipelineViewportStateCreateFlags          flags
            1u,        // uint32_t                                    viewportCount
            &viewport, // const VkViewport*                           pViewports
            1u,        // uint32_t                                    scissorCount
            &scissor   // const VkRect2D*                             pScissors
        };

        Move<VkShaderModule> fs;
        Move<VkShaderModule> vs;

        uint32_t numStages;
        if (m_stage == STAGE_VERTEX)
        {
            vs        = createShaderModule(vk, device, m_context.getBinaryCollection().get("test"), 0);
            fs        = createShaderModule(vk, device, m_context.getBinaryCollection().get("test"), 0); // bogus
            numStages = 1u;
        }
        else
        {
            vs        = createShaderModule(vk, device, m_context.getBinaryCollection().get("vert"), 0);
            fs        = createShaderModule(vk, device, m_context.getBinaryCollection().get("test"), 0);
            numStages = 2u;
        }

        const VkPipelineShaderStageCreateInfo shaderCreateInfo[2] = {
            {
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, (VkPipelineShaderStageCreateFlags)0,
                VK_SHADER_STAGE_VERTEX_BIT, // stage
                *vs,                        // shader
                "main",
                &specInfo, // pSpecializationInfo
            },
            {
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, (VkPipelineShaderStageCreateFlags)0,
                VK_SHADER_STAGE_FRAGMENT_BIT, // stage
                *fs,                          // shader
                "main",
                &specInfo, // pSpecializationInfo
            }};

        const VkGraphicsPipelineCreateInfo graphicsPipelineCreateInfo = {
            VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, // VkStructureType sType;
            nullptr,                                         // const void* pNext;
            (VkPipelineCreateFlags)0,                        // VkPipelineCreateFlags flags;
            numStages,                                       // uint32_t stageCount;
            &shaderCreateInfo[0],                            // const VkPipelineShaderStageCreateInfo* pStages;
            &vertexInputStateCreateInfo,   // const VkPipelineVertexInputStateCreateInfo* pVertexInputState;
            &inputAssemblyStateCreateInfo, // const VkPipelineInputAssemblyStateCreateInfo* pInputAssemblyState;
            nullptr,                       // const VkPipelineTessellationStateCreateInfo* pTessellationState;
            &viewportStateCreateInfo,      // const VkPipelineViewportStateCreateInfo* pViewportState;
            &rasterizationStateCreateInfo, // const VkPipelineRasterizationStateCreateInfo* pRasterizationState;
            &multisampleStateCreateInfo,   // const VkPipelineMultisampleStateCreateInfo* pMultisampleState;
            nullptr,                       // const VkPipelineDepthStencilStateCreateInfo* pDepthStencilState;
            nullptr,                       // const VkPipelineColorBlendStateCreateInfo* pColorBlendState;
            nullptr,                       // const VkPipelineDynamicStateCreateInfo* pDynamicState;
            pipelineLayout.get(),          // VkPipelineLayout layout;
            renderPass.get(),              // VkRenderPass renderPass;
            0u,                            // uint32_t subpass;
            VK_NULL_HANDLE,                // VkPipeline basePipelineHandle;
            0                              // int basePipelineIndex;
        };

        pipeline = createGraphicsPipeline(vk, device, VK_NULL_HANDLE, &graphicsPipelineCreateInfo);
    }

    Move<VkCommandPool> cmdPool =
        createCommandPool(vk, device, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, queueData.familyIndex);
    Move<VkCommandBuffer> cmdBuffer = allocateCommandBuffer(vk, device, *cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);

    VkBufferDeviceAddressInfo addrInfo = {
        VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, // VkStructureType sType;
        nullptr,                                      // const void*     pNext;
        VK_NULL_HANDLE,                               // VkBuffer        buffer;
    };

    VkImageSubresourceRange range = makeImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u);
    VkClearValue clearColor       = makeClearValueColorU32(0, 0, 0, 0);

    VkMemoryBarrier memBarrier = {
        VK_STRUCTURE_TYPE_MEMORY_BARRIER, // sType
        nullptr,                          // pNext
        0u,                               // srcAccessMask
        0u,                               // dstAccessMask
    };

    const VkBufferCopy copyParams = {
        (VkDeviceSize)0u,            // srcOffset
        (VkDeviceSize)0u,            // dstOffset
        bufferSizes[idxResultBuffer] // size
    };

    uint32_t NUM_SUBMITS = 4;

    for (uint32_t x = 0; x < NUM_SUBMITS; ++x)
    {
        beginCommandBuffer(vk, *cmdBuffer, 0u);

        if (x == 0)
            vk.cmdFillBuffer(*cmdBuffer, **buffers[idxResultBuffer], 0, bufferSizes[idxResultBuffer], 0);

        for (uint32_t i = 0; i < numBuffers; ++i)
        {
            if (m_buffers[i].sc != SC_IMAGE)
                continue;

            const VkImageMemoryBarrier imageBarrier = {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, // VkStructureType sType
                nullptr,                                // const void*     pNext
                0u,                                     // VkAccessFlags   srcAccessMask
                VK_ACCESS_TRANSFER_WRITE_BIT,           // VkAccessFlags   dstAccessMask
                VK_IMAGE_LAYOUT_UNDEFINED,              // VkImageLayout   oldLayout
                VK_IMAGE_LAYOUT_GENERAL,                // VkImageLayout   newLayout
                VK_QUEUE_FAMILY_IGNORED,                // uint32_t        srcQueueFamilyIndex
                VK_QUEUE_FAMILY_IGNORED,                // uint32_t        dstQueueFamilyIndex
                **images[i],                            // VkImage         image
                {
                    VK_IMAGE_ASPECT_COLOR_BIT, // VkImageAspectFlags aspectMask
                    0u,                        // uint32_t           baseMipLevel
                    1u,                        // uint32_t           mipLevels
                    0u,                        // uint32_t           baseArray
                    1u,                        // uint32_t           arraySize
                }};

            vk.cmdPipelineBarrier(*cmdBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  (VkDependencyFlags)0, 0, nullptr, 0, nullptr, 1, &imageBarrier);
        }

        vk.cmdBindDescriptorSets(*cmdBuffer, bindPoint, *pipelineLayout, 0u, 1, &*descriptorSet, 0u, nullptr);
        vk.cmdBindPipeline(*cmdBuffer, bindPoint, *pipeline);

        for (uint32_t i = 0; i < numBuffers; ++i)
        {
            if (m_buffers[i].sc != SC_PHYSBUFFER)
                continue;

            addrInfo.buffer      = **buffers[i];
            VkDeviceAddress addr = vk.getBufferDeviceAddress(device, &addrInfo);
            vk.cmdPushConstants(*cmdBuffer, *pipelineLayout, allShaderStages, 8 * i, sizeof(VkDeviceSize), &addr);
        }

        for (uint32_t iters = 0; iters < 50; ++iters)
        {
            for (uint32_t i = 0; i < idxResultBuffer; ++i)
            {
                if (buffers[i])
                    vk.cmdFillBuffer(*cmdBuffer, **buffers[i], 0, bufferSizes[i], 0);
                if (images[i])
                    vk.cmdClearColorImage(*cmdBuffer, **images[i], VK_IMAGE_LAYOUT_GENERAL, &clearColor.color, 1,
                                          &range);
            }

            memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vk.cmdPipelineBarrier(*cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, allPipelineStages, 0, 1, &memBarrier, 0,
                                  nullptr, 0, nullptr);

            if (m_stage == STAGE_COMPUTE)
            {
                vk.cmdDispatch(*cmdBuffer, NUM_WORKGROUP_EACH_DIM, NUM_WORKGROUP_EACH_DIM, 1);
            }
            else
            {
                beginRenderPass(vk, *cmdBuffer, *renderPass, *framebuffer,
                                makeRect2D(DIM * NUM_WORKGROUP_EACH_DIM, DIM * NUM_WORKGROUP_EACH_DIM), 0, nullptr,
                                VK_SUBPASS_CONTENTS_INLINE);
                // Draw a point cloud for vertex shader testing, and a single quad for fragment shader testing
                if (m_stage == STAGE_VERTEX)
                {
                    vk.cmdDraw(*cmdBuffer, DIM * DIM * NUM_WORKGROUP_EACH_DIM * NUM_WORKGROUP_EACH_DIM, 1u, 0u, 0u);
                }
                else
                {
                    vk.cmdDraw(*cmdBuffer, 4u, 1u, 0u, 0u);
                }
                endRenderPass(vk, *cmdBuffer);
            }

            memBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            memBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            vk.cmdPipelineBarrier(*cmdBuffer, allPipelineStages, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &memBarrier, 0,
                                  nullptr, 0, nullptr);
        }

        if (x == NUM_SUBMITS - 1)
        {
            vk.cmdCopyBuffer(*cmdBuffer, **buffers[idxResultBuffer], **copyBuffer, 1, &copyParams);
            memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            memBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vk.cmdPipelineBarrier(*cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                                  &memBarrier, 0, nullptr, 0, nullptr);
        }

        endCommandBuffer(vk, *cmdBuffer);

        submitCommandsAndWait(vk, device, queueData.handle, cmdBuffer.get());

        m_context.resetCommandPoolForVKSC(device, *cmdPool);
    }

    tcu::TestLog &log = m_context.getTestContext().getLog();

    uint32_t *ptr = (uint32_t *)copyBuffer->getAllocation().getHostPtr();
    invalidateAlloc(vk, device, copyBuffer->getAllocation());
    qpTestResult res = QP_TEST_RESULT_PASS;

    uint32_t numErrors = 0;
    for (uint32_t i = 0; i < NUM_INVOCATIONS; ++i)
    {
        if (ptr[i] != 0)
        {
            if (numErrors < 256)
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

} // namespace MemoryModel
} // namespace vkt
