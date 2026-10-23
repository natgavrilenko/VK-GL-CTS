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
 * \brief Memory Model Pattern Test Case
 *//*--------------------------------------------------------------------*/

#include "vktMemoryModelPatternTestCase.hpp"

#include "tcuDefs.hpp"
#include "tcuTestCase.hpp"
#include "vkQueryUtil.hpp"

namespace vkt
{
namespace MemoryModel
{
using namespace vk;

void MemoryModelTestCase::checkSupportBuffers(const Context &context) const
{
    bool hasInt64      = false;
    bool hasPhysBuffer = false;

    for (const auto &buffer : m_buffers)
    {
        if (buffer.type == DATA_TYPE_FLOAT64 && buffer.sc == SC_IMAGE)
        {
            TCU_THROW(NotSupportedError,
                      "VkShaderAtomicFloat64: 64-bit floating point image atomic operations not supported");
        }
        hasInt64 |= (buffer.type == DATA_TYPE_UINT64);
        hasPhysBuffer |= (buffer.sc == SC_PHYSBUFFER);
    }

    if (hasPhysBuffer && !context.isBufferDeviceAddressSupported())
        TCU_THROW(NotSupportedError, "Physical storage buffer pointers not supported");

    if (hasInt64 && !context.getDeviceFeatures().shaderInt64)
        TCU_THROW(NotSupportedError, "64-bit integer in shaders not supported");

    const auto &vki       = context.getInstanceInterface();
    const auto physDevice = context.getPhysicalDevice();
    const bool isDevAddrSupported =
        hasPhysBuffer && context.isDeviceFunctionalitySupported("VK_KHR_buffer_device_address");

    const VkPhysicalDeviceMemoryProperties properties = getPhysicalDeviceMemoryProperties(vki, physDevice);

    for (const auto &buffer : m_buffers)
    {
        if (buffer.sc != SC_BUFFER && buffer.sc != SC_PHYSBUFFER)
            continue;

        const bool isDeviceAddr        = isDevAddrSupported && (buffer.sc == SC_PHYSBUFFER);
        const MemoryRequirement memReq = (isDeviceAddr ? MemoryRequirement::DeviceAddress : MemoryRequirement::Any) |
                                         (buffer.local ? MemoryRequirement::Local : MemoryRequirement::NonLocal);
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

void MemoryModelTestCase::checkSupportStageAtomics(const Context &context, const Stage stage) const
{
    if (stage == STAGE_VERTEX && !context.getDeviceFeatures().vertexPipelineStoresAndAtomics)
        TCU_THROW(NotSupportedError, "vertexPipelineStoresAndAtomics not supported");
    if (stage == STAGE_FRAGMENT && !context.getDeviceFeatures().fragmentStoresAndAtomics)
        TCU_THROW(NotSupportedError, "fragmentStoresAndAtomics not supported");
}

void MemoryModelTestCase::checkSupportStageSubgroupFeatures(const Context &context, const Stage stage) const
{
    VkPhysicalDeviceSubgroupProperties subgroupProperties;
    subgroupProperties.sType               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    subgroupProperties.pNext               = nullptr;
    subgroupProperties.supportedOperations = 0;

    VkPhysicalDeviceProperties2 properties;
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &subgroupProperties;

    context.getInstanceInterface().getPhysicalDeviceProperties2(context.getPhysicalDevice(), &properties);

    if (!(subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) ||
        !(subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) ||
        !(subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT))
        TCU_THROW(NotSupportedError, "Subgroup features not supported");

    VkShaderStageFlags flags = 0;
    if (stage == STAGE_COMPUTE)
        flags = VK_SHADER_STAGE_COMPUTE_BIT;
    else if (stage == STAGE_VERTEX)
        flags = VK_SHADER_STAGE_VERTEX_BIT;
    else if (stage == STAGE_FRAGMENT)
        flags = VK_SHADER_STAGE_FRAGMENT_BIT;

    if ((subgroupProperties.supportedStages & flags) == 0)
        TCU_THROW(NotSupportedError, "Device does not support subgroup operations for this stage");
}

void MemoryModelTestCase::checkSupportStageSplitBarriers(const Context &context, const Stage stage) const
{
    if (stage != STAGE_COMPUTE)
        TCU_THROW(NotSupportedError, "VK_EXT_shader_split_barrier is only supported in compute shaders");

#ifndef CTS_USES_VULKANSC
    if (!context.getShaderSplitBarrierFeaturesEXT().shaderSplitBarrier)
        TCU_THROW(NotSupportedError, "shaderSplitBarrier feature not supported");
#else
    DE_UNREF(context);
    TCU_THROW(NotSupportedError, "shaderSplitBarrier feature not supported");
#endif
}

const BufferDef &MemoryModelTestCase::getBuffer(const char *name) const
{
    const auto it = m_bufferIndices.find(name);
    DE_ASSERT(it != m_bufferIndices.end());
    return m_buffers[it->second];
}

uint32_t MemoryModelTestCase::getBufferSemantics(const char *name) const
{
    switch (getBuffer(name).sc)
    {
    case SC_PHYSBUFFER:
    case SC_BUFFER:
        return SEM_BUFFER;
    case SC_IMAGE:
        return SEM_IMAGE;
    case SC_WORKGROUP:
        return SEM_SHARED;
    default:
        DE_ASSERT(0);
        return 0;
    }
}

// Compute coordinates based on the storage class and scope.
// For workgroup scope, we pair up LocalInvocationID and DIM-1-LocalInvocationID.
// For device scope, we pair up GlobalInvocationID and DIM*NUMWORKGROUPS-1-GlobalInvocationID.
// For subgroup scope, we pair up LocalInvocationID and LocalInvocationID from subgroupId^(subgroupSize-1)
StreamOp MemoryModelGlTestCase::gl_layoutCoordinates(const Stage stage, const Scope mappingScope)
{
    return [=](std::ostream &os) {
        if (stage == STAGE_COMPUTE)
        {
            switch (mappingScope)
            {
                case SCOPE_SUBGROUP:
                    os << "   ivec2 localId           = ivec2(gl_LocalInvocationID.xy);\n"
                          "   ivec2 partnerLocalId    = subgroupShuffleXor(localId, gl_SubgroupSize-1);\n"
                          "   uint sharedCoord        = localId.y * DIM + localId.x;\n"
                          "   uint partnerSharedCoord = partnerLocalId.y * DIM + partnerLocalId.x;\n"
                          "   uint bufferCoord        = (gl_WorkGroupID.y * NUM_WORKGROUP_EACH_DIM + gl_WorkGroupID.x)*DIM*DIM + sharedCoord;\n"
                          "   uint partnerBufferCoord = (gl_WorkGroupID.y * NUM_WORKGROUP_EACH_DIM + gl_WorkGroupID.x)*DIM*DIM + partnerSharedCoord;\n"
                          "   ivec2 imageCoord        = ivec2(gl_WorkGroupID.xy * gl_WorkGroupSize.xy + localId);\n"
                          "   ivec2 partnerImageCoord = ivec2(gl_WorkGroupID.xy * gl_WorkGroupSize.xy + partnerLocalId);\n";
                    break;
                case SCOPE_WORKGROUP:
                    os << "   ivec2 localId           = ivec2(gl_LocalInvocationID.xy);\n"
                          "   ivec2 partnerLocalId    = ivec2(DIM-1)-ivec2(gl_LocalInvocationID.xy);\n"
                          "   uint sharedCoord        = localId.y * DIM + localId.x;\n"
                          "   uint partnerSharedCoord = partnerLocalId.y * DIM + partnerLocalId.x;\n"
                          "   uint bufferCoord        = (gl_WorkGroupID.y * NUM_WORKGROUP_EACH_DIM + gl_WorkGroupID.x)*DIM*DIM + sharedCoord;\n"
                          "   uint partnerBufferCoord = (gl_WorkGroupID.y * NUM_WORKGROUP_EACH_DIM + gl_WorkGroupID.x)*DIM*DIM + partnerSharedCoord;\n"
                          "   ivec2 imageCoord        = ivec2(gl_WorkGroupID.xy * gl_WorkGroupSize.xy + localId);\n"
                          "   ivec2 partnerImageCoord = ivec2(gl_WorkGroupID.xy * gl_WorkGroupSize.xy + partnerLocalId);\n";
                    break;
                case SCOPE_QUEUEFAMILY:
                case SCOPE_DEVICE:
                    os << "   ivec2 globalId          = ivec2(gl_GlobalInvocationID.xy);\n"
                          "   ivec2 partnerGlobalId   = ivec2(DIM*NUM_WORKGROUP_EACH_DIM-1) - ivec2(gl_GlobalInvocationID.xy);\n"
                          "   uint bufferCoord        = globalId.y * DIM*NUM_WORKGROUP_EACH_DIM + globalId.x;\n"
                          "   uint partnerBufferCoord = partnerGlobalId.y * DIM*NUM_WORKGROUP_EACH_DIM + partnerGlobalId.x;\n"
                          "   ivec2 imageCoord        = globalId;\n"
                          "   ivec2 partnerImageCoord = partnerGlobalId;\n";
                    break;
                default:
                    DE_ASSERT(0);
            }
        }
        else if (stage == STAGE_VERTEX)
        {
            switch (mappingScope)
            {
                case SCOPE_SUBGROUP:
                    os << "   uint bufferCoord        = gl_VertexIndex;\n"
                          "   uint partnerBufferCoord = subgroupShuffleXor(gl_VertexIndex, gl_SubgroupSize-1);\n"
                          "   ivec2 imageCoord        = ivec2(gl_VertexIndex % (DIM*NUM_WORKGROUP_EACH_DIM), gl_VertexIndex / (DIM*NUM_WORKGROUP_EACH_DIM));\n"
                          "   ivec2 partnerImageCoord = subgroupShuffleXor(imageCoord, gl_SubgroupSize-1);\n";
                    break;
                case SCOPE_QUEUEFAMILY:
                case SCOPE_DEVICE:
                    os << "   ivec2 globalId          = ivec2(gl_VertexIndex % (DIM*NUM_WORKGROUP_EACH_DIM), gl_VertexIndex / (DIM*NUM_WORKGROUP_EACH_DIM));\n"
                          "   ivec2 partnerGlobalId   = ivec2(DIM*NUM_WORKGROUP_EACH_DIM-1) - globalId;\n"
                          "   uint bufferCoord        = globalId.y * DIM*NUM_WORKGROUP_EACH_DIM + globalId.x;\n"
                          "   uint partnerBufferCoord = partnerGlobalId.y * DIM*NUM_WORKGROUP_EACH_DIM + partnerGlobalId.x;\n"
                          "   ivec2 imageCoord        = globalId;\n"
                          "   ivec2 partnerImageCoord = partnerGlobalId;\n";
                    break;
                default:
                    DE_ASSERT(0);
            }
            os << "   gl_PointSize            = 1.0f;\n"
                  "   gl_Position             = vec4(0.0f, 0.0f, 0.0f, 1.0f);\n\n";
        }
        else if (stage == STAGE_FRAGMENT)
        {
            switch (mappingScope)
            {
                case SCOPE_SUBGROUP:
                    os << "   ivec2 localId        = ivec2(gl_FragCoord.xy) % ivec2(DIM);\n"
                          "   ivec2 groupId        = ivec2(gl_FragCoord.xy) / ivec2(DIM);\n"
                          "   ivec2 partnerLocalId = subgroupShuffleXor(localId, gl_SubgroupSize-1);\n"
                          "   ivec2 partnerGroupId = subgroupShuffleXor(groupId, gl_SubgroupSize-1);\n"
                          "   uint sharedCoord     = localId.y * DIM + localId.x;\n"
                          "   uint partnerSharedCoord = partnerLocalId.y * DIM + partnerLocalId.x;\n"
                          "   uint bufferCoord     = (groupId.y * NUM_WORKGROUP_EACH_DIM + groupId.x)*DIM*DIM + sharedCoord;\n"
                          "   uint partnerBufferCoord = (partnerGroupId.y * NUM_WORKGROUP_EACH_DIM + partnerGroupId.x)*DIM*DIM + partnerSharedCoord;\n"
                          "   ivec2 imageCoord     = ivec2(groupId.xy * ivec2(DIM) + localId);\n"
                          "   ivec2 partnerImageCoord = ivec2(partnerGroupId.xy * ivec2(DIM) + partnerLocalId);\n";
                    break;
                case SCOPE_QUEUEFAMILY:
                case SCOPE_DEVICE:
                    os << "   ivec2 localId       = ivec2(gl_FragCoord.xy) % ivec2(DIM);\n"
                          "   ivec2 groupId       = ivec2(gl_FragCoord.xy) / ivec2(DIM);\n"
                          "   ivec2 partnerLocalId = ivec2(DIM-1)-localId;\n"
                          "   ivec2 partnerGroupId = groupId;\n"
                          "   uint sharedCoord    = localId.y * DIM + localId.x;\n"
                          "   uint partnerSharedCoord = partnerLocalId.y * DIM + partnerLocalId.x;\n"
                          "   uint bufferCoord    = (groupId.y * NUM_WORKGROUP_EACH_DIM + groupId.x)*DIM*DIM + sharedCoord;\n"
                          "   uint partnerBufferCoord = (partnerGroupId.y * NUM_WORKGROUP_EACH_DIM + partnerGroupId.x)*DIM*DIM + partnerSharedCoord;\n"
                          "   ivec2 imageCoord    = ivec2(groupId.xy * ivec2(DIM) + localId);\n"
                          "   ivec2 partnerImageCoord = ivec2(partnerGroupId.xy * ivec2(DIM) + partnerLocalId);\n";
                    break;
                default:
                    DE_ASSERT(0);
            }
        }
    };
}

const char *MemoryModelGlTestCase::gl_barrier()
{
    return "barrier()";
}

StreamOp MemoryModelGlTestCase::gl_controlBarrier(Scope scope1, Scope scope2, uint32_t sem)
{
    return [=](std::ostream &os) {
        os << "controlBarrier(" << gl_scope(scope1) << ", " << gl_scope(scope2) << ", "
           << gl_semantics1(sem) << ", " << gl_semantics2(sem) << ")";
    };
}

StreamOp MemoryModelGlTestCase::gl_controlBarrierArrive(const Scope scope1, const Scope scope2, const uint32_t sem)
{
    return [=](std::ostream &os) {
        os << "controlBarrierArrive(" << gl_scope(scope1) << ", " << gl_scope(scope2) << ", "
           << gl_semantics1(sem) << ", " << gl_semantics2(sem) << ")";
    };
}

StreamOp MemoryModelGlTestCase::gl_controlBarrierWait(const Scope scope1, const Scope scope2, const uint32_t sem)
{
    return [=](std::ostream &os) {
        os << "controlBarrierWait(" << gl_scope(scope1) << ", " << gl_scope(scope2) << ", "
           << gl_semantics1(sem) << ", " << gl_semantics2(sem) << ")";
    };
}

StreamOp MemoryModelGlTestCase::gl_memoryBarrier(const Scope scope, const uint32_t sem)
{
    return [=](std::ostream &os) {
        os << "memoryBarrier(" << gl_scope(scope) << ", "
           << gl_semantics1(sem) << ", " << gl_semantics2(sem) << ")";
    };
}

StreamOp MemoryModelGlTestCase::gl_atomicLoad(const char *name, const char* idx, const Scope scope, const uint32_t sem) const
{
    auto buffer = getBuffer(name);
    return [=](std::ostream &os) {
        if (buffer.sc == SC_IMAGE)
            os << "imageAtomicLoad(" << buffer.name << ", " << idx << ", ";
        else
            os << "atomicLoad(" << buffer.name << ".x[" << idx << "], ";
        os << gl_scope(scope) << ", " << gl_semantics1(sem) << ", " << gl_semantics2(sem) << ")";
    };
}

StreamOp MemoryModelGlTestCase::gl_atomicStore(const char *name, const char* idx, const char* value, const Scope scope, const uint32_t sem) const
{
    auto buffer = getBuffer(name);
    return [=](std::ostream &os) {
        const bool noCast   = (buffer.sc == SC_IMAGE) &&
                              ((buffer.type == DATA_TYPE_UINT64) || (buffer.type == DATA_TYPE_FLOAT64));
        const char *typeStr = noCast ? "" : gl_type(buffer.type);

        if (buffer.sc == SC_IMAGE)
            os << "imageAtomicStore(" << buffer.name << ", " << idx << ", ";
        else
            os << "atomicStore(" << buffer.name << ".x[" << idx << "], ";
        os << typeStr << "(" << value << "), " << gl_scope(scope) << ", "
           << gl_semantics1(sem) << ", " << gl_semantics2(sem) << ")";
    };
}

StreamOp MemoryModelGlTestCase::gl_atomicExchange(const char *name, const char* idx, const char* value, const Scope scope, const uint32_t sem) const
{
    auto buffer = getBuffer(name);
    return [=](std::ostream &os) {
        const bool noCast   = (buffer.sc == SC_IMAGE) &&
                              ((buffer.type == DATA_TYPE_UINT64) || (buffer.type == DATA_TYPE_FLOAT64));
        const char *typeStr = noCast ? "" : gl_type(buffer.type);

        if (buffer.sc == SC_IMAGE)
            os << "imageAtomicExchange(" << buffer.name << ", " << idx << ", ";
        else
            os << "atomicExchange(" << buffer.name << ".x[" << idx << "], ";
        os << typeStr << "(" << value << "), " << gl_scope(scope) << ", "
           << gl_semantics1(sem) << ", " << gl_semantics2(sem) << ")";
    };
}

StreamOp MemoryModelGlTestCase::gl_load(const char *name, const char* idx) const
{
    auto buffer = getBuffer(name);
    return [=](std::ostream &os) {
        if (buffer.sc == SC_IMAGE)
            os << "imageLoad(" << buffer.name << ", " << idx << ").x";
        else
            os << buffer.name << ".x[" << idx << "]";
    };
}

StreamOp MemoryModelGlTestCase::gl_store(const char *name, const char* idx, const char* value) const
{
    auto buffer = getBuffer(name);
    return [=](std::ostream &os) {
        if (buffer.sc == SC_IMAGE)
        {
            if (buffer.type == DATA_TYPE_UINT || buffer.type == DATA_TYPE_UINT64)
                os << "imageStore(" << buffer.name << ", " << idx << ", uvec4(" << value << ", 0, 0, 0))";
            else
                os << "imageStore(" << buffer.name << ", " << idx << ", vec4(" << gl_type(buffer.type) << "(" << value << "), 0, 0, 0))";
        }
        else
            os << buffer.name << ".x[" << idx << "] = " << gl_type(buffer.type) << "(" << value << ")";
    };
}

StreamOp MemoryModelGlTestCase::gl_layout() const
{
    return [=](std::ostream &os) {
        bool hasPhysBuffer = false;
        for (size_t i = 0; i < m_buffers.size(); i++)
        {
            auto buffer = m_buffers[i];
            bool intType = buffer.type == DATA_TYPE_UINT || buffer.type == DATA_TYPE_UINT64;
            switch (buffer.sc)
            {
            case SC_PHYSBUFFER:
                os << "layout(buffer_reference) buffer T" << buffer.name << "Ref { " << gl_type(buffer.type) << " x[]; };\n";
                hasPhysBuffer = true;
                break;
            case SC_BUFFER:
                os << "layout(set=0, binding=" << i << ") " << gl_memqual(buffer.memqual) << " buffer T" << buffer.name << " { " << gl_type(buffer.type) << " x[]; } " << buffer.name << ";\n";
                break;
            case SC_IMAGE:
                if (intType)
                    os << "layout(set=0, binding=" << i << ", r32ui) uniform " << gl_memqual(buffer.memqual) << " uimage2D " << buffer.name << ";\n";
                else
                    os << "layout(set=0, binding=" << i << ", r32f) uniform " << gl_memqual(buffer.memqual) << " image2D " << buffer.name << ";\n";
                break;
            case SC_WORKGROUP:
                os << "shared S " << buffer.name << ";\n";
                break;
            default:
                DE_ASSERT(0);
            }
        }
        if (hasPhysBuffer)
        {
            os << "layout (push_constant, std430) uniform PC {\n";
            for (size_t i = 0; i < m_buffers.size(); i++)
            {
                if (m_buffers[i].sc == SC_PHYSBUFFER)
                {
                    const char *suffix = m_buffers[i].memqual == MEM_NONE ? "" : "ref";
                    os << "layout(offset = " << (i * 8) << ") T" << m_buffers[i].name << "Ref " << m_buffers[i].name << suffix << ";\n";
                }
            }
            os << "};\n";
        }
    };
}

StreamOp MemoryModelGlTestCase::gl_layoutReferences() const
{
    return [=](std::ostream &os) {
        for (const auto &buffer : m_buffers)
        {
            if (buffer.sc == SC_PHYSBUFFER && buffer.memqual != MEM_NONE)
            {
                os << "   " << gl_memqual(buffer.memqual) << " T" << buffer.name << "Ref " << buffer.name << " = " << buffer.name << "ref;\n";
            }
        }
    };
}

StreamOp MemoryModelGlTestCase::gl_layoutBufferCoordinates() const
{
    return [=](std::ostream &os) {
        for (const auto &buffer : m_buffers)
        {
            switch (buffer.sc)
            {
                case SC_PHYSBUFFER:
                case SC_BUFFER:
                    os << "   uint " << buffer.name << "Coord = bufferCoord;\n"
                          "   uint " << buffer.name << "PartnerCoord = partnerBufferCoord;\n";
                    break;
                case SC_IMAGE:
                    os << "   ivec2 " << buffer.name << "Coord = imageCoord;\n"
                          "   ivec2 " << buffer.name << "PartnerCoord = partnerImageCoord;\n";
                    break;
                case SC_WORKGROUP:
                    os << "   uint " << buffer.name << "Coord = sharedCoord;\n"
                          "   uint " << buffer.name << "PartnerCoord = partnerSharedCoord;\n";
                    break;
                default:
                    DE_ASSERT(0);
            }
        }
    };
}

StreamOp MemoryModelGlTestCase::gl_layoutBufferCoordinates00() const
{
    return [=](std::ostream &os) {
        for (const auto &buffer : m_buffers)
        {
            switch (buffer.sc)
            {
                case SC_PHYSBUFFER:
                case SC_BUFFER:
                    os << "   uint " << buffer.name << "Coord00 = bufferCoord00;\n"
                          "   uint " << buffer.name << "PartnerCoord00 = partnerBufferCoord00;\n";
                    break;
                case SC_IMAGE:
                    os << "   ivec2 " << buffer.name << "Coord00 = imageCoord00;\n"
                          "   ivec2 " << buffer.name << "PartnerCoord00 = partnerImageCoord00;\n";
                    break;
                default:
                    DE_ASSERT(0);
            }
        }
    };
}

StreamOp MemoryModelGlTestCase::gl_layoutInitSharedMemory(const Scope scope) const
{
    return [=](std::ostream &os) {
        bool needBarrier = false;
        for (const auto &buffer : m_buffers)
        {
            if (buffer.sc == SC_WORKGROUP)
            {
                os << "   " << buffer.name <<".x[sharedCoord] = 0;\n";
                needBarrier = true;
            }
        }
        if (needBarrier)
        {
            switch (scope)
            {
            case SCOPE_SUBGROUP:
                os << "   subgroupBarrier();\n";
                break;
            case SCOPE_WORKGROUP:
                os << "   barrier();\n";
                break;
            default:
                DE_ASSERT(0);
            }
        }
    };
}

const char *MemoryModelGlTestCase::gl_type(const char *name) const
{
    return gl_type(getBuffer(name).type);
}

const char *MemoryModelGlTestCase::gl_type(const DataType type) {
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

const char *MemoryModelGlTestCase::gl_memqual(const GlMemQualifier qualifier) {
    switch (qualifier)
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

const char *MemoryModelGlTestCase::gl_scope(const Scope scope) {
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

template <std::size_t N>
StreamOp MemoryModelGlTestCase::formatSemantics(
    uint32_t value, const std::pair<uint32_t, const char *> (&mappings)[N])
{
    return [=, &mappings](std::ostream &os) {
        bool first = true;
        if (value > 0)
        {
            for (const auto &mapping : mappings)
            {
                if (value & mapping.first)
                {
                    if (!first)
                        os << " | ";
                    os << mapping.second;
                    first = false;
                }
            }
        }
        if (first)
            os << "0";
    };
}

StreamOp MemoryModelGlTestCase::gl_semantics1(uint32_t value)
{
    static constexpr std::pair<uint32_t, const char *> kMappings[] = {
        { SEM_BUFFER, "gl_StorageSemanticsBuffer" },
        { SEM_IMAGE,  "gl_StorageSemanticsImage"  },
        { SEM_SHARED, "gl_StorageSemanticsShared" },
    };
    return formatSemantics(value & GL_SEM1, kMappings);
}

StreamOp MemoryModelGlTestCase::gl_semantics2(uint32_t value)
{
    static constexpr std::pair<uint32_t, const char *> kMappings[] = {
        { SEM_ACQUIRE,         "gl_SemanticsAcquire"        },
        { SEM_RELEASE,         "gl_SemanticsRelease"        },
        { SEM_ACQUIRE_RELEASE, "gl_SemanticsAcquireRelease" },
        { SEM_AVAILABLE,       "gl_SemanticsMakeAvailable"  },
        { SEM_VISIBLE,         "gl_SemanticsMakeVisible"    },
    };
    return formatSemantics(value & GL_SEM2, kMappings);
}

} // namespace MemoryModel
} // namespace vkt
