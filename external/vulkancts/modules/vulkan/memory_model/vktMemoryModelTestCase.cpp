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

constexpr uint32_t SPV_OP_LOAD                 = 61;
constexpr uint32_t SPV_OP_STORE                = 62;
constexpr uint32_t SPV_CAPABILITY_DEVICE_SCOPE = 5346;
constexpr uint32_t SPV_ALIGNED                 = 0x2;
constexpr uint32_t SPV_MAKE_POINTER_AVAILABLE  = 0x8;
constexpr uint32_t SPV_MAKE_POINTER_VISIBLE    = 0x10;
constexpr uint32_t SPV_NON_PRIVATE_POINTER     = 0x20;

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

const char *gl_imageFormat(const DataType type)
{
    switch (type)
    {
    case DATA_TYPE_UINT:
        return "r32ui";
    case DATA_TYPE_UINT64:
        return "r64ui";
    case DATA_TYPE_FLOAT32:
        return "r32f";
    default:
        DE_ASSERT(0);
        return "";
    }
}

const char *gl_imageType(const DataType type)
{
    switch (type)
    {
    case DATA_TYPE_UINT:
        return "uimage2D";
    case DATA_TYPE_UINT64:
        return "u64image2D";
    case DATA_TYPE_FLOAT32:
        return "image2D";
    default:
        DE_ASSERT(0);
        return "";
    }
}

glu::ShaderType getShaderType(const Stage stage)
{
    switch (stage)
    {
    case STAGE_COMPUTE:
        return glu::SHADERTYPE_COMPUTE;
    case STAGE_VERTEX:
        return glu::SHADERTYPE_VERTEX;
    case STAGE_FRAGMENT:
        return glu::SHADERTYPE_FRAGMENT;
    default:
        DE_ASSERT(0);
        return glu::SHADERTYPE_LAST;
    }
}

VkShaderStageFlagBits getShaderStageFlag(const Stage stage)
{
    switch (stage)
    {
    case STAGE_COMPUTE:
        return VK_SHADER_STAGE_COMPUTE_BIT;
    case STAGE_VERTEX:
        return VK_SHADER_STAGE_VERTEX_BIT;
    case STAGE_FRAGMENT:
        return VK_SHADER_STAGE_FRAGMENT_BIT;
    default:
        DE_ASSERT(0);
        return VK_SHADER_STAGE_ALL;
    }
}

VkPipelineStageFlagBits getPipelineStageFlag(const Stage stage)
{
    switch (stage)
    {
    case STAGE_COMPUTE:
        return VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    case STAGE_VERTEX:
        return VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
    case STAGE_FRAGMENT:
        return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    default:
        DE_ASSERT(0);
        return VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }
}

VkFormat getImageFormat(const DataType type)
{
    switch (type)
    {
    case DATA_TYPE_UINT:
        return VK_FORMAT_R32_UINT;
    case DATA_TYPE_UINT64:
        return VK_FORMAT_R64_UINT;
    case DATA_TYPE_FLOAT32:
        return VK_FORMAT_R32_SFLOAT;
    default:
        DE_ASSERT(0);
        return VK_FORMAT_UNDEFINED;
    }
}

MemoryRequirement getVarMemoryRequirement(const VarDef &var, bool isDeviceAddressSupported)
{
    const bool deviceAddress = isDeviceAddressSupported && var.sc == SC_PHYSBUFFER;
    return (deviceAddress ? MemoryRequirement::DeviceAddress : MemoryRequirement::Any) |
           (var.local ? MemoryRequirement::Local : MemoryRequirement::NonLocal);
}

} // namespace

void MemoryModelTestCase::initPrograms(SourceCollections &programCollection) const
{
    std::ostringstream css;

    css << "#version 450 core\n";
    if (!m_params.core11)
        css << "#pragma use_vulkan_memory_model\n";
    css << "#extension GL_EXT_shader_atomic_float : enable\n"
           "#extension GL_EXT_split_barrier : enable\n"
           "#extension GL_KHR_shader_subgroup_basic : enable\n"
           "#extension GL_KHR_shader_subgroup_shuffle : enable\n"
           "#extension GL_KHR_shader_subgroup_ballot : enable\n"
           "#extension GL_KHR_memory_scope_semantics : enable\n"
           "#extension GL_ARB_gpu_shader_int64 : enable\n"
           "#extension GL_EXT_buffer_reference : enable\n"
           "#extension GL_EXT_shader_image_int64 : enable\n";

    // Can be overridden by a child class, empty by default
    css << declareSpvIntrinsics();

    // A workgroup has DIM x DIM invocations, the grid has gridDim x gridDim invocations
    css << "// DIM/NUM_WORKGROUP_EACH_DIM overridden by spec constants\n"
           "layout(constant_id = 0) const int DIM = 1;\n"
           "layout(constant_id = 1) const int NUM_WORKGROUP_EACH_DIM = 1;\n"
           "const uint gridDim = uint(DIM * NUM_WORKGROUP_EACH_DIM);\n";

    // Can be overridden by a child class, empty by default
    css << declareSharedVars();

    if (m_params.stage == STAGE_COMPUTE)
        css << "layout(local_size_x_id = 0, local_size_y_id = 0, local_size_z = 1) in;\n";

    css << declareVars();

    // Declare functions to compute invocation ID and coordinates
    // Increments invocationIds by one to avoid collisions with initial zero
    css << "uint getInvocationId(ivec2 pos) { return uint(pos.y * gridDim + pos.x) + 1u; }\n"
        << "uint getSharedCoord(ivec2 pos) { return uint((pos.y % DIM) * DIM + pos.x % DIM); }\n"
        << "uint getBufferCoord(ivec2 pos) { return uint((pos.y / DIM * NUM_WORKGROUP_EACH_DIM + pos.x / DIM) * "
           "DIM * DIM) + getSharedCoord(pos); }\n";

    css << "void main()\n"
           "{\n";

    // Stage-specific setup, and the invocation's 2D position (global ID)
    switch (m_params.stage)
    {
    case STAGE_COMPUTE:
        css << "   ivec2 pos = ivec2(gl_GlobalInvocationID.xy);\n";
        break;
    case STAGE_VERTEX:
        css << "   gl_PointSize = 1.0f;\n"
               "   gl_Position = vec4(0.0f, 0.0f, 0.0f, 1.0f);\n"
               "   ivec2 pos = ivec2(gl_VertexIndex % gridDim, gl_VertexIndex / gridDim);\n";
        break;
    case STAGE_FRAGMENT:
        // Kill helper invocations so they don't load outside the bounds of the SSBO.
        // Helper pixels are also initially "active" and if a thread gets one as its
        // partner in SCOPE_SUBGROUP mode, it can't run the test.
        css << "   if (gl_HelperInvocation) { return; }\n"
            << "   ivec2 pos = ivec2(gl_FragCoord.xy);\n";
        break;
    default:
        DE_ASSERT(0);
    }

    css << "   uint invocationId = getInvocationId(pos);\n";

    for (const auto &var : m_vars)
    {
        if (var.sc == SC_PHYSBUFFER && var.qual != MEM_NONE)
        {
            css << "   " << gl_qualifier(var.qual) << " T" << var.name << "Ref " << var.name << " = " << var.name
                << "ref;\n";
        }
    }

    css << initVarCoordinates("pos", "Coord");

    // Initialize shared memory
    for (const auto &var : m_vars)
    {
        if (var.sc == SC_WORKGROUP)
            css << "   " << var.name << ".x[" << var.name << "Coord] = 0;\n";
    }

    if (hasStorageClass(SC_WORKGROUP))
        css << "   barrier();\n";

    // Test logic, implemented by a child class
    css << initProgram();

    css << "}\n";

    // Build shaders
    const ShaderBuildOptions buildOptions(programCollection.usedVulkanVersion, SPIRV_VERSION_1_3, 0u);
    programCollection.glslSources.add("test")
        << glu::ShaderSource(getShaderType(m_params.stage), css.str()) << buildOptions;

    if (m_params.stage == STAGE_FRAGMENT)
    {
        // Draw a fullscreen triangle strip based on gl_VertexIndex
        static const char *vertShader = "#version 450 core\n"
                                        "vec2 coords[4] = {ivec2(-1,-1), ivec2(-1, 1), ivec2(1, -1), ivec2(1, 1)};\n"
                                        "void main() { gl_Position = vec4(coords[gl_VertexIndex], 0, 1); }\n";
        programCollection.glslSources.add("vert") << glu::VertexSource(vertShader);
    }
}

std::string MemoryModelTestCase::declareVars() const
{
    std::ostringstream css;
    std::ostringstream pcs;

    for (size_t i = 0; i < m_vars.size(); i++)
    {
        const auto &var  = m_vars[i];
        const char *name = var.name;

        switch (var.sc)
        {
        case SC_PHYSBUFFER:
        {
            const char *ref = var.qual == MEM_NONE ? "" : "ref";
            css << "layout(buffer_reference) buffer T" << name << "Ref { " << gl_type(var.type) << " x[]; };\n";
            pcs << "layout(offset = " << (i * DEVICE_ADDRESS_SIZE) << ") T" << name << "Ref " << name << ref << ";\n";
            break;
        }
        case SC_BUFFER:
        {
            css << "layout(set=0, binding=" << i << ") " << gl_qualifier(var.qual) << " buffer T" << name << " { "
                << gl_type(var.type) << " x[]; } " << name << ";\n";
            break;
        }
        case SC_IMAGE:
        {
            css << "layout(set=0, binding=" << i << ", " << gl_imageFormat(var.type) << ") uniform "
                << gl_qualifier(var.qual) << " " << gl_imageType(var.type) << " " << name << ";\n";
            break;
        }
        case SC_WORKGROUP:
        {
            css << "struct S" << name << " { " << gl_type(var.type) << " x[DIM*DIM]; };\n"
                << "shared S" << name << " " << name << ";\n";
            break;
        }
        default:
            DE_ASSERT(0);
        }
    }

    if (hasStorageClass(SC_PHYSBUFFER))
        css << "layout (push_constant, std430) uniform PC {\n" << pcs.str() << "};\n";

    return css.str();
}

std::string MemoryModelTestCase::declareVarSpvIntrinsics(const char *name, const bool hasDeviceScope) const
{
    DE_ASSERT(!m_params.core11);

    const VarDef &var = getVar(name);
    DE_ASSERT(var.sc != SC_IMAGE);

    const char *typeStr = gl_type(var.type);

    const std::string capabilities =
        hasDeviceScope ? "capabilities = [" + std::to_string(SPV_CAPABILITY_DEVICE_SCOPE) + "], " : "";

    std::ostringstream css;

    css << "spirv_instruction(" << capabilities << "id = " << SPV_OP_LOAD << ")\n"
        << typeStr << " spv_" << name << "Load(spirv_by_reference " << typeStr << " p, "
        << "spirv_literal int memoryAccess, spirv_literal int alignment);\n";

    css << "spirv_instruction(" << capabilities << "id = " << SPV_OP_LOAD << ")\n"
        << typeStr << " spv_" << name << "LoadVis(spirv_by_reference " << typeStr << " p, "
        << "spirv_literal int memoryAccess, spirv_literal int alignment, int scope);\n";

    css << "spirv_instruction(" << capabilities << "id = " << SPV_OP_STORE << ")\n"
        << "void spv_" << name << "Store(spirv_by_reference " << typeStr << " p, " << typeStr << " value, "
        << "spirv_literal int memoryAccess, spirv_literal int alignment);\n";

    css << "spirv_instruction(" << capabilities << "id = " << SPV_OP_STORE << ")\n"
        << "void spv_" << name << "StoreAv(spirv_by_reference " << typeStr << " p, " << typeStr << " value, "
        << "spirv_literal int memoryAccess, spirv_literal int alignment, int scope);\n";

    return css.str();
}

std::string MemoryModelTestCase::initPartnerCoordinates(const Scope scope, const char *suffix) const
{
    return initPartners(scope, 1, false, suffix);
}

std::string MemoryModelTestCase::initMultiPartnerCoordinates(const Scope scope, const uint32_t numPartners) const
{
    return initPartners(scope, numPartners, true, "");
}

std::string MemoryModelTestCase::initPartners(const Scope scope, const uint32_t numPartners, const bool isIndexRotation,
                                              const char *suffix) const
{
    DE_ASSERT(numPartners >= 1 && (isIndexRotation || numPartners == 1));

    // Helpers to concatenate variable names with a suffix
    const auto name        = [suffix](const char *base) { return std::string{base} + suffix; };
    const auto partnerName = [&](const char *base, const uint32_t i)
    { return name(base) + (numPartners > 1 ? std::to_string(i) : std::string{}); };

    // Group size is the number of partners plus the invocation itself
    const uint32_t groupSize = numPartners + 1;

    // Partners are taken from a block of invocations given by the scope
    // blockSize is the number of invocations in the block
    // blockIndex is the index of this invocation in the block
    // blockPos is the position of a given index in the block
    const char *blockSize                             = nullptr;
    const char *blockIndex                            = nullptr;
    std::string (*blockPos)(const std::string &index) = nullptr;

    // Unlike compute and vertex shaders, fragment shaders take their queue family and device scope partners from the
    // DIM x DIM tile instead of the whole grid, so that partners are likely in the same triangle and run concurrently.
    const bool isTileBlock = (m_params.stage == STAGE_FRAGMENT && scope != SCOPE_SUBGROUP);
    const Scope blockScope = isTileBlock ? SCOPE_WORKGROUP : scope;

    switch (blockScope)
    {
    case SCOPE_SUBGROUP:
        blockSize  = "gl_SubgroupSize";
        blockIndex = "gl_SubgroupInvocationID";
        blockPos   = [](const std::string &index) { return "subgroupShuffle(pos, " + index + ")"; };
        break;
    case SCOPE_WORKGROUP:
        blockSize  = "DIM * DIM";
        blockIndex = "uint((pos.y % DIM) * DIM + pos.x % DIM)";
        blockPos   = [](const std::string &index)
        { return "pos - pos % DIM + ivec2(" + index + " % DIM, " + index + " / DIM)"; };
        break;
    default:
        // Queue family and device scope use the whole grid in compute and vertex shaders
        blockSize  = "gridDim * gridDim";
        blockIndex = "pos.y * gridDim + pos.x";
        blockPos   = [](const std::string &index) { return "ivec2(" + index + " % gridDim, " + index + " / gridDim)"; };
        break;
    }

    std::ostringstream css;
    css << "   uint " << name("size") << " = " << blockSize << ";\n"
        << "   uint " << name("index") << " = " << blockIndex << ";\n";

    if (isIndexRotation)
    {
        // Rotate each partner's index by stride
        css << "   uint " << name("stride") << " = " << name("size") << " / " << groupSize << "u;\n";
        for (uint32_t i = 1; i <= numPartners; i++)
        {
            css << "   uint " << partnerName("indexPartner", i) << " = (" << name("index") << " + " << i << "u * "
                << name("stride") << ") % (" << groupSize << "u * " << name("stride") << ");\n";
        }
        // An invocation is idle if it is left over after the last full group
        css << "   bool " << name("idle") << " = " << name("index") << " >= " << groupSize << "u * " << name("stride")
            << ";\n";
    }
    else
    {
        // Mirror partner's index, assumes only one partner
        css << "   uint " << partnerName("indexPartner", 1) << " = " << name("size") << " - 1u - " << name("index")
            << ";\n"
            // No invocation is left outside of a group, with odd block size the middle invocation maps to itself
            << "   bool " << name("idle") << " = false;\n";
    }

    // At subgroup scope, an invocation is also idle if any of its partners is inactive
    if (scope == SCOPE_SUBGROUP)
    {
        css << "   uvec4 " << name("ballot") << " = subgroupBallot(true);\n";
        for (uint32_t i = 1; i <= numPartners; i++)
        {
            css << "   " << name("idle") << " = " << name("idle") << " || !subgroupBallotBitExtract(" << name("ballot")
                << ", " << partnerName("indexPartner", i) << ");\n";
        }
    }

    // Returning from idle invocations can lead to barrier divergence, so we map idle invocations to themselves
    if (isIndexRotation || scope == SCOPE_SUBGROUP)
    {
        css << "   if (" << name("idle") << ") {\n";
        for (uint32_t i = 1; i <= numPartners; i++)
            css << "      " << partnerName("indexPartner", i) << " = " << name("index") << ";\n";
        css << "   }\n";
    }

    // Position, invocationId, and memory coordinates for each partner
    for (uint32_t i = 1; i <= numPartners; i++)
    {
        const std::string posPartner = partnerName("posPartner", i);
        css << "   ivec2 " << posPartner << " = " << blockPos(partnerName("indexPartner", i)) << ";\n"
            << "   uint " << partnerName("invocationIdPartner", i) << " = getInvocationId(" << posPartner << ");\n"
            << initVarCoordinates(posPartner.c_str(), partnerName("CoordPartner", i).c_str());
    }

    return css.str();
}

std::string MemoryModelTestCase::initVarCoordinates(const char *pos, const char *suffix) const
{
    std::ostringstream css;
    for (const auto &var : m_vars)
    {
        switch (var.sc)
        {
        case SC_BUFFER:
        case SC_PHYSBUFFER:
            css << "   uint " << var.name << suffix << " = getBufferCoord(" << pos << ");\n";
            break;
        case SC_IMAGE:
            css << "   ivec2 " << var.name << suffix << " = " << pos << ";\n";
            break;
        case SC_WORKGROUP:
            css << "   uint " << var.name << suffix << " = getSharedCoord(" << pos << ");\n";
            break;
        default:
            DE_ASSERT(0);
        }
    }
    return css.str();
}

void MemoryModelTestCase::checkSupportVars(const Context &context) const
{
    if (hasDataType(DATA_TYPE_UINT64) && !context.getDeviceFeatures().shaderInt64)
        TCU_THROW(NotSupportedError, "64-bit integer type is not supported");

    if (hasDataType(DATA_TYPE_FLOAT64) && !context.getDeviceFeatures().shaderFloat64)
        TCU_THROW(NotSupportedError, "64-bit floating point type is not supported");

    if (hasStorageClass(SC_PHYSBUFFER) && !context.isBufferDeviceAddressSupported())
        TCU_THROW(NotSupportedError, "Physical storage buffer pointers not supported");

    const auto &vki                     = context.getInstanceInterface();
    const auto physDevice               = context.getPhysicalDevice();
    const bool isDeviceAddressSupported = context.isDeviceFunctionalitySupported("VK_KHR_buffer_device_address");

    const VkPhysicalDeviceMemoryProperties properties = getPhysicalDeviceMemoryProperties(vki, physDevice);

    for (const auto &var : m_vars)
    {
        // Vulkan has no 64-bit floating point images, so the tests must not create them
        DE_ASSERT(var.sc != SC_IMAGE || var.type != DATA_TYPE_FLOAT64);

        if (var.sc == SC_IMAGE && var.type == DATA_TYPE_UINT64)
        {
            // A 64-bit integer image needs the Int64ImageEXT capability, also without atomics
            if (!context.getShaderImageAtomicInt64FeaturesEXT().shaderImageInt64Atomics)
                TCU_THROW(NotSupportedError, "64-bit integer images not supported");

            const VkFormatFeatureFlags requiredFeatures =
                VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT |
                VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
            const VkFormatProperties formatProperties =
                getPhysicalDeviceFormatProperties(vki, physDevice, VK_FORMAT_R64_UINT);
            if ((formatProperties.optimalTilingFeatures & requiredFeatures) != requiredFeatures)
                TCU_THROW(NotSupportedError, "VK_FORMAT_R64_UINT does not support the required format features");
        }

        if (var.sc != SC_BUFFER && var.sc != SC_PHYSBUFFER)
            continue;

        const MemoryRequirement memReq = getVarMemoryRequirement(var, isDeviceAddressSupported);
        if (getCompatibleMemoryTypes(properties, memReq) == 0)
            TCU_THROW(NotSupportedError, "Test variant uses memory type which is not supported");
    }
}

void MemoryModelTestCase::checkSupportVarAtomics(const Context &context, const char *name) const
{
    const VarDef &var = getVar(name);

    switch (var.type)
    {
    case DATA_TYPE_UINT64:
        if ((var.sc == SC_BUFFER || var.sc == SC_PHYSBUFFER) &&
            !context.getShaderAtomicInt64Features().shaderBufferInt64Atomics)
            TCU_THROW(NotSupportedError, "64-bit integer buffer atomics not supported");
        // SC_IMAGE is checked in checkSupportVars
        if (var.sc == SC_WORKGROUP && !context.getShaderAtomicInt64Features().shaderSharedInt64Atomics)
            TCU_THROW(NotSupportedError, "64-bit integer shared atomics not supported");
        break;

    case DATA_TYPE_FLOAT32:
        if (!context.isDeviceFunctionalitySupported("VK_EXT_shader_atomic_float"))
            TCU_THROW(NotSupportedError, "Missing extension: VK_EXT_shader_atomic_float");
        if ((var.sc == SC_BUFFER || var.sc == SC_PHYSBUFFER) &&
            !context.getShaderAtomicFloatFeaturesEXT().shaderBufferFloat32Atomics)
            TCU_THROW(NotSupportedError, "VkShaderAtomicFloat32: 32-bit floating point buffer atomics not supported");
        if (var.sc == SC_IMAGE && !context.getShaderAtomicFloatFeaturesEXT().shaderImageFloat32Atomics)
            TCU_THROW(NotSupportedError, "VkShaderAtomicFloat32: 32-bit floating point image atomics not supported");
        if (var.sc == SC_WORKGROUP && !context.getShaderAtomicFloatFeaturesEXT().shaderSharedFloat32Atomics)
            TCU_THROW(NotSupportedError, "VkShaderAtomicFloat32: 32-bit floating point shared atomics not supported");
        break;

    case DATA_TYPE_FLOAT64:
        if (!context.isDeviceFunctionalitySupported("VK_EXT_shader_atomic_float"))
            TCU_THROW(NotSupportedError, "Missing extension: VK_EXT_shader_atomic_float");
        if ((var.sc == SC_BUFFER || var.sc == SC_PHYSBUFFER) &&
            !context.getShaderAtomicFloatFeaturesEXT().shaderBufferFloat64Atomics)
            TCU_THROW(NotSupportedError, "VkShaderAtomicFloat64: 64-bit floating point buffer atomics not supported");
        // SC_IMAGE is rejected in checkSupportVars
        if (var.sc == SC_WORKGROUP && !context.getShaderAtomicFloatFeaturesEXT().shaderSharedFloat64Atomics)
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

void MemoryModelTestCase::checkSupportSubgroupFeatures(const Context &context, const uint32_t minSize) const
{
    const VkPhysicalDeviceSubgroupProperties &subgroupProperties = context.getSubgroupProperties();

    if (subgroupProperties.subgroupSize < minSize)
        TCU_THROW(NotSupportedError, "Device does not support required subgroup size");

    const VkSubgroupFeatureFlags requiredOperations =
        VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;

    if ((subgroupProperties.supportedOperations & requiredOperations) != requiredOperations)
        TCU_THROW(NotSupportedError, "Subgroup features not supported");

    if ((subgroupProperties.supportedStages & getShaderStageFlag(m_params.stage)) == 0)
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
    const VarDef &var = getVar(name);
    std::ostringstream css;
    if (var.sc == SC_IMAGE)
        css << "imageAtomicLoad(" << var.name << ", " << idx << ", ";
    else
        css << "atomicLoad(" << var.name << ".x[" << idx << "], ";
    css << gl_scope(scope) << ", " << gl_semantics1(sem) << ", " << gl_semantics2(sem) << ")";
    return css.str();
}

std::string MemoryModelTestCase::gl_atomicStore(const char *name, const char *idx, const char *value, const Scope scope,
                                                const uint32_t sem) const
{
    const VarDef &var   = getVar(name);
    const char *typeStr = gl_type(var.type);

    std::ostringstream css;
    if (var.sc == SC_IMAGE)
        css << "imageAtomicStore(" << var.name << ", " << idx << ", ";
    else
        css << "atomicStore(" << var.name << ".x[" << idx << "], ";
    css << typeStr << "(" << value << "), " << gl_scope(scope) << ", " << gl_semantics1(sem) << ", "
        << gl_semantics2(sem) << ")";
    return css.str();
}

std::string MemoryModelTestCase::gl_atomicExchange(const char *name, const char *idx, const char *value,
                                                   const Scope scope, const uint32_t sem) const
{
    const VarDef &var   = getVar(name);
    const char *typeStr = gl_type(var.type);

    std::ostringstream css;
    if (var.sc == SC_IMAGE)
        css << "imageAtomicExchange(" << var.name << ", " << idx << ", ";
    else
        css << "atomicExchange(" << var.name << ".x[" << idx << "], ";
    css << typeStr << "(" << value << "), " << gl_scope(scope) << ", " << gl_semantics1(sem) << ", "
        << gl_semantics2(sem) << ")";
    return css.str();
}

std::string MemoryModelTestCase::gl_load(const char *name, const char *idx) const
{
    const VarDef &var = getVar(name);
    std::ostringstream css;
    if (var.sc == SC_IMAGE)
        css << "imageLoad(" << var.name << ", " << idx << ").x";
    else
        css << var.name << ".x[" << idx << "]";
    return css.str();
}

std::string MemoryModelTestCase::gl_store(const char *name, const char *idx, const char *value) const
{
    const VarDef &var = getVar(name);
    std::ostringstream css;
    if (var.sc != SC_IMAGE)
        css << var.name << ".x[" << idx << "] = " << gl_type(var.type) << "(" << value << ")";
    else if (var.type == DATA_TYPE_UINT64)
        css << "imageStore(" << var.name << ", " << idx << ", u64vec4(uint64_t(" << value << "), 0, 0, 0))";
    else if (isIntType(var.type))
        css << "imageStore(" << var.name << ", " << idx << ", uvec4(" << value << ", 0, 0, 0))";
    else
        css << "imageStore(" << var.name << ", " << idx << ", vec4(" << gl_type(var.type) << "(" << value
            << "), 0, 0, 0))";
    return css.str();
}

std::string MemoryModelTestCase::spv_load(const char *name, const char *idx) const
{
    const VarDef &var        = getVar(name);
    const uint32_t alignment = is64BitType(var.type) ? 8 : 4;
    const uint32_t operands  = SPV_ALIGNED | SPV_NON_PRIVATE_POINTER;

    std::ostringstream css;
    css << "spv_" << name << "Load(" << name << ".x[" << idx << "], " << std::showbase << std::hex << operands
        << std::dec << ", " << alignment << ")";
    return css.str();
}

std::string MemoryModelTestCase::spv_load(const char *name, const char *idx, const Scope scope) const
{
    const VarDef &var        = getVar(name);
    const uint32_t alignment = is64BitType(var.type) ? 8 : 4;
    const uint32_t operands  = SPV_ALIGNED | SPV_NON_PRIVATE_POINTER | SPV_MAKE_POINTER_VISIBLE;

    std::ostringstream css;
    css << "spv_" << name << "LoadVis(" << name << ".x[" << idx << "], " << std::showbase << std::hex << operands
        << std::dec << ", " << alignment << ", " << gl_scope(scope) << ")";
    return css.str();
}

std::string MemoryModelTestCase::spv_store(const char *name, const char *idx, const char *value) const
{
    const VarDef &var        = getVar(name);
    const uint32_t alignment = is64BitType(var.type) ? 8 : 4;
    const uint32_t operands  = SPV_ALIGNED | SPV_NON_PRIVATE_POINTER;

    std::ostringstream css;
    css << "spv_" << name << "Store(" << name << ".x[" << idx << "], " << value << ", " << std::showbase << std::hex
        << operands << std::dec << ", " << alignment << ")";
    return css.str();
}

std::string MemoryModelTestCase::spv_store(const char *name, const char *idx, const char *value,
                                           const Scope scope) const
{
    const VarDef &var        = getVar(name);
    const uint32_t alignment = is64BitType(var.type) ? 8 : 4;
    const uint32_t operands  = SPV_ALIGNED | SPV_NON_PRIVATE_POINTER | SPV_MAKE_POINTER_AVAILABLE;

    std::ostringstream css;
    css << "spv_" << name << "StoreAv(" << name << ".x[" << idx << "], " << value << ", " << std::showbase << std::hex
        << operands << std::dec << ", " << alignment << ", " << gl_scope(scope) << ")";
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

const VarDef &MemoryModelTestCase::getVar(const char *name) const
{
    for (const auto &var : m_vars)
    {
        if (name == std::string_view(var.name))
            return var;
    }
    TCU_THROW(InternalError, std::string("Unknown var: ") + name);
}

bool MemoryModelTestCase::hasStorageClass(const StorageClass sc) const
{
    return std::any_of(m_vars.begin(), m_vars.end(), [sc](const VarDef &b) { return b.sc == sc; });
}

bool MemoryModelTestCase::hasDataType(const DataType type) const
{
    return std::any_of(m_vars.begin(), m_vars.end(), [type](const VarDef &b) { return b.type == type; });
}

uint32_t MemoryModelTestCase::getVarSemantics(const char *name) const
{
    return scToSem(getVar(name).sc);
}

namespace
{

class MemoryModelTestInstance : public vkt::MultiQueueRunnerTestInstance
{
public:
    MemoryModelTestInstance(Context &context, Stage stage, const std::vector<VarDef> &vars)
        : vkt::MultiQueueRunnerTestInstance(context, stage == STAGE_COMPUTE ? vkt::COMPUTE_QUEUE : vkt::GRAPHICS_QUEUE)
        , m_stage(stage)
        , m_vars(vars)
    {
    }

    tcu::TestStatus queuePass(const vkt::QueueData &queueData) override;

private:
    const Stage m_stage;
    const std::vector<VarDef> m_vars;
};

tcu::TestStatus MemoryModelTestInstance::queuePass(const vkt::QueueData &queueData)
{
    const DeviceInterface &vk = m_context.getDeviceInterface();
    const VkDevice device     = m_context.getDevice();
    Allocator &allocator      = m_context.getDefaultAllocator();

    const VkShaderStageFlags shaderStageFlags     = getShaderStageFlag(m_stage);
    const VkPipelineStageFlags pipelineStageFlags = getPipelineStageFlag(m_stage);

    // Workgroup dimension, shrink workgroup size to fit HW limits if necessary
    uint32_t dim = 31;

    const VkPhysicalDeviceLimits &limits = m_context.getDeviceProperties().limits;
    if (dim * dim > limits.maxComputeWorkGroupInvocations)
        dim = (uint32_t)deFloatSqrt((float)limits.maxComputeWorkGroupInvocations);

    const uint32_t DIM                    = dim;
    const uint32_t NUM_WORKGROUP_EACH_DIM = 8;
    const uint32_t GRID_DIM               = DIM * NUM_WORKGROUP_EACH_DIM;
    const uint32_t NUM_INVOCATIONS        = GRID_DIM * GRID_DIM;

    // Create buffers and images, the last variable is always the result buffer
    const uint32_t numVars      = static_cast<uint32_t>(m_vars.size());
    const uint32_t idxResultVar = numVars - 1;
    DE_ASSERT(!m_vars.empty() && m_vars[idxResultVar].sc == SC_BUFFER && m_vars[idxResultVar].type == DATA_TYPE_UINT);

    std::vector<de::MovePtr<BufferWithMemory>> buffers(numVars);
    std::vector<de::MovePtr<ImageWithMemory>> images(numVars);
    std::vector<Move<VkImageView>> imageViews(numVars);

    const bool isDeviceAddressSupported = m_context.isDeviceFunctionalitySupported("VK_KHR_buffer_device_address");
    const VkImageSubresourceRange range = makeImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u);

    for (uint32_t i = 0; i < numVars; ++i)
    {
        const VarDef &var = m_vars[i];

        switch (var.sc)
        {
        case SC_BUFFER:
        case SC_PHYSBUFFER:
        {
            const size_t elementSize       = is64BitType(var.type) ? sizeof(uint64_t) : sizeof(uint32_t);
            const size_t bufferSize        = NUM_INVOCATIONS * elementSize;
            const MemoryRequirement memReq = getVarMemoryRequirement(var, isDeviceAddressSupported);
            const VkBufferUsageFlags usageFlags =
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                (var.sc == SC_PHYSBUFFER ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0);

            buffers[i] = de::MovePtr<BufferWithMemory>(
                new BufferWithMemory(vk, device, allocator, makeBufferCreateInfo(bufferSize, usageFlags), memReq));
            break;
        }
        case SC_IMAGE:
        {
            const VkFormat imageFormat     = getImageFormat(var.type);
            const MemoryRequirement memReq = var.local ? MemoryRequirement::Local : MemoryRequirement::NonLocal;

            VkImageCreateInfo imageCreateInfo = initVulkanStructure();
            imageCreateInfo.imageType         = VK_IMAGE_TYPE_2D;
            imageCreateInfo.format            = imageFormat;
            imageCreateInfo.extent            = makeExtent3D(GRID_DIM, GRID_DIM, 1u);
            imageCreateInfo.mipLevels         = 1u;
            imageCreateInfo.arrayLayers       = 1u;
            imageCreateInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
            imageCreateInfo.usage =
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

            try
            {
                images[i] =
                    de::MovePtr<ImageWithMemory>(new ImageWithMemory(vk, device, allocator, imageCreateInfo, memReq));
            }
            catch (const tcu::NotSupportedError &)
            {
                if (var.local)
                    throw;
                TCU_THROW(NotSupportedError, "Test variant uses non-device-local memory, which is not supported");
            }

            imageViews[i] = makeImageView(vk, device, **images[i], VK_IMAGE_VIEW_TYPE_2D, imageFormat, range);
            break;
        }
        case SC_WORKGROUP:
            break;
        default:
            DE_ASSERT(0);
        }
    }

    // Initialize layout and descriptor set
    DescriptorSetLayoutBuilder layoutBuilder;
    for (uint32_t i = 0; i < numVars; ++i)
    {
        if (m_vars[i].sc == SC_BUFFER)
            layoutBuilder.addSingleIndexedBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, shaderStageFlags, i);
        else if (m_vars[i].sc == SC_IMAGE)
            layoutBuilder.addSingleIndexedBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, shaderStageFlags, i);
    }

    const Unique<VkDescriptorSetLayout> descriptorSetLayout(layoutBuilder.build(vk, device));

    const Unique<VkDescriptorPool> descriptorPool(
        DescriptorPoolBuilder()
            .addType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, numVars)
            .addType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, numVars)
            .build(vk, device, VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, 1u));

    const Unique<VkDescriptorSet> descriptorSet(makeDescriptorSet(vk, device, *descriptorPool, *descriptorSetLayout));

    DescriptorSetUpdateBuilder setUpdateBuilder;
    for (uint32_t i = 0; i < numVars; ++i)
    {
        if (m_vars[i].sc == SC_BUFFER)
        {
            const VkDescriptorBufferInfo bufferInfo = makeDescriptorBufferInfo(**buffers[i], 0, VK_WHOLE_SIZE);
            setUpdateBuilder.writeSingle(*descriptorSet, DescriptorSetUpdateBuilder::Location::binding(i),
                                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &bufferInfo);
        }
        else if (m_vars[i].sc == SC_IMAGE)
        {
            const VkDescriptorImageInfo imageInfo =
                makeDescriptorImageInfo(VK_NULL_HANDLE, *imageViews[i], VK_IMAGE_LAYOUT_GENERAL);
            setUpdateBuilder.writeSingle(*descriptorSet, DescriptorSetUpdateBuilder::Location::binding(i),
                                         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageInfo);
        }
    }
    setUpdateBuilder.update(vk, device);

    // Physical storage buffer addresses are push constants at an offset given by the variable index
    // The result buffer is never a physical storage buffer, so the range ends before it
    const VkPushConstantRange pushConstRange = {shaderStageFlags, 0, idxResultVar * DEVICE_ADDRESS_SIZE};
    const bool hasPhysBuffers =
        std::any_of(m_vars.begin(), m_vars.end(), [](const VarDef &b) { return b.sc == SC_PHYSBUFFER; });
    const Unique<VkPipelineLayout> pipelineLayout(
        makePipelineLayout(vk, device, *descriptorSetLayout, hasPhysBuffers ? &pushConstRange : nullptr));

    const VkPipelineBindPoint bindPoint =
        m_stage == STAGE_COMPUTE ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS;

    const VkSpecializationMapEntry entries[2] = {
        makeSpecializationMapEntry(0u, 0u, sizeof(uint32_t)),
        makeSpecializationMapEntry(1u, sizeof(uint32_t), sizeof(uint32_t)),
    };
    const uint32_t specData[2]          = {DIM, NUM_WORKGROUP_EACH_DIM};
    const VkSpecializationInfo specInfo = makeSpecializationInfo(2u, entries, sizeof(specData), specData);

    // Initialize pipeline and command buffer
    Move<VkPipeline> pipeline;
    Move<VkRenderPass> renderPass;
    Move<VkFramebuffer> framebuffer;

    if (m_stage == STAGE_COMPUTE)
    {
        const Unique<VkShaderModule> shader(
            createShaderModule(vk, device, m_context.getBinaryCollection().get("test"), 0));
        pipeline = makeComputePipeline(vk, device, *pipelineLayout, 0u, nullptr, *shader, 0u, &specInfo);
    }
    else
    {
        // Vertex tests run "test" as the only shader stage, fragment tests pair it with the fullscreen "vert" shader
        // Without a fragment stage, makeGraphicsPipeline disables rasterization
        const bool isVertexStage = (m_stage == STAGE_VERTEX);
        const Unique<VkShaderModule> vertexShader(
            createShaderModule(vk, device, m_context.getBinaryCollection().get(isVertexStage ? "test" : "vert"), 0));
        Move<VkShaderModule> fragmentShader;
        std::vector<VkPipelineShaderStageCreateInfo> shaderStages{
            makePipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, *vertexShader, &specInfo)};
        if (!isVertexStage)
        {
            fragmentShader = createShaderModule(vk, device, m_context.getBinaryCollection().get("test"), 0);
            shaderStages.push_back(
                makePipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, *fragmentShader, &specInfo));
        }

        const VkPipelineVertexInputStateCreateInfo vertexInputState = initVulkanStructure();
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyState   = initVulkanStructure();
        inputAssemblyState.topology =
            isVertexStage ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;

        const std::vector<VkViewport> viewports{makeViewport(GRID_DIM, GRID_DIM)};
        const std::vector<VkRect2D> scissors{makeRect2D(GRID_DIM, GRID_DIM)};

        renderPass  = makeRenderPass(vk, device);
        framebuffer = makeFramebuffer(vk, device, *renderPass, 0, nullptr, GRID_DIM, GRID_DIM);
        pipeline    = makeGraphicsPipeline(vk, device, VK_NULL_HANDLE, *pipelineLayout, 0u, shaderStages, *renderPass,
                                           viewports, scissors, 0u, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                           &vertexInputState, &inputAssemblyState);
    }

    const Unique<VkCommandPool> cmdPool(makeCommandPool(vk, device, queueData.familyIndex));
    const Unique<VkCommandBuffer> cmdBuffer(
        allocateCommandBuffer(vk, device, *cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY));

    // Initialize the buffer the CPU will read from
    // Try to use cached host memory, fallback to host visible memory
    de::MovePtr<BufferWithMemory> copyBuffer;
    const size_t sizeCopyBuffer = sizeof(uint32_t) * NUM_INVOCATIONS;
    try
    {
        copyBuffer = de::MovePtr<BufferWithMemory>(new BufferWithMemory(
            vk, device, allocator, makeBufferCreateInfo(sizeCopyBuffer, VK_BUFFER_USAGE_TRANSFER_DST_BIT),
            MemoryRequirement::HostVisible | MemoryRequirement::Cached));
    }
    catch (const tcu::NotSupportedError &)
    {
        copyBuffer = de::MovePtr<BufferWithMemory>(new BufferWithMemory(
            vk, device, allocator, makeBufferCreateInfo(sizeCopyBuffer, VK_BUFFER_USAGE_TRANSFER_DST_BIT),
            MemoryRequirement::HostVisible));
    }

    const VkBufferCopy copyParams = makeBufferCopy(0u, 0u, sizeCopyBuffer);
    const VkClearValue clearColor = makeClearValueColorU32(0, 0, 0, 0);

    constexpr uint32_t NUM_SUBMITS    = 4;
    constexpr uint32_t NUM_ITERATIONS = 50;

    for (uint32_t submit = 0; submit < NUM_SUBMITS; submit++)
    {
        beginCommandBuffer(vk, *cmdBuffer, 0u);

        if (submit == 0)
        {
            // The result buffer is cleared once and collects the failures of all submits and iterations
            vk.cmdFillBuffer(*cmdBuffer, **buffers[idxResultVar], 0, VK_WHOLE_SIZE, 0);

            for (uint32_t i = 0; i < numVars; ++i)
            {
                if (m_vars[i].sc != SC_IMAGE)
                    continue;

                const VkImageMemoryBarrier imageBarrier =
                    makeImageMemoryBarrier(0u, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                                           VK_IMAGE_LAYOUT_GENERAL, **images[i], range);
                cmdPipelineImageMemoryBarrier(vk, *cmdBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                              VK_PIPELINE_STAGE_TRANSFER_BIT, &imageBarrier);
            }
        }

        vk.cmdBindDescriptorSets(*cmdBuffer, bindPoint, *pipelineLayout, 0u, 1, &*descriptorSet, 0u, nullptr);
        vk.cmdBindPipeline(*cmdBuffer, bindPoint, *pipeline);

        for (uint32_t i = 0; i < numVars; ++i)
        {
            if (m_vars[i].sc != SC_PHYSBUFFER)
                continue;

            const VkDeviceAddress address = getBufferDeviceAddress(vk, device, **buffers[i]);
            vk.cmdPushConstants(*cmdBuffer, *pipelineLayout, shaderStageFlags, DEVICE_ADDRESS_SIZE * i,
                                DEVICE_ADDRESS_SIZE, &address);
        }

        for (uint32_t iteration = 0; iteration < NUM_ITERATIONS; iteration++)
        {
            for (uint32_t i = 0; i < idxResultVar; ++i)
            {
                if (buffers[i])
                    vk.cmdFillBuffer(*cmdBuffer, **buffers[i], 0, VK_WHOLE_SIZE, 0);
                if (images[i])
                    vk.cmdClearColorImage(*cmdBuffer, **images[i], VK_IMAGE_LAYOUT_GENERAL, &clearColor.color, 1,
                                          &range);
            }

            const VkMemoryBarrier transferToShader =
                makeMemoryBarrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            cmdPipelineMemoryBarrier(vk, *cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, pipelineStageFlags,
                                     &transferToShader);

            if (m_stage == STAGE_COMPUTE)
            {
                vk.cmdDispatch(*cmdBuffer, NUM_WORKGROUP_EACH_DIM, NUM_WORKGROUP_EACH_DIM, 1);
            }
            else
            {
                // Draw a point cloud for vertex shader testing, and a single quad for fragment shader testing
                beginRenderPass(vk, *cmdBuffer, *renderPass, *framebuffer, makeRect2D(GRID_DIM, GRID_DIM), 0, nullptr,
                                VK_SUBPASS_CONTENTS_INLINE);
                vk.cmdDraw(*cmdBuffer, m_stage == STAGE_VERTEX ? NUM_INVOCATIONS : 4u, 1u, 0u, 0u);
                endRenderPass(vk, *cmdBuffer);
            }

            const VkMemoryBarrier shaderToTransfer =
                makeMemoryBarrier(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                  VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            cmdPipelineMemoryBarrier(vk, *cmdBuffer, pipelineStageFlags, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     &shaderToTransfer);
        }

        if (submit == NUM_SUBMITS - 1)
        {
            vk.cmdCopyBuffer(*cmdBuffer, **buffers[idxResultVar], **copyBuffer, 1, &copyParams);
            const VkMemoryBarrier transferToHost =
                makeMemoryBarrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            cmdPipelineMemoryBarrier(vk, *cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                     &transferToHost);
        }

        endCommandBuffer(vk, *cmdBuffer);
        submitCommandsAndWait(vk, device, queueData.handle, cmdBuffer.get());
        m_context.resetCommandPoolForVKSC(device, *cmdPool);
    }

    invalidateAlloc(vk, device, copyBuffer->getAllocation());

    tcu::TestLog &log                = m_context.getTestContext().getLog();
    const uint32_t *ptr              = static_cast<const uint32_t *>(copyBuffer->getAllocation().getHostPtr());
    const uint32_t MAX_LOGGED_ERRORS = 256;

    uint32_t numErrors = 0;
    for (uint32_t i = 0; i < NUM_INVOCATIONS; ++i)
    {
        if (ptr[i] == 0)
            continue;
        if (numErrors++ < MAX_LOGGED_ERRORS)
            log << tcu::TestLog::Message << "Failed invocation: " << i << tcu::TestLog::EndMessage;
    }

    if (numErrors == 0)
        return tcu::TestStatus::pass("Pass");

    log << tcu::TestLog::Message << "Total Errors: " << numErrors << tcu::TestLog::EndMessage;
    return tcu::TestStatus::fail("Fail");
}

} // namespace

TestInstance *MemoryModelTestCase::createInstance(Context &context) const
{
    return new MemoryModelTestInstance(context, m_params.stage, m_vars);
}

} // namespace MemoryModel
} // namespace vkt
