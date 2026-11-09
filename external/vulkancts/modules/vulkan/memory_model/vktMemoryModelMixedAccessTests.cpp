/*------------------------------------------------------------------------
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
 * \brief Memory Model Mixed Access Tests
 *//*--------------------------------------------------------------------*/

#include "vktMemoryModelMixedAccessTests.hpp"
#include "vktMemoryModelTestCase.hpp"
#include "vktMemoryModelUtil.hpp"

#include "deUniquePtr.hpp"
#include "tcuTestCase.hpp"
#include "vktTestCase.hpp"

#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace vkt::MemoryModel
{
namespace
{

struct CaseDef
{
    bool core11;
    bool payloadMemLocal;
    StorageClass payloadSC;
    DataType payloadType;
    Scope syncScope;
    std::optional<Scope> visScope;
    Stage stage;
};

MemQualifier getMemQualifier(const CaseDef &data)
{
    if (data.core11)
        return MEM_COHERENT;
    if (!data.visScope)
        return MEM_NONPRIVATE;
    return scopeToQualifier(*data.visScope);
}

Scope getMappingScope(const CaseDef &data)
{
    const bool hasSharedBuffer = data.payloadSC == SC_WORKGROUP;
    if ((data.syncScope == SCOPE_DEVICE || data.syncScope == SCOPE_QUEUEFAMILY) && hasSharedBuffer)
    {
        return SCOPE_WORKGROUP;
    }
    return data.syncScope;
}

// Memory qualifiers cannot be used on workgroup memory, so with the Vulkan memory model its non-atomic loads
// get their memory access operands (NonPrivatePointer, MakePointerVisible) from a SPIR-V intrinsic
bool usesLoadIntrinsic(const CaseDef &data)
{
    return !data.core11 && data.payloadSC == SC_WORKGROUP;
}

std::vector<BufferDef> createBuffers(const CaseDef &data)
{
    const MemQualifier payloadQualifier = getMemQualifier(data);
    return std::vector<BufferDef>{{"payload", payloadQualifier, data.payloadSC, data.payloadType, data.payloadMemLocal},
                                  {"fail", MEM_NONE, SC_BUFFER, DATA_TYPE_UINT, true}};
}

class MixedAccessTestCase : public MemoryModelTestCase
{
public:
    MixedAccessTestCase(tcu::TestContext &context, const char *name, const CaseDef &data);
    void checkSupport(Context &context) const override;

private:
    std::string declareSpirvIntrinsics() const override;
    std::string initProgram() const override;
    const CaseDef m_data;
};

MixedAccessTestCase::MixedAccessTestCase(tcu::TestContext &context, const char *name, const CaseDef &data)
    : MemoryModelTestCase(context, name, {data.core11, data.stage, getMappingScope(data), 1}, createBuffers(data))
    , m_data(data)
{
}

void MixedAccessTestCase::checkSupport(Context &context) const
{
    if (!context.contextSupports(vk::ApiVersion(0, 1, 1, 0)))
        TCU_THROW(NotSupportedError, "Vulkan 1.1 not supported");

    if (!m_data.core11)
    {
        if (!context.getVulkanMemoryModelFeatures().vulkanMemoryModel)
            TCU_THROW(NotSupportedError, "vulkanMemoryModel not supported");

        if ((m_data.syncScope == SCOPE_DEVICE || m_data.visScope == SCOPE_DEVICE) &&
            !context.getVulkanMemoryModelFeatures().vulkanMemoryModelDeviceScope)
        {
            TCU_THROW(NotSupportedError, "vulkanMemoryModelDeviceScope not supported");
        }
    }

    if (m_data.syncScope == SCOPE_SUBGROUP)
        checkSupportSubgroupFeatures(context);

    checkSupportBuffers(context);
    checkSupportBufferAtomics(context, "payload");
    checkSupportStageAtomics(context);
}

std::string MixedAccessTestCase::declareSpirvIntrinsics() const
{
    if (!usesLoadIntrinsic(m_data))
        return {};

    const char *typeStr            = gl_type(getBuffer("payload").type);
    const bool hasDeviceVisibility = m_data.visScope == SCOPE_DEVICE;
    std::ostringstream css;
    css << "#extension GL_EXT_spirv_intrinsics : enable\n"
        << "spirv_instruction(" << (hasDeviceVisibility ? "capabilities = [5346], " : "") << "id = 61) " << typeStr
        << " loadPayload(spirv_by_reference " << typeStr << " p, spirv_literal int memoryAccess"
        << (m_data.visScope.has_value() ? ", int scope" : "") << ");\n";
    return css.str();
}

std::string MixedAccessTestCase::initProgram() const
{
    const Scope syncScope = m_data.syncScope;
    const DataType type   = getBuffer("payload").type;
    const char *typeStr   = gl_type(type);

    // Non-atomic load of the partner's payload, used for the prefetch and for the checked read
    std::string load = gl_load("payload", "payloadCoordPartner1");
    if (usesLoadIntrinsic(m_data))
    {
        const std::string operands = m_data.visScope ? std::string("0x30, ") + gl_scope(*m_data.visScope) : "0x20";
        load                       = "loadPayload(payload.x[payloadCoordPartner1], " + operands + ")";
    }

    std::ostringstream css;

    css << "   uint value = invocationId;\n"
        << "   " << typeStr << " r0 = 0;\n"
        << "   " << typeStr << " r1 = 0;\n"
        << "   " << typeStr << " expected = " << typeStr << "(invocationIdPartner1);\n";

    // Prefetch the partner's payload; this adds 0 because every stored value is below 2^31
    css << "   value += " << gl_toUint(type, load) << " >> 31;\n"
        << "   " << gl_atomicStore("payload", "payloadCoord", "value", syncScope, 0) << ";\n"
        << "   r0 = " << gl_atomicLoad("payload", "payloadCoordPartner1", syncScope, 0) << ";\n"
        << "   if (r0 == expected) {\n"
        << "      r1 = " << load << ";\n"
        << "      if (!idle && r1 != expected) {\n"
        << "         " << gl_store("fail", "failCoord", "1") << ";\n"
        << "      }\n"
           "   }\n";

    return css.str();
}

} // namespace

tcu::TestCaseGroup *createMixedAccessTests(tcu::TestContext &testCtx)
{
    static const NamedValue<bool> core11Cases[] = {
        {true, "core11"},
        {false, "ext"},
    };

    static const NamedValue<DataType> typeCases[] = {
        {DATA_TYPE_UINT, "u32"},
        {DATA_TYPE_UINT64, "u64"},
        {DATA_TYPE_FLOAT32, "f32"},
        {DATA_TYPE_FLOAT64, "f64"},
    };

    static const NamedValue<bool> localCases[] = {
        {false, "payload_nonlocal"},
        {true, "payload_local"},
    };

    static const NamedValue<StorageClass> scCases[] = {
        {SC_BUFFER, "buffer"},
        {SC_IMAGE, "image"},
        {SC_WORKGROUP, "workgroup"},
        {SC_PHYSBUFFER, "physbuffer"},
    };

    static const NamedValue<Scope> syncScopeCases[] = {
        {SCOPE_DEVICE, "device"},
        {SCOPE_QUEUEFAMILY, "queuefamily"},
        {SCOPE_WORKGROUP, "workgroup"},
        {SCOPE_SUBGROUP, "subgroup"},
    };

    static const NamedValue<std::optional<Scope>> visScopeCases[] = {
        {std::nullopt, "nonprivate"},          {SCOPE_SUBGROUP, "vissubgroup"}, {SCOPE_WORKGROUP, "visworkgroup"},
        {SCOPE_QUEUEFAMILY, "visqueuefamily"}, {SCOPE_DEVICE, "visdevice"},
    };

    static const NamedValue<Stage> stageCases[] = {
        {STAGE_COMPUTE, "comp"},
        {STAGE_VERTEX, "vert"},
        {STAGE_FRAGMENT, "frag"},
    };

    de::MovePtr<tcu::TestCaseGroup> ttGroup(new tcu::TestCaseGroup(testCtx, "mixed_access"));

    for (const auto &[core11, type, local, sc, syncScope, visScope, stage] :
         Combinations(core11Cases, typeCases, localCases, scCases, syncScopeCases, visScopeCases, stageCases))
    {
        CaseDef c = {
            core11.value,    // bool core11;
            local.value,     // bool payloadMemLocal;
            sc.value,        // StorageClass payloadSC;
            type.value,      // DataType payloadType;
            syncScope.value, // Scope syncScope;
            visScope.value,  // std::optional<Scope> visScope;
            stage.value,     // Stage stage;
        };

        // Mustpass11 tests should only exercise things we expect to work on existing implementations. Exclude
        // noncoherent tests which require new extensions. Exclude queuefamily scope which doesn't exist in Vulkan 1.1.
        // Exclude physical storage buffer which doesn't support coherent decoration.
        if (c.core11 && (c.visScope || c.payloadType == DATA_TYPE_UINT64 || c.payloadType == DATA_TYPE_FLOAT64 ||
                         c.syncScope == SCOPE_QUEUEFAMILY || c.payloadSC == SC_PHYSBUFFER))
        {
            continue;
        }

        // Don't exercise workgroup memory for non-compute stages
        if (c.stage != STAGE_COMPUTE &&
            (c.syncScope == SCOPE_WORKGROUP || c.payloadSC == SC_WORKGROUP || c.visScope == SCOPE_WORKGROUP))
        {
            continue;
        }

        // Don't exercise local and non-local for workgroup memory
        if (c.payloadSC == SC_WORKGROUP && c.payloadMemLocal != 0)
        {
            continue;
        }

        // No 64-bit image types, so skip tests for 64-bit types
        if (c.payloadSC == SC_IMAGE && (c.payloadType == DATA_TYPE_UINT64 || c.payloadType == DATA_TYPE_FLOAT64))
        {
            continue;
        }

        const bool coherent = core11.value && !visScope.value.has_value();
        addTestCase(
            *ttGroup,
            {core11.name, type.name, local.name, sc.name, syncScope.name, coherent ? "coherent" : visScope.name},
            new MixedAccessTestCase(testCtx, stage.name, c));
    }

    return ttGroup.release();
}

} // namespace vkt::MemoryModel
