/*------------------------------------------------------------------------
 * Vulkan Conformance Tests
 * ------------------------
 *
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
    DataType dataType;
    StorageClass payloadSC;
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
    const bool hasSharedVar = data.payloadSC == SC_WORKGROUP;
    if ((data.syncScope == SCOPE_DEVICE || data.syncScope == SCOPE_QUEUEFAMILY) && hasSharedVar)
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

std::vector<VarDef> createVars(const CaseDef &data)
{
    const MemQualifier payloadQualifier = getMemQualifier(data);
    return std::vector<VarDef>{{"payload", payloadQualifier, data.payloadSC, data.dataType, data.payloadMemLocal},
                               {"fail", MEM_NONE, SC_BUFFER, DATA_TYPE_UINT, true}};
}

class MixedAccessTestCase : public MemoryModelTestCase
{
public:
    MixedAccessTestCase(tcu::TestContext &context, const char *name, const CaseDef &data);
    void checkSupport(Context &context) const override;

private:
    std::string declareSpvIntrinsics() const override;
    std::string initProgram() const override;
    const CaseDef m_data;
};

MixedAccessTestCase::MixedAccessTestCase(tcu::TestContext &context, const char *name, const CaseDef &data)
    : MemoryModelTestCase(context, name, {data.stage, data.core11}, createVars(data))
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

    checkSupportVars(context);
    checkSupportVarAtomics(context, "payload");
    checkSupportStageAtomics(context);
}

std::string MixedAccessTestCase::declareSpvIntrinsics() const
{
    if (!usesLoadIntrinsic(m_data))
        return {};

    std::ostringstream css;
    css << "#extension GL_EXT_spirv_intrinsics : enable\n"
        << declareVarSpvIntrinsics("payload", m_data.visScope == SCOPE_DEVICE);
    return css.str();
}

std::string MixedAccessTestCase::initProgram() const
{
    const Scope syncScope = m_data.syncScope;
    const DataType type   = getVar("payload").type;
    const char *typeStr   = gl_type(type);

    std::string loadPayload;
    if (usesLoadIntrinsic(m_data))
    {
        if (m_data.visScope.has_value())
            loadPayload = spv_load("payload", "payloadCoordPartner", *m_data.visScope);
        else
            loadPayload = spv_load("payload", "payloadCoordPartner");
    }
    else
    {
        loadPayload = gl_load("payload", "payloadCoordPartner");
    }

    std::ostringstream css;

    css << initPartnerCoordinates(getMappingScope(m_data));

    css << "   uint value = invocationId;\n"
        << "   " << typeStr << " r1 = 0;\n"
        << "   " << typeStr << " r2 = 0;\n"
        << "   " << typeStr << " expected = " << typeStr << "(invocationIdPartner);\n";

    // Prefetch the partner's payload; this adds 0 because every stored value is below 2^31
    css << "   value += " << gl_toUint(type, loadPayload) << " >> 31;\n"
        << "   " << gl_atomicStore("payload", "payloadCoord", "value", syncScope, 0) << ";\n"
        << "   r1 = " << gl_atomicLoad("payload", "payloadCoordPartner", syncScope, 0) << ";\n"
        << "   r2 = " << loadPayload << ";\n"
        << "   if (!idle && r1 == expected && r2 != expected) {\n"
        << "      " << gl_store("fail", "failCoord", "1") << ";\n"
        << "   }\n";

    return css.str();
}

} // namespace

tcu::TestCaseGroup *createMixedAccessTests(tcu::TestContext &testCtx)
{
    static const NamedValue<bool> core11Cases[] = {
        // supported by Vulkan1.1
        {true, "core11"},
        // requires VK_KHR_vulkan_memory_model extension
        {false, "ext"},
    };

    static const NamedValue<DataType> typeCases[] = {
        // uint32_t atomics
        {DATA_TYPE_UINT, "u32"},
        // uint64_t atomics
        {DATA_TYPE_UINT64, "u64"},
        // float32 atomics
        {DATA_TYPE_FLOAT32, "f32"},
        // float64 atomics
        {DATA_TYPE_FLOAT64, "f64"},
    };

    static const NamedValue<StorageClass> scCases[] = {
        // variable in buffer memory
        {SC_BUFFER, "buffer"},
        // variable in image memory
        {SC_IMAGE, "image"},
        // variable in workgroup memory
        {SC_WORKGROUP, "workgroup"},
        // variable in physical storage buffer memory
        {SC_PHYSBUFFER, "physbuffer"},
    };

    static const NamedValue<bool> localCases[] = {
        // variable in non-local memory
        {false, "nonlocal"},
        // variable in local memory
        {true, "local"},
    };

    static const NamedValue<Scope> syncScopeCases[] = {
        // device scope
        {SCOPE_DEVICE, "device"},
        // queue family scope
        {SCOPE_QUEUEFAMILY, "queuefamily"},
        // workgroup scope
        {SCOPE_WORKGROUP, "workgroup"},
        // subgroup scope
        {SCOPE_SUBGROUP, "subgroup"},
    };

    static const NamedValue<std::optional<Scope>> visScopeCases[] = {
        // no visibility scope
        {std::nullopt, "nonprivate"},
        // subgroup scope visibility
        {SCOPE_SUBGROUP, "vissubgroup"},
        // workgroup scope visibility
        {SCOPE_WORKGROUP, "visworkgroup"},
        // queue family scope visibility
        {SCOPE_QUEUEFAMILY, "visqueuefamily"},
        // device scope visibility
        {SCOPE_DEVICE, "visdevice"},
    };

    static const NamedValue<Stage> stageCases[] = {
        // compute stage
        {STAGE_COMPUTE, "comp"},
        // vertex stage
        {STAGE_VERTEX, "vert"},
        // fragment stage
        {STAGE_FRAGMENT, "frag"},
    };

    de::MovePtr<tcu::TestCaseGroup> group(new tcu::TestCaseGroup(testCtx, "mixed_access"));

    const auto groupName = [](const char *prefix, const char *name) { return std::string(prefix) + "_" + name; };

    for (const auto &[core11, type, local, sc, syncScope, visScope, stage] :
         Combinations(core11Cases, typeCases, localCases, scCases, syncScopeCases, visScopeCases, stageCases))
    {
        CaseDef c = {
            core11.value,    // bool core11;
            local.value,     // bool payloadMemLocal;
            type.value,      // DataType dataType;
            sc.value,        // StorageClass payloadSC;
            syncScope.value, // Scope syncScope;
            visScope.value,  // std::optional<Scope> visScope;
            stage.value,     // Stage stage;
        };

        // Mustpass11 tests should only exercise variations that are expected to work without Vulkan memory model.
        // Exclude explicit visibility scope, which requires Vulkan memory model. Exclude queue family scope which
        // doesn't exist in Vulkan 1.1. Exclude physical storage buffer which doesn't support coherent decoration.
        if (c.core11 && (c.visScope || c.dataType == DATA_TYPE_UINT64 || c.dataType == DATA_TYPE_FLOAT64 ||
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
        if (c.payloadSC == SC_IMAGE && (c.dataType == DATA_TYPE_UINT64 || c.dataType == DATA_TYPE_FLOAT64))
        {
            continue;
        }

        const bool coherent = core11.value && !visScope.value.has_value();
        addTestCase(*group,
                    {core11.name, type.name, groupName("payload", local.name), sc.name, syncScope.name,
                     coherent ? "coherent" : visScope.name},
                    new MixedAccessTestCase(testCtx, stage.name, c));
    }

    return group.release();
}

} // namespace vkt::MemoryModel
