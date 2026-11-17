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
    if ((data.payloadSC == SC_WORKGROUP) && (data.syncScope > SCOPE_WORKGROUP))
    {
        return SCOPE_WORKGROUP;
    }
    return data.syncScope;
}

std::vector<VarDef> createVars(const CaseDef &data)
{
    const MemQualifier qualifier = getMemQualifier(data);
    return std::vector<VarDef>{{"payload", qualifier, data.payloadSC, data.dataType, data.payloadMemLocal},
                               {"fail", MEM_NONE, SC_BUFFER, DATA_TYPE_UINT, true}};
}

class MixedAccessTestCase : public MemoryModelTestCase
{
public:
    MixedAccessTestCase(tcu::TestContext &context, const char *name, const CaseDef &data);
    void checkSupport(Context &context) const override;

private:
    std::string declareSpvIntrinsics() const override;
    std::string payloadLoad(const char *coord) const;
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
    if (m_data.core11 || m_data.payloadSC != SC_WORKGROUP)
        return {};

    std::ostringstream css;
    css << "#extension GL_EXT_spirv_intrinsics : enable\n";
    css << declareVarSpvIntrinsics("payload", m_data.visScope == SCOPE_DEVICE);
    return css.str();
}

// By default, compiler sets workgroup scope visibility for shared memory variables.
// Force precise semantics with SPIR-V intrinsics.
std::string MixedAccessTestCase::payloadLoad(const char *coord) const
{
    if (m_data.core11 || m_data.payloadSC != SC_WORKGROUP)
        return gl_load("payload", coord);

    if (!m_data.visScope.has_value())
        return spv_load("payload", coord);

    return spv_load("payload", coord, *m_data.visScope);
}

std::string MixedAccessTestCase::initProgram() const
{
    const DataType type   = getVar("payload").type;
    const char *typeStr   = gl_type(type);
    const Scope syncScope = m_data.syncScope;

    std::ostringstream css;

    css << initPartnerCoordinates(getMappingScope(m_data));

    css << "   uint value = invocationId;\n"
        << "   " << typeStr << " r1 = 0;\n"
        << "   " << typeStr << " r2 = 0;\n"
        << "   " << typeStr << " expected = " << typeStr << "(invocationIdPartner);\n"
        << "   value += " << gl_toUint(type, payloadLoad("payloadCoordPartner")) << " >> 31;\n"
        << "   " << gl_atomicStore("payload", "payloadCoord", "value", syncScope, 0) << ";\n"
        << "   r1 = " << gl_atomicLoad("payload", "payloadCoordPartner", syncScope, 0) << ";\n"
        << "   r2 = " << payloadLoad("payloadCoordPartner") << ";\n"
        << "   if (!idle && r1 == expected && r2 != expected) {\n"
        << "      " << gl_store("fail", "failCoord", "1") << ";\n"
        << "   }\n";

    return css.str();
}

} // namespace

tcu::TestCaseGroup *createMixedAccessTests(tcu::TestContext &testCtx)
{
    static const NamedValue<bool> core11Cases[] = {
        {true, "core11"}, // supported by Vulkan 1.1
        {false, "ext"},   // requires Vulkan memory model
    };

    static const NamedValue<DataType> typeCases[] = {
        {DATA_TYPE_UINT, "u32"},
        {DATA_TYPE_UINT64, "u64"},
        {DATA_TYPE_FLOAT32, "f32"},
        {DATA_TYPE_FLOAT64, "f64"},
    };

    static const NamedValue<StorageClass> scCases[] = {
        {SC_BUFFER, "buffer"},
        {SC_IMAGE, "image"},
        {SC_WORKGROUP, "workgroup"},
        {SC_PHYSBUFFER, "physbuffer"},
    };

    static const NamedValue<bool> localCases[] = {
        {false, "nonlocal"},
        {true, "local"},
    };

    static const NamedValue<Scope> syncScopeCases[] = {
        {SCOPE_DEVICE, "device"},
        {SCOPE_QUEUEFAMILY, "queuefamily"},
        {SCOPE_WORKGROUP, "workgroup"},
        {SCOPE_SUBGROUP, "subgroup"},
    };

    static const NamedValue<std::optional<Scope>> visScopeCases[] = {
        {std::nullopt, "nonprivate"},
        {SCOPE_SUBGROUP, "vissubgroup"},
        {SCOPE_WORKGROUP, "visworkgroup"},
        {SCOPE_QUEUEFAMILY, "visqueuefamily"},
        {SCOPE_DEVICE, "visdevice"},
    };

    static const NamedValue<Stage> stageCases[] = {
        {STAGE_COMPUTE, "comp"},
        {STAGE_VERTEX, "vert"},
        {STAGE_FRAGMENT, "frag"},
    };

    de::MovePtr<tcu::TestCaseGroup> group(new tcu::TestCaseGroup(testCtx, "mixed_access"));

    const auto groupName = [](const char *prefix, const char *name) { return std::string(prefix) + "_" + name; };

    for (const auto &[core11, type, local, sc, syncScope, visScope, stage] :
         Combinations(core11Cases, typeCases, localCases, scCases, syncScopeCases, visScopeCases, stageCases))
    {
        const CaseDef c = {
            core11.value,    // bool core11;
            local.value,     // bool payloadMemLocal;
            type.value,      // DataType dataType;
            sc.value,        // StorageClass payloadSC;
            syncScope.value, // Scope syncScope;
            visScope.value,  // std::optional<Scope> visScope;
            stage.value,     // Stage stage;
        };

        // Tests in core11 include only variations that do not require Vulkan memory model. Skip explicit visibility
        // scope. Skip queue family scope which doesn't exist in Vulkan 1.1. Skip physical storage buffer which doesn't
        // support coherent decoration.
        if (c.core11 && (c.visScope || c.dataType == DATA_TYPE_UINT64 || c.dataType == DATA_TYPE_FLOAT64 ||
                         c.syncScope == SCOPE_QUEUEFAMILY || c.payloadSC == SC_PHYSBUFFER))
        {
            continue;
        }

        // Workgroup memory is only tested as non-local
        if (c.payloadSC == SC_WORKGROUP && c.payloadMemLocal)
        {
            continue;
        }

        // Skip workgroup scope for non-compute stages
        if (c.stage != STAGE_COMPUTE && (c.syncScope == SCOPE_WORKGROUP || c.visScope == SCOPE_WORKGROUP))
        {
            continue;
        }

        // Skip workgroup storage class for non-compute stages
        if (c.stage != STAGE_COMPUTE && c.payloadSC == SC_WORKGROUP)
        {
            continue;
        }

        // No 64-bit image types, so skip tests with payload in image memory
        if (c.payloadSC == SC_IMAGE && is64BitType(c.dataType))
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
