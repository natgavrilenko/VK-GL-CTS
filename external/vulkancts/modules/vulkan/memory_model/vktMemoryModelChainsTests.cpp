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
 * \brief Memory Model Chains Tests
 *//*--------------------------------------------------------------------*/

#include "vktMemoryModelChainsTests.hpp"
#include "vktMemoryModelTestCase.hpp"
#include "vktMemoryModelUtil.hpp"

#include "deUniquePtr.hpp"
#include "tcuTestCase.hpp"
#include "vktTestCase.hpp"

#include <sstream>
#include <string>
#include <vector>

namespace vkt::MemoryModel
{
namespace
{

struct CaseDef
{
    bool payloadMemLocal;
    bool guard1MemLocal;
    bool guard2MemLocal;
    StorageClass payloadSC;
    StorageClass guard1SC;
    StorageClass guard2SC;
    Scope nearScope;
    Scope farScope;
    Stage stage;
};

std::vector<VarDef> createVars(const CaseDef &data)
{
    return std::vector<VarDef>{
        {"payload", MEM_NONPRIVATE, data.payloadSC, DATA_TYPE_UINT, data.payloadMemLocal},
        {"guard1", MEM_NONE, data.guard1SC, DATA_TYPE_UINT, data.guard1MemLocal},
        {"guard2", MEM_NONE, data.guard2SC, DATA_TYPE_UINT, data.guard2MemLocal},
        {"fail", MEM_NONE, SC_BUFFER, DATA_TYPE_UINT, true},
    };
}

class ChainsTestCase : public MemoryModelTestCase
{
public:
    ChainsTestCase(tcu::TestContext &context, const char *name, const CaseDef &data);
    void checkSupport(Context &context) const override;

protected:
    std::string declareSpvIntrinsics() const override;
    std::string payloadStore(const char *coord, const char *value) const;
    std::string payloadLoad(const char *coord) const;
    const CaseDef m_data;
};

ChainsTestCase::ChainsTestCase(tcu::TestContext &context, const char *name, const CaseDef &data)
    : MemoryModelTestCase(context, name, {data.stage, false}, createVars(data))
    , m_data(data)
{
}

void ChainsTestCase::checkSupport(Context &context) const
{
    if (!context.contextSupports(vk::ApiVersion(0, 1, 1, 0)))
        TCU_THROW(NotSupportedError, "Vulkan 1.1 not supported");

    if (!context.getVulkanMemoryModelFeatures().vulkanMemoryModel)
        TCU_THROW(NotSupportedError, "vulkanMemoryModel not supported");

    if (!context.getVulkanMemoryModelFeatures().vulkanMemoryModelAvailabilityVisibilityChains)
        TCU_THROW(NotSupportedError, "vulkanMemoryModelAvailabilityVisibilityChains not supported");

    if (m_data.farScope == SCOPE_DEVICE && !context.getVulkanMemoryModelFeatures().vulkanMemoryModelDeviceScope)
        TCU_THROW(NotSupportedError, "vulkanMemoryModelDeviceScope not supported");

    if (m_data.nearScope == SCOPE_SUBGROUP)
        checkSupportSubgroupFeatures(context);

    checkSupportVars(context);
    checkSupportVarAtomics(context, "guard1");
    checkSupportVarAtomics(context, "guard2");
    checkSupportStageAtomics(context);
}

std::string ChainsTestCase::declareSpvIntrinsics() const
{
    if (m_data.payloadSC != SC_WORKGROUP)
        return {};

    std::ostringstream css;
    css << "#extension GL_EXT_spirv_intrinsics : enable\n";
    css << declareVarSpvIntrinsics("payload", false);
    return css.str();
}

std::string ChainsTestCase::payloadStore(const char *coord, const char *value) const
{
    if (m_data.payloadSC != SC_WORKGROUP)
        return gl_store("payload", coord, value);

    // Precise semantics instead of workgroup scope availability by compiler
    return spv_store("payload", coord, value);
}

std::string ChainsTestCase::payloadLoad(const char *coord) const
{
    if (m_data.payloadSC != SC_WORKGROUP)
        return gl_load("payload", coord);

    // Precise semantics instead of workgroup scope visibility by compiler
    return spv_load("payload", coord);
}

class AvailabilityChainsAtomicTestCase : public ChainsTestCase
{
public:
    using ChainsTestCase::ChainsTestCase;
    static constexpr bool nearFirst = true;

private:
    std::string initProgram() const override;
};

std::string AvailabilityChainsAtomicTestCase::initProgram() const
{
    const Scope nearScope     = m_data.nearScope;
    const Scope farScope      = m_data.farScope;
    const DataType type       = getVar("payload").type;
    const char *typeStr       = gl_type(type);
    const uint32_t semPayload = getVarSemantics("payload");
    const uint32_t semRelNear = SEM_RELEASE | semPayload | SEM_AVAILABLE;
    const uint32_t semAcqNear = SEM_ACQUIRE | semPayload;
    const uint32_t semRelFar  = SEM_RELEASE | semPayload | SEM_AVAILABLE;
    const uint32_t semAcqFar  = SEM_ACQUIRE | semPayload | SEM_VISIBLE;

    std::ostringstream css;

    css << initPartnerCoordinates(nearScope, "Near");
    css << initPartnerCoordinates(farScope, "Far");

    css << "   bool skip1 = false;\n"
        << "   bool skip2 = false;\n"
        << "   uint value = invocationIdPartnerNear;\n"
        << "   " << typeStr << " r = 0;\n"
        << "   " << typeStr << " expected = " << typeStr << "(invocationIdPartnerFar);\n"
        << "   value += " << gl_toUint(type, payloadLoad("payloadCoordPartnerFar")) << " >> 31;\n"
        << "   " << payloadStore("payloadCoordPartnerNear", "value") << ";\n"
        << "   " << gl_atomicStore("guard1", "guard1CoordPartnerNear", "1u", nearScope, semRelNear) << ";\n"
        << "   skip1 = " << gl_atomicLoad("guard1", "guard1Coord", nearScope, semAcqNear) << " == 0;\n"
        << "   if (!skip1) {\n"
        << "      " << gl_atomicStore("guard2", "guard2CoordPartnerFar", "1u", farScope, semRelFar) << ";\n"
        << "   }\n"
        << "   skip2 = " << gl_atomicLoad("guard2", "guard2Coord", farScope, semAcqFar) << " == 0;\n"
        << "   r = " << payloadLoad("payloadCoordPartnerFar") << ";\n"
        << "   if (!idleNear && !idleFar && !skip2 && r != expected) {\n"
        << "      " << gl_store("fail", "failCoord", "1") << ";\n"
        << "   }\n";

    return css.str();
}

class AvailabilityChainsFenceTestCase : public ChainsTestCase
{
public:
    using ChainsTestCase::ChainsTestCase;
    static constexpr bool nearFirst = true;

private:
    std::string initProgram() const override;
};

std::string AvailabilityChainsFenceTestCase::initProgram() const
{
    const DataType type       = getVar("payload").type;
    const char *typeStr       = gl_type(type);
    const Scope nearScope     = m_data.nearScope;
    const Scope farScope      = m_data.farScope;
    const uint32_t semPayload = getVarSemantics("payload");
    const uint32_t semGuard1  = getVarSemantics("guard1");
    const uint32_t semGuard2  = getVarSemantics("guard2");
    const uint32_t semRelNear = SEM_RELEASE | semPayload | semGuard1 | SEM_AVAILABLE;
    const uint32_t semAcqNear = SEM_ACQUIRE | semPayload | semGuard1;
    const uint32_t semRelFar  = SEM_RELEASE | semPayload | semGuard2 | SEM_AVAILABLE;
    const uint32_t semAcqFar  = SEM_ACQUIRE | semPayload | semGuard2 | SEM_VISIBLE;

    std::ostringstream css;

    css << initPartnerCoordinates(nearScope, "Near");
    css << initPartnerCoordinates(farScope, "Far");

    css << "   bool skip1 = false;\n"
        << "   bool skip2 = false;\n"
        << "   uint value = invocationIdPartnerNear;\n"
        << "   " << typeStr << " r = 0;\n"
        << "   " << typeStr << " expected = " << typeStr << "(invocationIdPartnerFar);\n"
        << "   value += " << gl_toUint(type, payloadLoad("payloadCoordPartnerFar")) << " >> 31;\n"
        << "   " << payloadStore("payloadCoordPartnerNear", "value") << ";\n"
        << "   " << gl_memoryBarrier(nearScope, semRelNear) << ";\n"
        << "   " << gl_atomicStore("guard1", "guard1CoordPartnerNear", "1u", nearScope, 0) << ";\n"
        << "   skip1 = " << gl_atomicLoad("guard1", "guard1Coord", nearScope, 0) << " == 0;\n"
        << "   " << gl_memoryBarrier(nearScope, semAcqNear) << ";\n"
        << "   if (!skip1) {\n"
        << "      " << gl_memoryBarrier(farScope, semRelFar) << ";\n"
        << "      " << gl_atomicStore("guard2", "guard2CoordPartnerFar", "1u", farScope, 0) << ";\n"
        << "   }\n"
        << "   skip2 = " << gl_atomicLoad("guard2", "guard2Coord", farScope, 0) << " == 0;\n"
        << "   " << gl_memoryBarrier(farScope, semAcqFar) << ";\n"
        << "   r = " << payloadLoad("payloadCoordPartnerFar") << ";\n"
        << "   if (!idleNear && !idleFar && !skip2 && r != expected) {\n"
        << "      " << gl_store("fail", "failCoord", "1") << ";\n"
        << "   }\n";

    return css.str();
}

class VisibilityChainsAtomicTestCase : public ChainsTestCase
{
public:
    using ChainsTestCase::ChainsTestCase;
    static constexpr bool nearFirst = false;

private:
    std::string initProgram() const override;
};

std::string VisibilityChainsAtomicTestCase::initProgram() const
{
    const Scope nearScope     = m_data.nearScope;
    const Scope farScope      = m_data.farScope;
    const DataType type       = getVar("payload").type;
    const char *typeStr       = gl_type(type);
    const uint32_t semPayload = getVarSemantics("payload");
    const uint32_t semRelFar  = SEM_RELEASE | semPayload | SEM_AVAILABLE;
    const uint32_t semAcqFar  = SEM_ACQUIRE | semPayload | SEM_VISIBLE;
    const uint32_t semRelNear = SEM_RELEASE | semPayload;
    const uint32_t semAcqNear = SEM_ACQUIRE | semPayload | SEM_VISIBLE;

    std::ostringstream css;

    css << initPartnerCoordinates(nearScope, "Near");
    css << initPartnerCoordinates(farScope, "Far");

    css << "   bool skip1 = false;\n"
        << "   bool skip2 = false;\n"
        << "   uint value = invocationIdPartnerFar;\n"
        << "   " << typeStr << " r = 0;\n"
        << "   " << typeStr << " expected = " << typeStr << "(invocationIdPartnerNear);\n"
        << "   value += " << gl_toUint(type, payloadLoad("payloadCoordPartnerNear")) << " >> 31;\n"
        << "   " << payloadStore("payloadCoordPartnerFar", "value") << ";\n"
        << "   " << gl_atomicStore("guard1", "guard1CoordPartnerFar", "1u", farScope, semRelFar) << ";\n"
        << "   skip1 = " << gl_atomicLoad("guard1", "guard1Coord", farScope, semAcqFar) << " == 0;\n"
        << "   if (!skip1) {\n"
        << "      " << gl_atomicStore("guard2", "guard2CoordPartnerNear", "1u", nearScope, semRelNear) << ";\n"
        << "   }\n"
        << "   skip2 = " << gl_atomicLoad("guard2", "guard2Coord", nearScope, semAcqNear) << " == 0;\n"
        << "   r = " << payloadLoad("payloadCoordPartnerNear") << ";\n"
        << "   if (!idleNear && !idleFar && !skip2 && r != expected) {\n"
        << "      " << gl_store("fail", "failCoord", "1") << ";\n"
        << "   }\n";

    return css.str();
}

class VisibilityChainsFenceTestCase : public ChainsTestCase
{
public:
    using ChainsTestCase::ChainsTestCase;
    static constexpr bool nearFirst = false;

private:
    std::string initProgram() const override;
};

std::string VisibilityChainsFenceTestCase::initProgram() const
{
    const Scope nearScope     = m_data.nearScope;
    const Scope farScope      = m_data.farScope;
    const DataType type       = getVar("payload").type;
    const char *typeStr       = gl_type(type);
    const uint32_t semPayload = getVarSemantics("payload");
    const uint32_t semGuard1  = getVarSemantics("guard1");
    const uint32_t semGuard2  = getVarSemantics("guard2");
    const uint32_t semRelFar  = SEM_RELEASE | semPayload | semGuard1 | SEM_AVAILABLE;
    const uint32_t semAcqFar  = SEM_ACQUIRE | semPayload | semGuard1 | SEM_VISIBLE;
    const uint32_t semRelNear = SEM_RELEASE | semPayload | semGuard2;
    const uint32_t semAcqNear = SEM_ACQUIRE | semPayload | semGuard2 | SEM_VISIBLE;

    std::ostringstream css;

    css << initPartnerCoordinates(nearScope, "Near");
    css << initPartnerCoordinates(farScope, "Far");

    css << "   bool skip1 = false;\n"
        << "   bool skip2 = false;\n"
        << "   uint value = invocationIdPartnerFar;\n"
        << "   " << typeStr << " r = 0;\n"
        << "   " << typeStr << " expected = " << typeStr << "(invocationIdPartnerNear);\n"
        << "   value += " << gl_toUint(type, payloadLoad("payloadCoordPartnerNear")) << " >> 31;\n"
        << "   " << payloadStore("payloadCoordPartnerFar", "value") << ";\n"
        << "   " << gl_memoryBarrier(farScope, semRelFar) << ";\n"
        << "   " << gl_atomicStore("guard1", "guard1CoordPartnerFar", "1u", farScope, 0) << ";\n"
        << "   skip1 = " << gl_atomicLoad("guard1", "guard1Coord", farScope, 0) << " == 0;\n"
        << "   " << gl_memoryBarrier(farScope, semAcqFar) << ";\n"
        << "   if (!skip1) {\n"
        << "      " << gl_memoryBarrier(nearScope, semRelNear) << ";\n"
        << "      " << gl_atomicStore("guard2", "guard2CoordPartnerNear", "1u", nearScope, 0) << ";\n"
        << "   }\n"
        << "   skip2 = " << gl_atomicLoad("guard2", "guard2Coord", nearScope, 0) << " == 0;\n"
        << "   " << gl_memoryBarrier(nearScope, semAcqNear) << ";\n"
        << "   r = " << payloadLoad("payloadCoordPartnerNear") << ";\n"
        << "   if (!idleNear && !idleFar && !skip2 && r != expected) {\n"
        << "      " << gl_store("fail", "failCoord", "1") << ";\n"
        << "   }\n";

    return css.str();
}

template <typename VariantTestCase>
void addChainsTests(tcu::TestCaseGroup &root, const char *variantName, const char *syncName)
{
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

    static const NamedValue<Scope> scopeCases[] = {
        {SCOPE_DEVICE, "device"},
        {SCOPE_QUEUEFAMILY, "queuefamily"},
        {SCOPE_WORKGROUP, "workgroup"},
        {SCOPE_SUBGROUP, "subgroup"},
    };

    static const NamedValue<Stage> stageCases[] = {
        {STAGE_COMPUTE, "comp"},
        {STAGE_VERTEX, "vert"},
        {STAGE_FRAGMENT, "frag"},
    };

    const auto groupName = [](const char *prefix, const char *name) { return std::string(prefix) + "_" + name; };

    for (const auto &[local0, sc0, local1, sc1, local2, sc2, scope0, scope1, stage] : Combinations(
             localCases, scCases, localCases, scCases, localCases, scCases, scopeCases, scopeCases, stageCases))
    {
        const CaseDef c = {
            local0.value, // bool payloadMemLocal;
            local1.value, // bool guard1MemLocal;
            local2.value, // bool guard2MemLocal;
            sc0.value,    // StorageClass payloadSC;
            sc1.value,    // StorageClass guard1SC;
            sc2.value,    // StorageClass guard2SC;
            scope0.value, // Scope nearScope;
            scope1.value, // Scope farScope;
            stage.value,  // Stage stage;
        };

        // Near scope must be narrower than far scope, skip queue family nearScope because tests run in the same queue
        if (c.nearScope >= c.farScope || c.nearScope == SCOPE_QUEUEFAMILY)
        {
            continue;
        }

        // Skip tests where a variable is in workgroup memory, but its accesses cross larger scopes
        const Scope guard1Scope = VariantTestCase::nearFirst ? c.nearScope : c.farScope;
        const Scope guard2Scope = VariantTestCase::nearFirst ? c.farScope : c.nearScope;
        if ((c.payloadSC == SC_WORKGROUP && c.farScope > SCOPE_WORKGROUP) ||
            (c.guard1SC == SC_WORKGROUP && guard1Scope > SCOPE_WORKGROUP) ||
            (c.guard2SC == SC_WORKGROUP && guard2Scope > SCOPE_WORKGROUP))
        {
            continue;
        }

        // Workgroup memory is only tested as non-local
        if ((c.payloadMemLocal && c.payloadSC == SC_WORKGROUP) || (c.guard1MemLocal && c.guard1SC == SC_WORKGROUP) ||
            (c.guard2MemLocal && c.guard2SC == SC_WORKGROUP))
        {
            continue;
        }

        // Skip workgroup scope for non-compute stages
        if (c.stage != STAGE_COMPUTE && (c.nearScope == SCOPE_WORKGROUP || c.farScope == SCOPE_WORKGROUP))
        {
            continue;
        }

        // Skip workgroup storage class for non-compute stages
        if (c.stage != STAGE_COMPUTE &&
            (c.payloadSC == SC_WORKGROUP || c.guard1SC == SC_WORKGROUP || c.guard2SC == SC_WORKGROUP))
        {
            continue;
        }

        addTestCase(root,
                    {variantName, syncName, groupName("payload", local0.name), sc0.name,
                     groupName("guard1", local1.name), sc1.name, groupName("guard2", local2.name), sc2.name,
                     groupName("near", scope0.name), groupName("far", scope1.name)},
                    new VariantTestCase(root.getTestContext(), stage.name, c));
    }
}

} // namespace

tcu::TestCaseGroup *createChainsTests(tcu::TestContext &testCtx)
{
    de::MovePtr<tcu::TestCaseGroup> group(new tcu::TestCaseGroup(testCtx, "chains"));

    addChainsTests<AvailabilityChainsAtomicTestCase>(*group, "availability", "atomic");
    addChainsTests<AvailabilityChainsFenceTestCase>(*group, "availability", "fence");
    addChainsTests<VisibilityChainsAtomicTestCase>(*group, "visibility", "atomic");
    addChainsTests<VisibilityChainsFenceTestCase>(*group, "visibility", "fence");

    return group.release();
}

} // namespace vkt::MemoryModel
