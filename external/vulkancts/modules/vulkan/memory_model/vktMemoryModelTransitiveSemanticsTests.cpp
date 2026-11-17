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
 * \brief Memory Model Transitive Semantics Tests
 *//*--------------------------------------------------------------------*/

#include "vktMemoryModelTransitiveSemanticsTests.hpp"
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
    bool core11;
    bool payloadMemLocal;
    bool guard1MemLocal;
    bool guard2MemLocal;
    StorageClass payloadSC;
    StorageClass guard1SC;
    StorageClass guard2SC;
    Scope scope;
};

std::vector<VarDef> createVars(const CaseDef &data)
{
    const MemQualifier payloadQualifier = data.core11 ? MEM_COHERENT : scopeToQualifier(data.scope);
    const MemQualifier guard1Qualifier  = (data.core11 && data.guard1SC == SC_IMAGE) ? MEM_COHERENT : MEM_NONE;
    const MemQualifier guard2Qualifier  = (data.core11 && data.guard2SC == SC_IMAGE) ? MEM_COHERENT : MEM_NONE;
    return std::vector<VarDef>{{"payload", payloadQualifier, data.payloadSC, DATA_TYPE_UINT, data.payloadMemLocal},
                               {"guard1", guard1Qualifier, data.guard1SC, DATA_TYPE_UINT, data.guard1MemLocal},
                               {"guard2", guard2Qualifier, data.guard2SC, DATA_TYPE_UINT, data.guard2MemLocal},
                               {"fail", MEM_NONE, SC_BUFFER, DATA_TYPE_UINT, true}};
}

class TransitiveSemanticsTestCase : public MemoryModelTestCase
{
public:
    TransitiveSemanticsTestCase(tcu::TestContext &context, const char *name, const CaseDef &data);
    void checkSupport(Context &context) const override;

protected:
    const CaseDef m_data;
};

TransitiveSemanticsTestCase::TransitiveSemanticsTestCase(tcu::TestContext &context, const char *name,
                                                         const CaseDef &data)
    : MemoryModelTestCase(context, name, {STAGE_COMPUTE, data.core11}, createVars(data))
    , m_data(data)
{
}

void TransitiveSemanticsTestCase::checkSupport(Context &context) const
{
    if (!context.contextSupports(vk::ApiVersion(0, 1, 1, 0)))
        TCU_THROW(NotSupportedError, "Vulkan 1.1 not supported");

    if (!m_data.core11)
    {
        if (!context.getVulkanMemoryModelFeatures().vulkanMemoryModel)
            TCU_THROW(NotSupportedError, "vulkanMemoryModel not supported");

        if (m_data.scope == SCOPE_DEVICE && !context.getVulkanMemoryModelFeatures().vulkanMemoryModelDeviceScope)
            TCU_THROW(NotSupportedError, "vulkanMemoryModelDeviceScope not supported");
    }

    if (m_data.scope == SCOPE_SUBGROUP)
        checkSupportSubgroupFeatures(context, 3);

    checkSupportVars(context);
    checkSupportVarAtomics(context, "guard1");
    checkSupportVarAtomics(context, "guard2");
}

class CombinedAtomicTestCase : public TransitiveSemanticsTestCase
{
public:
    using TransitiveSemanticsTestCase::TransitiveSemanticsTestCase;

private:
    std::string initProgram() const override;
};

std::string CombinedAtomicTestCase::initProgram() const
{
    const DataType type       = getVar("payload").type;
    const char *typeStr       = gl_type(type);
    const Scope scope         = m_data.scope;
    const uint32_t semPayload = getVarSemantics("payload");
    const uint32_t semGuard1  = getVarSemantics("guard1");
    const uint32_t semGuard2  = getVarSemantics("guard2");
    const uint32_t semRel1    = SEM_RELEASE | semPayload | semGuard1;
    const uint32_t semAcq1    = SEM_ACQUIRE | semPayload | semGuard1;
    const uint32_t semRel2    = SEM_RELEASE | semPayload | semGuard2;
    const uint32_t semAcq2    = SEM_ACQUIRE | semPayload | semGuard2;

    std::ostringstream css;

    css << initMultiPartnerCoordinates(m_data.scope, 2);

    css << "   bool skip1 = false;\n"
        << "   bool skip2 = false;\n"
        << "   uint value = invocationId;\n"
        << "   " << typeStr << " r = 0;\n"
        << "   " << typeStr << " expected = " << typeStr << "(invocationIdPartner1);\n"
        << "   value += " << gl_toUint(type, gl_load("payload", "payloadCoordPartner1")) << " >> 31;\n"
        << "   " << gl_store("payload", "payloadCoord", "value") << ";\n"
        << "   " << gl_atomicStore("guard1", "guard1CoordPartner1", "1u", scope, semRel1) << ";\n"
        << "   skip1 = " << gl_atomicLoad("guard1", "guard1Coord", scope, semAcq1) << " == 0;\n"
        << "   if (!skip1) { " << gl_atomicStore("guard2", "guard2CoordPartner1", "1u", scope, semRel2) << "; }\n"
        << "   skip2 = " << gl_atomicLoad("guard2", "guard2Coord", scope, semAcq2) << " == 0;\n"
        << "   r = " << gl_load("payload", "payloadCoordPartner1") << ";\n"
        << "   if (!idle && !skip2 && r != expected) {\n"
        << "      " << gl_store("fail", "failCoord", "1") << ";\n"
        << "   }\n";

    return css.str();
}

class CombinedFenceTestCase : public TransitiveSemanticsTestCase
{
public:
    using TransitiveSemanticsTestCase::TransitiveSemanticsTestCase;

private:
    std::string initProgram() const override;
};

std::string CombinedFenceTestCase::initProgram() const
{
    const DataType type       = getVar("payload").type;
    const char *typeStr       = gl_type(type);
    const Scope scope         = m_data.scope;
    const uint32_t semPayload = getVarSemantics("payload");
    const uint32_t semGuard1  = getVarSemantics("guard1");
    const uint32_t semGuard2  = getVarSemantics("guard2");
    const uint32_t semRel1    = SEM_RELEASE | semPayload | semGuard1;
    const uint32_t semAcq1    = SEM_ACQUIRE | semPayload | semGuard1;
    const uint32_t semRel2    = SEM_RELEASE | semPayload | semGuard2;
    const uint32_t semAcq2    = SEM_ACQUIRE | semPayload | semGuard2;

    std::ostringstream css;

    css << initMultiPartnerCoordinates(m_data.scope, 2);

    css << "   bool skip1 = false;\n"
        << "   bool skip2 = false;\n"
        << "   uint value = invocationId;\n"
        << "   " << typeStr << " r = 0;\n"
        << "   " << typeStr << " expected = " << typeStr << "(invocationIdPartner1);\n"
        << "   value += " << gl_toUint(type, gl_load("payload", "payloadCoordPartner1")) << " >> 31;\n"
        << "   " << gl_store("payload", "payloadCoord", "value") << ";\n"
        << "   " << gl_memoryBarrier(scope, semRel1) << ";\n"
        << "   " << gl_atomicStore("guard1", "guard1CoordPartner1", "1u", scope, 0) << ";\n"
        << "   skip1 = " << gl_atomicLoad("guard1", "guard1Coord", scope, 0) << " == 0;\n"
        << "   " << gl_memoryBarrier(scope, semAcq1) << ";\n"
        << "   if (!skip1) {\n"
        << "      " << gl_memoryBarrier(scope, semRel2) << ";\n"
        << "      " << gl_atomicStore("guard2", "guard2CoordPartner1", "1u", scope, 0) << ";\n"
        << "   }\n"
        << "   skip2 = " << gl_atomicLoad("guard2", "guard2Coord", scope, 0) << " == 0;\n"
        << "   " << gl_memoryBarrier(scope, semAcq2) << ";\n"
        << "   r = " << gl_load("payload", "payloadCoordPartner1") << ";\n"
        << "   if (!idle && !skip2 && r != expected) {\n"
        << "      " << gl_store("fail", "failCoord", "1") << ";\n"
        << "   }\n";

    return css.str();
}

// Similar to CombinedAtomicTestCase, except that each invocation has a dedicated role in a group.
// The middle thread never accesses payload's storage class and its atomics are in different storage classes.
class RelayAtomicTestCase : public TransitiveSemanticsTestCase
{
public:
    using TransitiveSemanticsTestCase::TransitiveSemanticsTestCase;

private:
    std::string initProgram() const override;
};

std::string RelayAtomicTestCase::initProgram() const
{
    const DataType type       = getVar("payload").type;
    const char *typeStr       = gl_type(type);
    const Scope scope         = m_data.scope;
    const uint32_t semPayload = getVarSemantics("payload");
    const uint32_t semGuard1  = getVarSemantics("guard1");
    const uint32_t semGuard2  = getVarSemantics("guard2");
    const uint32_t semRel1    = SEM_RELEASE | semPayload | semGuard1;
    const uint32_t semAcq1    = SEM_ACQUIRE | semPayload | semGuard1;
    const uint32_t semRel2    = SEM_RELEASE | semPayload | semGuard2;
    const uint32_t semAcq2    = SEM_ACQUIRE | semPayload | semGuard2;

    std::ostringstream css;

    css << initMultiPartnerCoordinates(m_data.scope, 2);

    css << "   bool skip1 = false;\n"
        << "   bool skip2 = false;\n"
        << "   uint value = invocationId;\n"
        << "   " << typeStr << " r = 0;\n"
        << "   " << typeStr << " expected = " << typeStr << "(invocationIdPartner1);\n"
        << "   uint roleId = (index / stride + index % stride) % 3u;\n"
        << "   if (roleId == 0) {\n"
        << "      " << gl_store("payload", "payloadCoord", "value") << ";\n"
        << "      " << gl_atomicStore("guard1", "guard1CoordPartner1", "1u", scope, semRel1) << ";\n"
        << "   }\n"
        << "   else if (roleId == 1) {\n"
        << "      skip1 = " << gl_atomicLoad("guard1", "guard1Coord", scope, semAcq1) << " == 0;\n"
        << "      if (!skip1) { " << gl_atomicStore("guard2", "guard2CoordPartner1", "1u", scope, semRel2) << "; }\n"
        << "   }\n"
        << "   else {\n"
        << "      expected += " << gl_toUint(type, gl_load("payload", "payloadCoordPartner1")) << " >> 31;\n"
        << "      skip2 = " << gl_atomicLoad("guard2", "guard2Coord", scope, semAcq2) << " == 0;\n"
        << "      r = " << gl_load("payload", "payloadCoordPartner1") << ";\n"
        << "      if (!idle && !skip2 && r != expected) {\n"
        << "         " << gl_store("fail", "failCoord", "1") << ";\n"
        << "      }\n"
        << "   }\n";

    return css.str();
}

// Similar to CombinedFenceTestCase, except that each invocation has a dedicated role in a group.
// The middle thread never accesses payload's storage class and its atomics are in different storage classes.
class RelayFenceTestCase : public TransitiveSemanticsTestCase
{
public:
    using TransitiveSemanticsTestCase::TransitiveSemanticsTestCase;

private:
    std::string initProgram() const override;
};

std::string RelayFenceTestCase::initProgram() const
{
    const DataType type       = getVar("payload").type;
    const char *typeStr       = gl_type(type);
    const Scope scope         = m_data.scope;
    const uint32_t semPayload = getVarSemantics("payload");
    const uint32_t semGuard1  = getVarSemantics("guard1");
    const uint32_t semGuard2  = getVarSemantics("guard2");
    const uint32_t semRel1    = SEM_RELEASE | semPayload | semGuard1;
    const uint32_t semAcq1    = SEM_ACQUIRE | semPayload | semGuard1;
    const uint32_t semRel2    = SEM_RELEASE | semPayload | semGuard2;
    const uint32_t semAcq2    = SEM_ACQUIRE | semPayload | semGuard2;

    std::ostringstream css;

    css << initMultiPartnerCoordinates(m_data.scope, 2);

    css << "   bool skip1 = false;\n"
        << "   bool skip2 = false;\n"
        << "   uint value = invocationId;\n"
        << "   " << typeStr << " r = 0;\n"
        << "   " << typeStr << " expected = " << typeStr << "(invocationIdPartner1);\n"
        << "   uint roleId = (index / stride + index % stride) % 3u;\n"
        << "   if (roleId == 0) {\n"
        << "      " << gl_store("payload", "payloadCoord", "value") << ";\n"
        << "      " << gl_memoryBarrier(scope, semRel1) << ";\n"
        << "      " << gl_atomicStore("guard1", "guard1CoordPartner1", "1u", scope, 0) << ";\n"
        << "   }\n"
        << "   else if (roleId == 1) {\n"
        << "      skip1 = " << gl_atomicLoad("guard1", "guard1Coord", scope, 0) << " == 0;\n"
        << "      " << gl_memoryBarrier(scope, semAcq1) << ";\n"
        << "      if (!skip1) {\n"
        << "         " << gl_memoryBarrier(scope, semRel2) << ";\n"
        << "         " << gl_atomicStore("guard2", "guard2CoordPartner1", "1u", scope, 0) << ";\n"
        << "      }\n"
        << "   }\n"
        << "   else {\n"
        << "      expected += " << gl_toUint(type, gl_load("payload", "payloadCoordPartner1")) << " >> 31;\n"
        << "      skip2 = " << gl_atomicLoad("guard2", "guard2Coord", scope, 0) << " == 0;\n"
        << "      " << gl_memoryBarrier(scope, semAcq2) << ";\n"
        << "      r = " << gl_load("payload", "payloadCoordPartner1") << ";\n"
        << "      if (!idle && !skip2 && r != expected) {\n"
        << "         " << gl_store("fail", "failCoord", "1") << ";\n"
        << "      }\n"
        << "   }\n";

    return css.str();
}

template <typename VariantTestCase>
void addTransitiveSemanticsTests(tcu::TestCaseGroup &root, const char *variantName, const char *syncName)
{
    static const NamedValue<bool> core11Cases[] = {
        {true, "core11"}, // supported by Vulkan 1.1
        {false, "ext"},   // requires Vulkan memory model
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

    static const NamedValue<Scope> scopeCases[] = {
        {SCOPE_DEVICE, "device"},
        {SCOPE_QUEUEFAMILY, "queuefamily"},
        {SCOPE_WORKGROUP, "workgroup"},
        {SCOPE_SUBGROUP, "subgroup"},
    };

    const auto groupName = [](const char *prefix, const char *name) { return std::string(prefix) + "_" + name; };

    for (const auto &[core11, local0, sc0, local1, sc1, local2, sc2, scope] :
         Combinations(core11Cases, localCases, scCases, localCases, scCases, localCases, scCases, scopeCases))
    {
        const CaseDef c = {
            core11.value, // bool core11;
            local0.value, // bool payloadMemLocal;
            local1.value, // bool guard1MemLocal;
            local2.value, // bool guard2MemLocal;
            sc0.value,    // StorageClass payloadSC;
            sc1.value,    // StorageClass guard1SC;
            sc2.value,    // StorageClass guard2SC;
            scope.value,  // Scope scope;
        };

        // Tests in core11 include only variations that do not require Vulkan memory model. Skip release-acquire
        // atomics. Skip queue family scope which doesn't exist in Vulkan 1.1. Skip physical storage buffer which
        // doesn't support coherent decoration.
        if (c.core11 && (std::string(syncName) == "atomic" || c.scope == SCOPE_QUEUEFAMILY ||
                         c.payloadSC == SC_PHYSBUFFER || c.guard1SC == SC_PHYSBUFFER || c.guard2SC == SC_PHYSBUFFER))
        {
            continue;
        }

        // Skip tests where a variable is in workgroup memory, but its accesses cross larger scopes
        if (c.scope > SCOPE_WORKGROUP &&
            (c.payloadSC == SC_WORKGROUP || c.guard1SC == SC_WORKGROUP || c.guard2SC == SC_WORKGROUP))
        {
            continue;
        }

        // Workgroup memory is only tested as non-local
        if ((c.payloadMemLocal && c.payloadSC == SC_WORKGROUP) || (c.guard1MemLocal && c.guard1SC == SC_WORKGROUP) ||
            (c.guard2MemLocal && c.guard2SC == SC_WORKGROUP))
        {
            continue;
        }

        // Limit workgroup and subgroup scoped tests to variations with different semantics for each variable
        if (c.scope <= SCOPE_WORKGROUP &&
            (scToSem(c.guard1SC) == scToSem(c.payloadSC) || scToSem(c.guard2SC) == scToSem(c.payloadSC) ||
             scToSem(c.guard2SC) == scToSem(c.guard1SC)))
        {
            continue;
        }

        // Relay tests require a variable in workgroup memory, so restrict them to subgroup and workgroup scopes
        if (std::string(variantName) == "relay" && c.scope > SCOPE_WORKGROUP)
        {
            continue;
        }

        addTestCase(root,
                    {variantName, core11.name, syncName, groupName("payload", local0.name), sc0.name,
                     groupName("guard1", local1.name), sc1.name, groupName("guard2", local2.name), sc2.name},
                    new VariantTestCase(root.getTestContext(), scope.name, c));
    }
}

} // namespace

tcu::TestCaseGroup *createTransitiveSemanticsTests(tcu::TestContext &testCtx)
{
    de::MovePtr<tcu::TestCaseGroup> group(new tcu::TestCaseGroup(testCtx, "transitive_semantics"));

    addTransitiveSemanticsTests<CombinedAtomicTestCase>(*group, "combined", "atomic");
    addTransitiveSemanticsTests<CombinedFenceTestCase>(*group, "combined", "fence");
    addTransitiveSemanticsTests<RelayAtomicTestCase>(*group, "relay", "atomic");
    addTransitiveSemanticsTests<RelayFenceTestCase>(*group, "relay", "fence");

    return group.release();
}

} // namespace vkt::MemoryModel
