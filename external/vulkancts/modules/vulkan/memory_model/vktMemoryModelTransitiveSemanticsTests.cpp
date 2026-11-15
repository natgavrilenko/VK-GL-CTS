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
    DataType payloadType;
    DataType guard1Type;
    DataType guard2Type;
    Scope scope;
};

std::vector<VarDef> createVars(const CaseDef &data)
{
    const MemQualifier qualifier = data.core11 ? MEM_COHERENT : scopeToQualifier(data.scope);
    return std::vector<VarDef>{{"payload", qualifier, data.payloadSC, data.payloadType, data.payloadMemLocal},
                               {"guard1", qualifier, data.guard1SC, data.guard1Type, data.guard1MemLocal},
                               {"guard2", qualifier, data.guard2SC, data.guard2Type, data.guard2MemLocal},
                               {"fail", MEM_NONE, SC_BUFFER, DATA_TYPE_UINT, true}};
}

// The tests check that release and acquire semantics are transitive: the payload is released through guard1, the
// invocation that acquires guard1 releases guard2, and the acquire of guard2 must make the payload visible. Payload,
// guard1 and guard2 use three different storage classes, and each access names only the payload's class and its own.
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

    if (!m_data.core11 && !context.getVulkanMemoryModelFeatures().vulkanMemoryModel)
        TCU_THROW(NotSupportedError, "vulkanMemoryModel not supported");

    if (m_data.scope == SCOPE_SUBGROUP)
        checkSupportSubgroupFeatures(context, 3);

    checkSupportVars(context);
    checkSupportVarAtomics(context, "guard1");
    checkSupportVarAtomics(context, "guard2");
}

// Every invocation plays all three roles, for different indexes.
// The atomic accesses to the guards have the release and acquire semantics.
class TransitiveSemanticsCombinedAtomicTestCase : public TransitiveSemanticsTestCase
{
public:
    using TransitiveSemanticsTestCase::TransitiveSemanticsTestCase;

private:
    std::string initProgram() const override;
};

std::string TransitiveSemanticsCombinedAtomicTestCase::initProgram() const
{
    const Scope scope   = m_data.scope;
    const DataType type = getVar("payload").type;
    const char *typeStr = gl_type(type);

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

// Every invocation plays all three roles, for different indexes.
// The memory barriers have the release and acquire semantics, the atomic accesses to the guards have none.
class TransitiveSemanticsCombinedFenceTestCase : public TransitiveSemanticsTestCase
{
public:
    using TransitiveSemanticsTestCase::TransitiveSemanticsTestCase;

private:
    std::string initProgram() const override;
};

std::string TransitiveSemanticsCombinedFenceTestCase::initProgram() const
{
    const Scope scope   = m_data.scope;
    const DataType type = getVar("payload").type;
    const char *typeStr = gl_type(type);

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

// Each invocation has a dedicated role within a group.
// The middle invocation only relays the synchronization, never accessing the payload.
// The atomic accesses to the guards have the release and acquire semantics.
class TransitiveSemanticsRelayAtomicTestCase : public TransitiveSemanticsTestCase
{
public:
    using TransitiveSemanticsTestCase::TransitiveSemanticsTestCase;

private:
    std::string initProgram() const override;
};

std::string TransitiveSemanticsRelayAtomicTestCase::initProgram() const
{
    const Scope scope   = m_data.scope;
    const DataType type = getVar("payload").type;
    const char *typeStr = gl_type(type);

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
        << "      skip2 = " << gl_atomicLoad("guard2", "guard2Coord", scope, semAcq2) << " == 0;\n"
        << "      r = " << gl_load("payload", "payloadCoordPartner1") << ";\n"
        << "      if (!idle && !skip2 && r != expected) {\n"
        << "         " << gl_store("fail", "failCoord", "1") << ";\n"
        << "      }\n"
        << "   }\n";

    return css.str();
}

// Each invocation has a dedicated role within a group.
// The middle invocation only relays the synchronization, never accessing the payload.
// The memory barriers have the release and acquire semantics, the atomic accesses to the guards have none.
class TransitiveSemanticsRelayFenceTestCase : public TransitiveSemanticsTestCase
{
public:
    using TransitiveSemanticsTestCase::TransitiveSemanticsTestCase;

private:
    std::string initProgram() const override;
};

std::string TransitiveSemanticsRelayFenceTestCase::initProgram() const
{
    const Scope scope   = m_data.scope;
    const DataType type = getVar("payload").type;
    const char *typeStr = gl_type(type);

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
        << "      skip2 = " << gl_atomicLoad("guard2", "guard2Coord", scope, 0) << " == 0;\n"
        << "      " << gl_memoryBarrier(scope, semAcq2) << ";\n"
        << "      r = " << gl_load("payload", "payloadCoordPartner1") << ";\n"
        << "      if (!idle && !skip2 && r != expected) {\n"
        << "         " << gl_store("fail", "failCoord", "1") << ";\n"
        << "      }\n"
        << "   }\n";

    return css.str();
}

template <typename PatternTestCase>
void addPatternTests(tcu::TestCaseGroup &root, const char *patternName, const char *syncName, bool hasCore11)
{
    static const NamedValue<bool> core11Cases[] = {
        // Supported by Vulkan1.1
        {true, "core11"},
        // Requires VK_KHR_vulkan_memory_model extension
        {false, "ext"},
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

    // Omit DATA_TYPE_FLOAT64 because all tests use image memory
    static const NamedValue<DataType> typeCases[] = {
        // uint32_t payload or atomics
        {DATA_TYPE_UINT, "u32"},
        // uint64_t payload or atomics
        {DATA_TYPE_UINT64, "u64"},
        // float payload or atomics
        {DATA_TYPE_FLOAT32, "f32"},
    };

    // Omit SCOPE_QUEUEFAMILY and SCOPE_DEVICE because all tests use workgroup memory
    static const NamedValue<Scope> scopeCases[] = {
        // workgroup scope
        {SCOPE_WORKGROUP, "workgroup"},
        // subgroup scope
        {SCOPE_SUBGROUP, "subgroup"},
    };

    const auto groupName = [](const char *prefix, const char *name) { return std::string(prefix) + "_" + name; };

    for (const auto &[core11, sc0, sc1, sc2, local0, local1, local2, type0, type1, type2, scope] :
         Combinations(core11Cases, scCases, scCases, scCases, localCases, localCases, localCases, typeCases, typeCases,
                      typeCases, scopeCases))
    {
        const CaseDef c = {
            core11.value, // bool core11;
            local0.value, // bool payloadMemLocal;
            local1.value, // bool guard1MemLocal;
            local2.value, // bool guard2MemLocal;
            sc0.value,    // StorageClass payloadSC;
            sc1.value,    // StorageClass guard1SC;
            sc2.value,    // StorageClass guard2SC;
            type0.value,  // DataType payloadType;
            type1.value,  // DataType guard1Type;
            type2.value,  // DataType guard2Type;
            scope.value,  // Scope scope;
        };

        // Mustpass11 tests should only exercise things we expect to work without Vulkan memory model. Exclude 64-bit
        // and floating point guards, whose atomic accesses require extensions, and 64-bit payloads. Exclude physical
        // storage buffer which doesn't support the legacy decorations.
        if (c.core11 && (!hasCore11 || c.payloadType == DATA_TYPE_UINT64 || c.guard1Type != DATA_TYPE_UINT ||
                         c.guard2Type != DATA_TYPE_UINT || c.payloadSC == SC_PHYSBUFFER ||
                         c.guard1SC == SC_PHYSBUFFER || c.guard2SC == SC_PHYSBUFFER))
        {
            continue;
        }

        // Payload, guard1 and guard2 use storage classes with different storage class semantics
        if (scToSem(c.guard1SC) == scToSem(c.payloadSC) || scToSem(c.guard2SC) == scToSem(c.payloadSC) ||
            scToSem(c.guard2SC) == scToSem(c.guard1SC))
        {
            continue;
        }

        // Workgroup memory is only tested as non-local
        if ((c.payloadMemLocal && c.payloadSC == SC_WORKGROUP) || (c.guard1MemLocal && c.guard1SC == SC_WORKGROUP) ||
            (c.guard2MemLocal && c.guard2SC == SC_WORKGROUP))
        {
            continue;
        }

        // Image memory has no 64-bit types
        if ((c.payloadType == DATA_TYPE_UINT64 && c.payloadSC == SC_IMAGE) ||
            (c.guard1Type == DATA_TYPE_UINT64 && c.guard1SC == SC_IMAGE) ||
            (c.guard2Type == DATA_TYPE_UINT64 && c.guard2SC == SC_IMAGE))
        {
            continue;
        }

        addTestCase(root,
                    {patternName, core11.name, syncName, groupName("payload", sc0.name), groupName("guard1", sc1.name),
                     groupName("guard2", sc2.name), groupName("payload", local0.name), groupName("guard1", local1.name),
                     groupName("guard2", local2.name), groupName("payload", type0.name),
                     groupName("guard1", type1.name), groupName("guard2", type2.name)},
                    new PatternTestCase(root.getTestContext(), scope.name, c));
    }
}

} // namespace

tcu::TestCaseGroup *createTransitiveSemanticsTests(tcu::TestContext &testCtx)
{
    de::MovePtr<tcu::TestCaseGroup> raGroup(new tcu::TestCaseGroup(testCtx, "transitive_semantics"));

    // Only the fence tests have core11 variations: without the Vulkan memory model, atomic synchronization is not
    // expected to work (i.e. atomics may be implemented as relaxed atomics)
    addPatternTests<TransitiveSemanticsCombinedAtomicTestCase>(*raGroup, "combined", "atomic", false);
    addPatternTests<TransitiveSemanticsCombinedFenceTestCase>(*raGroup, "combined", "fence", true);
    addPatternTests<TransitiveSemanticsRelayAtomicTestCase>(*raGroup, "relay", "atomic", false);
    addPatternTests<TransitiveSemanticsRelayFenceTestCase>(*raGroup, "relay", "fence", true);

    return raGroup.release();
}

} // namespace vkt::MemoryModel
