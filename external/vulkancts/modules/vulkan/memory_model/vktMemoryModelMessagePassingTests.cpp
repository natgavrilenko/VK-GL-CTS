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
 * \brief Memory Model Message Passing Tests
 *//*--------------------------------------------------------------------*/

#include "vktMemoryModelMessagePassingTests.hpp"
#include "vktMemoryModelTestCase.hpp"
#include "vktMemoryModelUtil.hpp"

#include "deDefs.h"
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

enum TestType
{
    TT_MP = 0, // message passing
    TT_WAR,    // write-after-read
};

enum SyncType
{
    ST_FENCE_FENCE = 0,
    ST_FENCE_ATOMIC,
    ST_ATOMIC_FENCE,
    ST_ATOMIC_ATOMIC,
    ST_ATOMIC_RMW,
    ST_CONTROL_BARRIER,
    ST_CONTROL_AND_MEMORY_BARRIER,
    ST_SPLIT_CONTROL_BARRIER,
    ST_SPLIT_CONTROL_AND_MEMORY_BARRIER,
    ST_SPLIT_CONTROL_BARRIER_AND_BARRIER
};

struct CaseDef
{
    bool core11;
    bool payloadMemLocal;
    bool guardMemLocal;
    bool coherent;
    TestType testType;
    DataType payloadType;
    DataType guardType;
    StorageClass payloadSC;
    StorageClass guardSC;
    SyncType syncType;
    Scope scope;
    Stage stage;
};

bool hasGuard(const CaseDef &data)
{
    return data.syncType == ST_FENCE_FENCE || data.syncType == ST_FENCE_ATOMIC || data.syncType == ST_ATOMIC_FENCE ||
           data.syncType == ST_ATOMIC_ATOMIC || data.syncType == ST_ATOMIC_RMW;
}

Scope getMappingScope(const CaseDef &data)
{
    const bool hasSharedVar = data.payloadSC == SC_WORKGROUP || (hasGuard(data) && data.guardSC == SC_WORKGROUP);
    if (hasSharedVar && data.scope > SCOPE_WORKGROUP)
    {
        return SCOPE_WORKGROUP;
    }
    return data.scope;
}

MemQualifier getMemQualifier(const CaseDef &data)
{
    if (data.core11)
        return MEM_COHERENT;
    if (!data.coherent)
        return MEM_NONPRIVATE;
    return scopeToQualifier(data.scope);
}

std::vector<VarDef> createVars(const CaseDef &data)
{
    const MemQualifier payloadQualifier = getMemQualifier(data);
    std::vector<VarDef> vars{{"payload", payloadQualifier, data.payloadSC, data.payloadType, data.payloadMemLocal}};
    if (hasGuard(data))
    {
        const MemQualifier guardQualifier = (data.core11 && data.guardSC == SC_IMAGE) ? MEM_COHERENT : MEM_NONE;
        vars.push_back({"guard", guardQualifier, data.guardSC, data.guardType, data.guardMemLocal});
    }
    vars.push_back({"fail", MEM_NONE, SC_BUFFER, DATA_TYPE_UINT, true});
    return vars;
}

class MessagePassingTestCase : public MemoryModelTestCase
{
public:
    MessagePassingTestCase(tcu::TestContext &context, const char *name, const CaseDef &data);
    void checkSupport(Context &context) const override;

private:
    std::string initProgram() const override;
    const CaseDef m_data;
};

MessagePassingTestCase::MessagePassingTestCase(tcu::TestContext &context, const char *name, const CaseDef &data)
    : MemoryModelTestCase(context, name, {data.stage, data.core11}, createVars(data))
    , m_data(data)
{
}

void MessagePassingTestCase::checkSupport(Context &context) const
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

    if (m_data.syncType == ST_SPLIT_CONTROL_BARRIER || m_data.syncType == ST_SPLIT_CONTROL_AND_MEMORY_BARRIER ||
        m_data.syncType == ST_SPLIT_CONTROL_BARRIER_AND_BARRIER)
        checkSupportSplitBarriers(context);

    if (m_data.scope == SCOPE_SUBGROUP)
        checkSupportSubgroupFeatures(context);

    checkSupportVars(context);
    if (hasGuard(m_data))
        checkSupportVarAtomics(context, "guard");
    checkSupportStageAtomics(context);
}

std::string MessagePassingTestCase::initProgram() const
{
    const DataType type       = getVar("payload").type;
    const char *typeStr       = gl_type(type);
    const Scope scope         = m_data.scope;
    const uint32_t semAv      = (m_data.testType == TT_WAR || m_data.coherent) ? 0u : uint32_t(SEM_AVAILABLE);
    const uint32_t semVis     = (m_data.testType == TT_WAR || m_data.coherent) ? 0u : uint32_t(SEM_VISIBLE);
    const uint32_t semPayload = getVarSemantics("payload");
    const uint32_t semGuard   = hasGuard(m_data) ? getVarSemantics("guard") : 0;
    const uint32_t semRel     = SEM_RELEASE | semPayload | semAv;
    const uint32_t semAcq     = SEM_ACQUIRE | semPayload | semVis;
    const uint32_t semAcqRel  = SEM_ACQUIRE_RELEASE | semPayload | semAv | semVis;

    const std::string expected = (m_data.testType == TT_MP) ? (std::string{typeStr} + "(invocationIdPartner)") : "0";

    std::ostringstream css;

    css << initPartnerCoordinates(getMappingScope(m_data));

    css << "   bool skip = false;\n"
        << "   uint value = invocationId;\n"
        << "   " << typeStr << " r = 0;\n"
        << "   " << typeStr << " expected = " << expected << ";\n";

    if (m_data.testType == TT_MP)
        css << "   value += " << gl_toUint(type, gl_load("payload", "payloadCoordPartner")) << " >> 31;\n"
            << "   " << gl_store("payload", "payloadCoord", "value") << ";\n";
    else
        css << "   r = " << gl_load("payload", "payloadCoordPartner") << ";\n";

    switch (m_data.syncType)
    {
    case ST_FENCE_FENCE:
        css << "   " << gl_memoryBarrier(scope, semRel | semGuard) << ";\n"
            << "   " << gl_atomicStore("guard", "guardCoord", "1u", scope, 0) << ";\n"
            << "   skip = " << gl_atomicLoad("guard", "guardCoordPartner", scope, 0) << " == 0;\n"
            << "   " << gl_memoryBarrier(scope, semAcq | semGuard) << ";\n";
        break;
    case ST_FENCE_ATOMIC:
        css << "   " << gl_memoryBarrier(scope, semRel | semGuard) << ";\n"
            << "   " << gl_atomicStore("guard", "guardCoord", "1u", scope, 0) << ";\n"
            << "   skip = " << gl_atomicLoad("guard", "guardCoordPartner", scope, semAcq) << " == 0;\n";
        break;
    case ST_ATOMIC_FENCE:
        css << "   " << gl_atomicStore("guard", "guardCoord", "1u", scope, semRel) << ";\n"
            << "   skip = " << gl_atomicLoad("guard", "guardCoordPartner", scope, 0) << " == 0;\n"
            << "   " << gl_memoryBarrier(scope, semAcq | semGuard) << ";\n";
        break;
    case ST_ATOMIC_ATOMIC:
        css << "   " << gl_atomicStore("guard", "guardCoord", "1u", scope, semRel) << ";\n"
            << "   skip = " << gl_atomicLoad("guard", "guardCoordPartner", scope, semAcq) << " == 0;\n";
        break;
    case ST_ATOMIC_RMW:
        css << "   " << gl_atomicExchange("guard", "guardCoord", "1u", scope, semRel) << ";\n"
            << "   skip = " << gl_atomicExchange("guard", "guardCoordPartner", "2u", scope, semAcq) << " == 0;\n";
        break;
    case ST_CONTROL_BARRIER:
        css << "   " << gl_controlBarrier(scope, scope, semAcqRel) << ";\n";
        break;
    case ST_CONTROL_AND_MEMORY_BARRIER:
        css << "   " << gl_memoryBarrier(scope, semRel) << ";\n"
            << "   " << gl_controlBarrier(scope, SCOPE_INVOCATION, 0) << ";\n"
            << "   " << gl_memoryBarrier(scope, semAcq) << ";\n";
        break;
    case ST_SPLIT_CONTROL_BARRIER:
        css << "   " << gl_controlBarrierArrive(scope, scope, semRel) << ";\n"
            << "   " << gl_controlBarrierWait(scope, scope, semAcq) << ";\n";
        break;
    case ST_SPLIT_CONTROL_AND_MEMORY_BARRIER:
        css << "   " << gl_memoryBarrier(scope, semRel) << ";\n"
            << "   " << gl_controlBarrierArrive(scope, scope, 0) << ";\n"
            << "   " << gl_controlBarrierWait(scope, scope, 0) << ";\n"
            << "   " << gl_memoryBarrier(scope, semAcq) << ";\n";
        break;
    case ST_SPLIT_CONTROL_BARRIER_AND_BARRIER:
        css << "   " << gl_controlBarrierArrive(scope, scope, semRel) << ";\n"
            << "   " << gl_barrier() << ";\n"
            << "   " << gl_controlBarrierWait(scope, scope, semAcq) << ";\n";
        break;
    default:
        DE_ASSERT(0);
    }

    if (m_data.testType == TT_MP)
        css << "   r = " << gl_load("payload", "payloadCoordPartner") << ";\n"
            << "   if (!idle && !skip && r != expected) {\n"
            << "      " << gl_store("fail", "failCoord", "1") << ";\n"
            << "   }\n";
    else
        css << "   if (!skip) { " << gl_store("payload", "payloadCoord", "value") << "; }\n"
            << "   if (!idle && r != expected) {\n"
            << "      " << gl_store("fail", "failCoord", "1") << ";\n"
            << "   }\n";

    return css.str();
}

tcu::TestCaseGroup *addTests(tcu::TestContext &testCtx, const char *variantName, TestType testType)
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

    static const NamedValue<SyncType> syncCases[] = {
        {ST_FENCE_FENCE, "fence_fence"},
        {ST_FENCE_ATOMIC, "fence_atomic"},
        {ST_ATOMIC_FENCE, "atomic_fence"},
        {ST_ATOMIC_ATOMIC, "atomic_atomic"},
        {ST_CONTROL_BARRIER, "control_barrier"},
        {ST_CONTROL_AND_MEMORY_BARRIER, "control_and_memory_barrier"},
        {ST_SPLIT_CONTROL_BARRIER, "split_control_barrier"},
        {ST_SPLIT_CONTROL_AND_MEMORY_BARRIER, "split_control_and_memory_barrier"},
        {ST_SPLIT_CONTROL_BARRIER_AND_BARRIER, "split_control_barrier_and_barrier"},
    };

    // A separate case list to preserve original test names
    static const NamedValue<bool> rmwCases[] = {
        {false, "atomicwrite"}, // syncCase from syncCases
        {true, "atomicrmw"},    // syncCase is ST_ATOMIC_RMW
    };

    static const NamedValue<Scope> scopeCases[] = {
        {SCOPE_DEVICE, "device"},
        {SCOPE_QUEUEFAMILY, "queuefamily"},
        {SCOPE_WORKGROUP, "workgroup"},
        {SCOPE_SUBGROUP, "subgroup"},
    };

    static const NamedValue<bool> cohCases[] = {
        {true, "coherent"},
        {false, "noncoherent"},
    };

    static const NamedValue<Stage> stageCases[] = {
        {STAGE_COMPUTE, "comp"},
        {STAGE_VERTEX, "vert"},
        {STAGE_FRAGMENT, "frag"},
    };

    de::MovePtr<tcu::TestCaseGroup> group(new tcu::TestCaseGroup(testCtx, variantName));

    const auto groupName = [](const char *prefix, const char *name) { return std::string(prefix) + "_" + name; };

    for (const auto &[core11, type, local0, sc0, local1, sc1, sync, rmw, scope, coh, stage] :
         Combinations(core11Cases, typeCases, localCases, scCases, localCases, scCases, syncCases, rmwCases, scopeCases,
                      cohCases, stageCases))
    {
        const SyncType syncType = rmw.value ? ST_ATOMIC_RMW : sync.value;

        // To keep the requirements of old tests, u64 tests with one variable in image memory use u32 for this variable.
        // New u64 tests with both variables in image memory use 64-bit images.
        const bool isUint64OneImage =
            type.value == DATA_TYPE_UINT64 && (sc0.value == SC_IMAGE) != (sc1.value == SC_IMAGE);
        const DataType payloadType = (isUint64OneImage && sc0.value == SC_IMAGE) ? DATA_TYPE_UINT : type.value;
        const DataType guardType   = (isUint64OneImage && sc1.value == SC_IMAGE) ? DATA_TYPE_UINT : type.value;

        const CaseDef c = {
            core11.value, // bool core11;
            local0.value, // bool payloadMemLocal;
            local1.value, // bool guardMemLocal;
            coh.value,    // bool coherent;
            testType,     // TestType testType;
            payloadType,  // DataType payloadType;
            guardType,    // DataType guardType;
            sc0.value,    // StorageClass payloadSC;
            sc1.value,    // StorageClass guardSC;
            syncType,     // SyncType syncType;
            scope.value,  // Scope scope;
            stage.value,  // Stage stage;
        };

        // Tests in core11 include only variations that do not require Vulkan memory model. Skip noncoherent tests,
        // 64-bit data types, and release-acquire atomics. Skip queue family scope which doesn't exist in Vulkan 1.1.
        // Skip physical storage buffer which doesn't support coherent decoration.
        if (c.core11 && (!c.coherent || c.syncType == ST_FENCE_ATOMIC || c.syncType == ST_ATOMIC_FENCE ||
                         c.syncType == ST_ATOMIC_ATOMIC || c.syncType == ST_ATOMIC_RMW ||
                         type.value == DATA_TYPE_UINT64 || type.value == DATA_TYPE_FLOAT64 ||
                         c.scope == SCOPE_QUEUEFAMILY || c.payloadSC == SC_PHYSBUFFER || c.guardSC == SC_PHYSBUFFER))
        {
            continue;
        }

        // Workgroup memory is only tested as non-local
        if ((c.payloadMemLocal && c.payloadSC == SC_WORKGROUP) || (c.guardMemLocal && c.guardSC == SC_WORKGROUP))
        {
            continue;
        }

        // Skip workgroup scope for non-compute stages
        if (c.stage != STAGE_COMPUTE && c.scope == SCOPE_WORKGROUP)
        {
            continue;
        }

        // Skip workgroup storage class for non-compute stages
        if (c.stage != STAGE_COMPUTE && (c.payloadSC == SC_WORKGROUP || c.guardSC == SC_WORKGROUP))
        {
            continue;
        }

        // No support for control/split barrier with larger than workgroup scope, or non-compute stages
        if (!hasGuard(c) && (c.scope == SCOPE_DEVICE || c.scope == SCOPE_QUEUEFAMILY || c.stage != STAGE_COMPUTE))
        {
            continue;
        }

        if (c.syncType == ST_SPLIT_CONTROL_BARRIER_AND_BARRIER && c.scope != SCOPE_WORKGROUP)
        {
            continue;
        }

        // Limit uint64/float32/float64 data types to ST_ATOMIC_ATOMIC and ST_ATOMIC_RMW tests
        if (c.syncType != ST_ATOMIC_ATOMIC && c.syncType != ST_ATOMIC_RMW &&
            (is64BitType(type.value) || !isIntType(type.value)))
        {
            continue;
        }

        // Vulkan has no 64-bit floating point images
        if (type.value == DATA_TYPE_FLOAT64 && (c.payloadSC == SC_IMAGE || c.guardSC == SC_IMAGE))
        {
            continue;
        }

        // Barrier tests have no guard, so run them only for one guard variation
        if (!hasGuard(c) && (c.guardSC > 0 || c.guardMemLocal))
        {
            continue;
        }

        // Execute RMW tests only if syncType is atomic (workaround to preserve old test names)
        if (rmw.value && sync.value != ST_ATOMIC_ATOMIC)
        {
            continue;
        }

        addTestCase(*group,
                    {core11.name, type.name, coh.name, sync.name, rmw.name, scope.name,
                     groupName("payload", local0.name), sc0.name, groupName("guard", local1.name), sc1.name},
                    new MessagePassingTestCase(testCtx, stage.name, c));
    }

    return group.release();
}

} // namespace

tcu::TestCaseGroup *createMessagePassingTests(tcu::TestContext &testCtx)
{
    return addTests(testCtx, "message_passing", TT_MP);
}

tcu::TestCaseGroup *createWriteAfterReadTests(tcu::TestContext &testCtx)
{
    return addTests(testCtx, "write_after_read", TT_WAR);
}

} // namespace vkt::MemoryModel
