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
    TT_WAR,    // write-after-read hazard
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
    bool payloadMemLocal;
    bool guardMemLocal;
    bool coherent;
    bool core11;
    TestType testType;
    StorageClass payloadSC;
    StorageClass guardSC;
    Scope scope;
    SyncType syncType;
    Stage stage;
    DataType dataType;
};

bool hasGuard(const CaseDef &data)
{
    return data.syncType == ST_FENCE_FENCE || data.syncType == ST_FENCE_ATOMIC || data.syncType == ST_ATOMIC_FENCE ||
           data.syncType == ST_ATOMIC_ATOMIC || data.syncType == ST_ATOMIC_RMW;
}

bool isSplitBarrier(SyncType syncType)
{
    return syncType == ST_SPLIT_CONTROL_BARRIER || syncType == ST_SPLIT_CONTROL_AND_MEMORY_BARRIER ||
           syncType == ST_SPLIT_CONTROL_BARRIER_AND_BARRIER;
}

Scope getMappingScope(const CaseDef &data)
{
    const bool hasSharedBuffer = data.payloadSC == SC_WORKGROUP || (hasGuard(data) && data.guardSC == SC_WORKGROUP);
    if ((data.scope == SCOPE_DEVICE || data.scope == SCOPE_QUEUEFAMILY) && hasSharedBuffer)
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

std::vector<BufferDef> createBuffers(const CaseDef &data)
{
    const MemQualifier payloadQualifier = getMemQualifier(data);
    std::vector<BufferDef> buffers{{"payload", payloadQualifier, data.payloadSC, data.dataType, data.payloadMemLocal}};
    if (hasGuard(data))
    {
        const MemQualifier guardQualifier = data.guardSC == SC_IMAGE ? payloadQualifier : MEM_NONE;
        buffers.push_back({"guard", guardQualifier, data.guardSC, data.dataType, data.guardMemLocal});
    }
    buffers.push_back({"fail", MEM_NONE, SC_BUFFER, DATA_TYPE_UINT, true});
    return buffers;
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
    : MemoryModelTestCase(context, name, {data.stage, data.core11}, createBuffers(data))
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

    if (isSplitBarrier(m_data.syncType))
        checkSupportSplitBarriers(context);

    if (m_data.scope == SCOPE_SUBGROUP)
        checkSupportSubgroupFeatures(context);

    checkSupportBuffers(context);
    if (hasGuard(m_data))
        checkSupportBufferAtomics(context, "guard");
    checkSupportStageAtomics(context);
}

std::string MessagePassingTestCase::initProgram() const
{
    const uint32_t semAv      = (m_data.testType == TT_WAR || m_data.coherent) ? 0u : uint32_t(SEM_AVAILABLE);
    const uint32_t semVis     = (m_data.testType == TT_WAR || m_data.coherent) ? 0u : uint32_t(SEM_VISIBLE);
    const uint32_t semPayload = getBufferSemantics("payload");
    const uint32_t semGuard   = hasGuard(m_data) ? getBufferSemantics("guard") : 0;
    const uint32_t semRel     = SEM_RELEASE | semPayload | semAv;
    const uint32_t semAcq     = SEM_ACQUIRE | semPayload | semVis;
    const uint32_t semAcqRel  = SEM_ACQUIRE_RELEASE | semPayload | semAv | semVis;
    const Scope scope         = m_data.scope;

    const DataType type        = getBuffer("payload").type;
    const char *typeStr        = gl_type(type);
    const std::string expected = (m_data.testType == TT_MP) ? (std::string{typeStr} + "(invocationIdPartner)") : "0";

    std::ostringstream css;

    css << initPartnerCoordinates(getMappingScope(m_data));

    css << "   bool skip = false;\n"
        << "   uint value = invocationId;\n"
        << "   " << typeStr << " r = 0;\n"
        << "   " << typeStr << " expected = " << expected << ";\n";

    if (m_data.testType == TT_MP)
    {
        // Prefetch the partner's payload; this adds 0 because every stored value is below 2^31
        css << "   value += " << gl_toUint(type, gl_load("payload", "payloadCoordPartner")) << " >> 31;\n"
            << "   " << gl_store("payload", "payloadCoord", "value") << ";\n";
    }
    else
    {
        css << "   r = " << gl_load("payload", "payloadCoordPartner") << ";\n";
    }

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
        css << "   if (!idle && !skip) {\n"
            << "      r = " << gl_load("payload", "payloadCoordPartner") << ";\n"
            << "      if (r != expected) { " << gl_store("fail", "failCoord", "1") << "; }\n"
            << "   }\n";
    else
        css << "   if (!skip) { " << gl_store("payload", "payloadCoord", "value") << "; }\n"
            << "   if (!idle && r != expected) { " << gl_store("fail", "failCoord", "1") << "; }\n";

    return css.str();
}

tcu::TestCaseGroup *createTests(tcu::TestContext &testCtx, const char *groupName, TestType testType)
{
    static const NamedValue<bool> core11Cases[] = {
        // Supported by Vulkan1.1
        {true, "core11"},
        // Requires VK_KHR_vulkan_memory_model extension
        {false, "ext"},
    };

    static const NamedValue<DataType> dtCases[] = {
        // uint32_t atomics
        {DATA_TYPE_UINT, "u32"},
        // uint64_t atomics
        {DATA_TYPE_UINT64, "u64"},
        // float32 atomics
        {DATA_TYPE_FLOAT32, "f32"},
        // float64 atomics
        {DATA_TYPE_FLOAT64, "f64"},
    };

    static const NamedValue<bool> cohCases[] = {
        // coherent payload variable
        {true, "coherent"},
        // noncoherent payload variable
        {false, "noncoherent"},
    };

    static const NamedValue<SyncType> stCases[] = {
        // release fence, acquire fence
        {ST_FENCE_FENCE, "fence_fence"},
        // release fence, atomic acquire
        {ST_FENCE_ATOMIC, "fence_atomic"},
        // atomic release, acquire fence
        {ST_ATOMIC_FENCE, "atomic_fence"},
        // atomic release, atomic acquire
        {ST_ATOMIC_ATOMIC, "atomic_atomic"},
        // control barrier
        {ST_CONTROL_BARRIER, "control_barrier"},
        // control barrier with separate memory barriers
        {ST_CONTROL_AND_MEMORY_BARRIER, "control_and_memory_barrier"},
        // split control barrier with release/acquire
        {ST_SPLIT_CONTROL_BARRIER, "split_control_barrier"},
        // split control barrier with separate memory barriers
        {ST_SPLIT_CONTROL_AND_MEMORY_BARRIER, "split_control_and_memory_barrier"},
        // split control barrier with barrier
        {ST_SPLIT_CONTROL_BARRIER_AND_BARRIER, "split_control_barrier_and_barrier"},
    };

    // Atomic RMW variants exist only for atomic release and acquire (ST_ATOMIC_ATOMIC),
    // where they replace the atomic store and load with exchanges (ST_ATOMIC_RMW)
    static const NamedValue<bool> rmwCases[] = {
        {false, "atomicwrite"},
        {true, "atomicrmw"},
    };

    static const NamedValue<Scope> scopeCases[] = {
        {SCOPE_DEVICE, "device"},
        {SCOPE_QUEUEFAMILY, "queuefamily"},
        {SCOPE_WORKGROUP, "workgroup"},
        {SCOPE_SUBGROUP, "subgroup"},
    };

    static const NamedValue<bool> plCases[] = {
        // payload variable in non-local memory
        {false, "payload_nonlocal"},
        // payload variable in local memory
        {true, "payload_local"},
    };

    static const NamedValue<StorageClass> pscCases[] = {
        // payload variable in buffer memory
        {SC_BUFFER, "buffer"},
        // payload variable in image memory
        {SC_IMAGE, "image"},
        // payload variable in workgroup memory
        {SC_WORKGROUP, "workgroup"},
        // payload variable in physical storage buffer memory
        {SC_PHYSBUFFER, "physbuffer"},
    };

    static const NamedValue<bool> glCases[] = {
        // guard variable in non-local memory
        {false, "guard_nonlocal"},
        // guard variable in local memory
        {true, "guard_local"},
    };

    static const NamedValue<StorageClass> gscCases[] = {
        // guard variable in buffer memory
        {SC_BUFFER, "buffer"},
        // guard variable in image memory
        {SC_IMAGE, "image"},
        // guard variable in workgroup memory
        {SC_WORKGROUP, "workgroup"},
        // guard variable in physical storage buffer memory
        {SC_PHYSBUFFER, "physbuffer"},
    };

    static const NamedValue<Stage> stageCases[] = {
        {STAGE_COMPUTE, "comp"},
        {STAGE_VERTEX, "vert"},
        {STAGE_FRAGMENT, "frag"},
    };

    de::MovePtr<tcu::TestCaseGroup> ttGroup(new tcu::TestCaseGroup(testCtx, groupName));

    for (const auto &[core11, dt, coh, st, rmw, scope, pl, psc, gl, gsc, stage] :
         Combinations(core11Cases, dtCases, cohCases, stCases, rmwCases, scopeCases, plCases, pscCases, glCases,
                      gscCases, stageCases))
    {
        const SyncType syncType = rmw.value ? ST_ATOMIC_RMW : st.value;

        CaseDef c = {
            pl.value,     // bool payloadMemLocal;
            gl.value,     // bool guardMemLocal;
            coh.value,    // bool coherent;
            core11.value, // bool core11;
            testType,     // TestType testType;
            psc.value,    // StorageClass payloadSC;
            gsc.value,    // StorageClass guardSC;
            scope.value,  // Scope scope;
            syncType,     // SyncType syncType;
            stage.value,  // Stage stage;
            dt.value,     // DataType dataType;
        };

        // Mustpass11 tests should only exercise things we expect to work on existing implementations. Exclude
        // noncoherent tests which require new extensions, and assume atomic synchronization wouldn't work (i.e. atomics
        // may be implemented as relaxed atomics). Exclude queuefamily scope which doesn't exist in Vulkan 1.1. Exclude
        // physical storage buffer which doesn't support the legacy decorations.
        if (c.core11 && (c.coherent == 0 || c.syncType == ST_FENCE_ATOMIC || c.syncType == ST_ATOMIC_FENCE ||
                         c.syncType == ST_ATOMIC_ATOMIC || c.syncType == ST_ATOMIC_RMW ||
                         c.dataType == DATA_TYPE_UINT64 || c.dataType == DATA_TYPE_FLOAT64 ||
                         c.scope == SCOPE_QUEUEFAMILY || c.payloadSC == SC_PHYSBUFFER || c.guardSC == SC_PHYSBUFFER))
        {
            continue;
        }

        if (c.stage != STAGE_COMPUTE && c.scope == SCOPE_WORKGROUP)
        {
            continue;
        }

        // Don't exercise local and non-local for workgroup memory
        // Also don't exercise workgroup memory for non-compute stages
        if (c.payloadSC == SC_WORKGROUP && (c.payloadMemLocal != 0 || c.stage != STAGE_COMPUTE))
        {
            continue;
        }

        if (c.guardSC == SC_WORKGROUP && (c.guardMemLocal != 0 || c.stage != STAGE_COMPUTE))
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

        // uint64/float32/float64 testing is primarily for atomics
        // so only test it for ST_ATOMIC_ATOMIC and ST_ATOMIC_RMW
        const bool atomicTesting =
            (c.dataType == DATA_TYPE_UINT64 || c.dataType == DATA_TYPE_FLOAT32 || c.dataType == DATA_TYPE_FLOAT64);
        if (atomicTesting && c.syncType != ST_ATOMIC_ATOMIC && c.syncType != ST_ATOMIC_RMW)
        {
            continue;
        }

        // No 64-bit image types, so skip tests with both payload and guard in image memory
        if (c.dataType == DATA_TYPE_UINT64 && c.payloadSC == SC_IMAGE && c.guardSC == SC_IMAGE)
        {
            continue;
        }

        // No support for atomic operations on 64-bit floating point images
        if (c.dataType == DATA_TYPE_FLOAT64 && (c.payloadSC == SC_IMAGE || c.guardSC == SC_IMAGE))
        {
            continue;
        }
        // Barrier tests have no guard, so only run them with gsc,gl==0
        if (!hasGuard(c) && (c.guardSC != 0 || c.guardMemLocal != 0))
        {
            continue;
        }

        // Atomic RMW is only tested with atomic release and acquire
        if (rmw.value && st.value != ST_ATOMIC_ATOMIC)
        {
            continue;
        }

        addTestCase(
            *ttGroup,
            {core11.name, dt.name, coh.name, st.name, rmw.name, scope.name, pl.name, psc.name, gl.name, gsc.name},
            new MessagePassingTestCase(testCtx, stage.name, c));
    }

    return ttGroup.release();
}

} // namespace

tcu::TestCaseGroup *createMessagePassingTests(tcu::TestContext &testCtx)
{
    return createTests(testCtx, "message_passing", TT_MP);
}

tcu::TestCaseGroup *createWriteAfterReadTests(tcu::TestContext &testCtx)
{
    return createTests(testCtx, "write_after_read", TT_WAR);
}

} // namespace vkt::MemoryModel
