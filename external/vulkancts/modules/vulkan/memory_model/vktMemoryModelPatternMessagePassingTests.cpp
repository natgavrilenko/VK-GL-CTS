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
 * \brief Memory Model Pattern Message Passing Tests
 *//*--------------------------------------------------------------------*/

#include "vktMemoryModelPatternMessagePassingTests.hpp"
#include "vktMemoryModelPatternTestCase.hpp"
#include "vktMemoryModelPatternTestInstance.hpp"

#include "deDefs.h"
#include "deSharedPtr.hpp"
#include "tcuTestCase.hpp"
#include "vkBuilderUtil.hpp"
#include "vktTestCase.hpp"

#include <algorithm>
#include <sstream>

namespace vkt::MemoryModel
{
namespace
{
using namespace vk;

enum SyncType
{
    ST_FENCE_FENCE = 0,
    ST_FENCE_ATOMIC,
    ST_ATOMIC_FENCE,
    ST_ATOMIC_ATOMIC,
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
    bool atomicRMW;
    StorageClass payloadSC;
    StorageClass guardSC;
    Scope scope;
    SyncType syncType;
    Stage stage;
    DataType dataType;
};

bool isSplitBarrier(SyncType syncType)
{
    return syncType == ST_SPLIT_CONTROL_BARRIER || syncType == ST_SPLIT_CONTROL_AND_MEMORY_BARRIER ||
           syncType == ST_SPLIT_CONTROL_BARRIER_AND_BARRIER;
}

class MemoryModelPatternBaseTestCase : public MemoryModelGlTestCase
{
public:
    MemoryModelPatternBaseTestCase(tcu::TestContext &context, const char *name, const CaseDef &data);
    void checkSupport(Context &context) const override;
    void initPrograms(SourceCollections &programCollection) const override;
    TestInstance *createInstance(Context &context) const override;

protected:
    virtual void initProgramBody(std::stringstream &css) const = 0;
    void initSyncPair(std::stringstream &css, uint32_t semAv, uint32_t semVis) const;
    CaseDef m_data;

private:
    static GlMemQualifier getMemQualifier(const CaseDef &data);
    static Scope getMappingScope(const CaseDef &data);
};

MemoryModelPatternBaseTestCase::MemoryModelPatternBaseTestCase(tcu::TestContext &context, const char *name, const CaseDef &data)
    : MemoryModelGlTestCase(context, name, {
          { "payload", data.payloadSC, data.dataType, getMemQualifier(data), data.payloadMemLocal },
          { "guard", data.guardSC, data.dataType, data.guardSC == SC_IMAGE ? getMemQualifier(data) : MEM_NONE, data.guardMemLocal },
          { "fail", SC_BUFFER, DATA_TYPE_UINT, MEM_NONE, true },
      })
    , m_data(data)
{
}

void MemoryModelPatternBaseTestCase::checkSupport(Context &context) const
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
        checkSupportStageSplitBarriers(context, m_data.stage);

    if (m_data.scope == SCOPE_SUBGROUP)
        checkSupportStageSubgroupFeatures(context, m_data.stage);

    checkSupportBuffers(context);
    checkSupportBufferAtomics(context, "guard");
    checkSupportStageAtomics(context, m_data.stage);
}

void MemoryModelPatternBaseTestCase::initPrograms(SourceCollections &programCollection) const
{
    const Scope mappingScope = getMappingScope(m_data);
    const bool splitBarrier = isSplitBarrier(m_data.syncType);
    const bool hasFloats = any_of(m_buffers.cbegin(), m_buffers.cend(), [](BufferDef b) { return b.type == DATA_TYPE_FLOAT32 || b.type == DATA_TYPE_FLOAT64; });

    std::stringstream css;
    css << "#version 450 core\n";
    if (!m_data.core11)
    {
        css << "#pragma use_vulkan_memory_model\n";
    }
    if (hasFloats)
    {
        css << "#extension GL_EXT_shader_atomic_float : enable\n";
    }
    if (splitBarrier)
    {
        css << "#extension GL_EXT_split_barrier : enable\n";
    }
    css << "#extension GL_KHR_shader_subgroup_basic : enable\n"
           "#extension GL_KHR_shader_subgroup_shuffle : enable\n"
           "#extension GL_KHR_shader_subgroup_ballot : enable\n"
           "#extension GL_KHR_memory_scope_semantics : enable\n"
           "#extension GL_ARB_gpu_shader_int64 : enable\n"
           "#extension GL_EXT_buffer_reference : enable\n"
           "// DIM/NUM_WORKGROUP_EACH_DIM overriden by spec constants\n"
           "layout(constant_id = 0) const int DIM = 1;\n"
           "layout(constant_id = 1) const int NUM_WORKGROUP_EACH_DIM = 1;\n"
           "struct S { " << gl_type("payload") << " x[DIM*DIM]; };\n";

    if (m_data.stage == STAGE_COMPUTE)
    {
        css << "layout(local_size_x_id = 0, local_size_y_id = 0, local_size_z = 1) in;\n";
    }
    css << gl_layout();

    css << "void main()\n"
           "{\n"
           "   bool pass = true;\n"
           "   bool skip = false;\n"
        << "   " << gl_type("payload") << " r = 0;\n"
        << gl_layoutReferences();

    if (m_data.stage == STAGE_FRAGMENT)
    {
        // Kill helper invocations so they don't load outside the bounds of the SSBO.
        // Helper pixels are also initially "active" and if a thread gets one as its
        // partner in SCOPE_SUBGROUP mode, it can't run the test.
        css << "   if (gl_HelperInvocation) { return; }\n";
    }

    if (mappingScope == SCOPE_SUBGROUP)
    {
        css << "   uvec4 ballot = subgroupBallot(true);\n"
               "   if (!subgroupBallotBitExtract(ballot, gl_SubgroupInvocationID^(gl_SubgroupSize-1))) { return; }\n";
    }

    css << gl_layoutCoordinates(m_data.stage, mappingScope)
        << gl_layoutBufferCoordinates()
        << gl_layoutInitSharedMemory(mappingScope);

    initProgramBody(css);

    css << "}\n";

    if (m_data.stage == STAGE_FRAGMENT)
    {
        // Draw a fullscreen triangle strip based on gl_VertexIndex
        std::stringstream vss;
        vss << "#version 450 core\n"
               "vec2 coords[4] = {ivec2(-1,-1), ivec2(-1, 1), ivec2(1, -1), ivec2(1, 1)};\n"
               "void main() { gl_Position = vec4(coords[gl_VertexIndex], 0, 1); }\n";
        programCollection.glslSources.add("vert") << glu::VertexSource(vss.str());
    }

    const vk::ShaderBuildOptions buildOptions(programCollection.usedVulkanVersion, vk::SPIRV_VERSION_1_3, 0u);

    switch (m_data.stage)
    {
    case STAGE_COMPUTE:
        programCollection.glslSources.add("test") << glu::ComputeSource(css.str()) << buildOptions;
        break;
    case STAGE_VERTEX:
        programCollection.glslSources.add("test") << glu::VertexSource(css.str()) << buildOptions;
        break;
    case STAGE_FRAGMENT:
        programCollection.glslSources.add("test") << glu::FragmentSource(css.str()) << buildOptions;
        break;
    default:
        DE_ASSERT(0);
    }
}

TestInstance *MemoryModelPatternBaseTestCase::createInstance(Context &context) const
{
    return new MemoryModelTestInstance(context, m_data.stage, m_buffers);
}

void MemoryModelPatternBaseTestCase::initSyncPair(std::stringstream &css, uint32_t semAv, uint32_t semVis) const
{
    const Scope scope = m_data.scope;
    const uint32_t semPayload = getBufferSemantics("payload");
    const uint32_t semGuard = getBufferSemantics("guard");
    const uint32_t semRel = SEM_RELEASE | semPayload | semAv;
    const uint32_t semAcq = SEM_ACQUIRE | semPayload | semVis;
    const uint32_t semAcqRel = SEM_ACQUIRE_RELEASE | semPayload | semAv | semVis;

    switch (m_data.syncType)
    {
    case ST_ATOMIC_ATOMIC:
    {
        if (m_data.atomicRMW)
        {
            css << "   " << gl_atomicExchange("guard", "guardCoord", "1u", scope, semRel) << ";\n"
                << "   skip = " << gl_atomicExchange("guard", "guardPartnerCoord", "2u", scope, semAcq) << " == 0;\n";
        }
        else
        {
            css << "   " << gl_atomicStore("guard", "guardCoord", "1u", scope, semRel) << ";\n"
                << "   skip = " << gl_atomicLoad("guard", "guardPartnerCoord", scope, semAcq) << " == 0;\n";
        }
        break;
    }
    case ST_ATOMIC_FENCE:
        css << "   " << gl_atomicStore("guard", "guardCoord", "1u", scope, semRel) << ";\n"
            << "   skip = " << gl_atomicLoad("guard", "guardPartnerCoord", scope, 0) << " == 0;\n"
            << "   " << gl_memoryBarrier(scope, semAcq | semGuard) << ";\n";
        break;
    case ST_FENCE_ATOMIC:
        css << "   " << gl_memoryBarrier(scope, semRel | semGuard) << ";\n"
            << "   " << gl_atomicStore("guard", "guardCoord", "1u", scope, 0) << ";\n"
            << "   skip = " << gl_atomicLoad("guard", "guardPartnerCoord", scope, semAcq) << " == 0;\n";
        break;
    case ST_FENCE_FENCE:
        css << "   " << gl_memoryBarrier(scope, semRel | semGuard) << ";\n"
            << "   " << gl_atomicStore("guard", "guardCoord", "1u", scope, 0) << ";\n"
            << "   skip = " << gl_atomicLoad("guard", "guardPartnerCoord", scope, 0) << " == 0;\n"
            << "   " << gl_memoryBarrier(scope, semAcq | semGuard) << ";\n";
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
}

GlMemQualifier MemoryModelPatternBaseTestCase::getMemQualifier(const CaseDef &data)
{
    if (data.core11)
        return MEM_COHERENT;
    if (!data.coherent)
        return MEM_NONPRIVATE;
    switch (data.scope)
    {
        case SCOPE_DEVICE:
            return MEM_DEVICE;
        case SCOPE_QUEUEFAMILY:
            return MEM_QUEUEFAMILY;
        case SCOPE_WORKGROUP:
            return MEM_WORKGROUP;
        case SCOPE_SUBGROUP:
            return MEM_SUBGROUP;
        default:
            DE_ASSERT(0);
            return MEM_NONE;
    }
}

Scope MemoryModelPatternBaseTestCase::getMappingScope(const CaseDef &data)
{
    if ((data.scope == SCOPE_DEVICE || data.scope == SCOPE_QUEUEFAMILY)
        && (data.payloadSC == SC_WORKGROUP || data.guardSC == SC_WORKGROUP))
    {
        return SCOPE_WORKGROUP;
    }
    return data.scope;
}

class MemoryModelMessagePassingTestCase : public MemoryModelPatternBaseTestCase
{
public:
    using MemoryModelPatternBaseTestCase::MemoryModelPatternBaseTestCase;

protected:
    void initProgramBody(std::stringstream &css) const override
    {
        const uint32_t semAv  = m_data.coherent ? 0u : uint32_t(SEM_AVAILABLE);
        const uint32_t semVis = m_data.coherent ? 0u : uint32_t(SEM_VISIBLE);

        const DataType type = getBuffer("payload").type;
        const char *initValue = (type == DATA_TYPE_UINT || type == DATA_TYPE_UINT64) ?
                                "bufferCoord + (init0>>31)" :
                                "bufferCoord + (floatBitsToInt(float(init0))>>31)";
        css << "   " << gl_type("payload") << " init0 = " << gl_load("payload", "payloadPartnerCoord") << ";\n"
            << "   " << gl_store("payload", "payloadCoord", initValue) << ";\n";
        initSyncPair(css, semAv, semVis);
        css << "   r = " << gl_load("payload", "payloadPartnerCoord") << ";\n"
            << "   if (!skip && r != " << gl_type("payload") << "(partnerBufferCoord)) { " << gl_store("fail", "bufferCoord", "1") << "; }\n";
    }
};

class MemoryModelLoadBufferTestCase : public MemoryModelPatternBaseTestCase
{
public:
    using MemoryModelPatternBaseTestCase::MemoryModelPatternBaseTestCase;

protected:
    void initProgramBody(std::stringstream &css) const override
    {
        css << "   r = " << gl_load("payload", "payloadPartnerCoord") << ";\n";
        initSyncPair(css, 0u, 0u);
        css << "   if (!skip) { " << gl_store("payload", "payloadCoord", "bufferCoord") << "; }\n"
            << "   if (r != 0) { " << gl_store("fail", "bufferCoord", "1") << "; }\n";
    }
};

template <typename TestCaseType>
tcu::TestCaseGroup *createPairPatternTests(tcu::TestContext &testCtx, const char *groupName)
{
    struct TestGroupCase
    {
        const uint32_t value;
        const char *name;
    };

    static const TestGroupCase core11Cases[] = {
        // Supported by Vulkan1.1
        {1, "core11"},
        // Requires VK_KHR_vulkan_memory_model extension
        {0, "ext"},
    };

    static const TestGroupCase dtCases[] = {
        // uint32_t atomics
        {DATA_TYPE_UINT, "u32"},
        // uint64_t atomics
        {DATA_TYPE_UINT64, "u64"},
        // float32 atomics
        {DATA_TYPE_FLOAT32, "f32"},
        // float64 atomics
        {DATA_TYPE_FLOAT64, "f64"},
    };

    static const TestGroupCase cohCases[] = {
        // coherent payload variable
        {1, "coherent"},
        // noncoherent payload variable
        {0, "noncoherent"},
    };

    static const TestGroupCase stCases[] = {
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

    static const TestGroupCase rmwCases[] = {
        {0, "atomicwrite"},
        {1, "atomicrmw"},
    };

    static const TestGroupCase scopeCases[] = {
        {SCOPE_DEVICE, "device"},
        {SCOPE_QUEUEFAMILY, "queuefamily"},
        {SCOPE_WORKGROUP, "workgroup"},
        {SCOPE_SUBGROUP, "subgroup"},
    };

    static const TestGroupCase plCases[] = {
        // payload variable in non-local memory
        {0, "payload_nonlocal"},
        // payload variable in local memory
        {1, "payload_local"},
    };

    static const TestGroupCase pscCases[] = {
        // payload variable in buffer memory
        {SC_BUFFER, "buffer"},
        // payload variable in image memory
        {SC_IMAGE, "image"},
        // payload variable in workgroup memory
        {SC_WORKGROUP, "workgroup"},
        // payload variable in physical storage buffer memory
        {SC_PHYSBUFFER, "physbuffer"},
    };

    static const TestGroupCase glCases[] = {
        // guard variable in non-local memory
        {0, "guard_nonlocal"},
        // guard variable in local memory
        {1, "guard_local"},
    };

    static const TestGroupCase gscCases[] = {
        // guard variable in buffer memory
        {SC_BUFFER, "buffer"},
        // guard variable in image memory
        {SC_IMAGE, "image"},
        // guard variable in workgroup memory
        {SC_WORKGROUP, "workgroup"},
        // guard variable in physical storage buffer memory
        {SC_PHYSBUFFER, "physbuffer"},
    };

    static const TestGroupCase stageCases[] = {
        {STAGE_COMPUTE, "comp"},
        {STAGE_VERTEX, "vert"},
        {STAGE_FRAGMENT, "frag"},
    };

    de::MovePtr<tcu::TestCaseGroup> ttGroup(new tcu::TestCaseGroup(testCtx, groupName));

    for (int core11Ndx = 0; core11Ndx < DE_LENGTH_OF_ARRAY(core11Cases); core11Ndx++)
    {
        de::MovePtr<tcu::TestCaseGroup> core11Group(new tcu::TestCaseGroup(testCtx, core11Cases[core11Ndx].name));
        for (int dtNdx = 0; dtNdx < DE_LENGTH_OF_ARRAY(dtCases); dtNdx++)
        {
            de::MovePtr<tcu::TestCaseGroup> dtGroup(new tcu::TestCaseGroup(testCtx, dtCases[dtNdx].name));
            for (int cohNdx = 0; cohNdx < DE_LENGTH_OF_ARRAY(cohCases); cohNdx++)
            {
                de::MovePtr<tcu::TestCaseGroup> cohGroup(new tcu::TestCaseGroup(testCtx, cohCases[cohNdx].name));
                for (int stNdx = 0; stNdx < DE_LENGTH_OF_ARRAY(stCases); stNdx++)
                {
                    de::MovePtr<tcu::TestCaseGroup> stGroup(new tcu::TestCaseGroup(testCtx, stCases[stNdx].name));
                    for (int rmwNdx = 0; rmwNdx < DE_LENGTH_OF_ARRAY(rmwCases); rmwNdx++)
                    {
                        de::MovePtr<tcu::TestCaseGroup> rmwGroup(
                            new tcu::TestCaseGroup(testCtx, rmwCases[rmwNdx].name));
                        for (int scopeNdx = 0; scopeNdx < DE_LENGTH_OF_ARRAY(scopeCases); scopeNdx++)
                        {
                            de::MovePtr<tcu::TestCaseGroup> scopeGroup(
                                new tcu::TestCaseGroup(testCtx, scopeCases[scopeNdx].name));
                            for (int plNdx = 0; plNdx < DE_LENGTH_OF_ARRAY(plCases); plNdx++)
                            {
                                de::MovePtr<tcu::TestCaseGroup> plGroup(
                                    new tcu::TestCaseGroup(testCtx, plCases[plNdx].name));
                                for (int pscNdx = 0; pscNdx < DE_LENGTH_OF_ARRAY(pscCases); pscNdx++)
                                {
                                    de::MovePtr<tcu::TestCaseGroup> pscGroup(
                                        new tcu::TestCaseGroup(testCtx, pscCases[pscNdx].name));
                                    for (int glNdx = 0; glNdx < DE_LENGTH_OF_ARRAY(glCases); glNdx++)
                                    {
                                        de::MovePtr<tcu::TestCaseGroup> glGroup(
                                            new tcu::TestCaseGroup(testCtx, glCases[glNdx].name));
                                        for (int gscNdx = 0; gscNdx < DE_LENGTH_OF_ARRAY(gscCases); gscNdx++)
                                        {
                                            de::MovePtr<tcu::TestCaseGroup> gscGroup(
                                                new tcu::TestCaseGroup(testCtx, gscCases[gscNdx].name));
                                            for (int stageNdx = 0; stageNdx < DE_LENGTH_OF_ARRAY(stageCases);
                                                 stageNdx++)
                                            {
                                                CaseDef c = {
                                                    !!plCases[plNdx].value,               // bool payloadMemLocal;
                                                    !!glCases[glNdx].value,               // bool guardMemLocal;
                                                    !!cohCases[cohNdx].value,             // bool coherent;
                                                    !!core11Cases[core11Ndx].value,       // bool core11;
                                                    !!rmwCases[rmwNdx].value,             // bool atomicRMW;
                                                    (StorageClass)pscCases[pscNdx].value, // StorageClass payloadSC;
                                                    (StorageClass)gscCases[gscNdx].value, // StorageClass guardSC;
                                                    (Scope)scopeCases[scopeNdx].value,    // Scope scope;
                                                    (SyncType)stCases[stNdx].value,       // SyncType syncType;
                                                    (Stage)stageCases[stageNdx].value,    // Stage stage;
                                                    (DataType)dtCases[dtNdx].value,       // DataType dataType;
                                                };

                                                // Mustpass11 tests should only exercise things we expect to work on
                                                // existing implementations. Exclude noncoherent tests which require
                                                // new extensions, and assume atomic synchronization wouldn't work
                                                // (i.e. atomics may be implemented as relaxed atomics). Exclude
                                                // queuefamily scope which doesn't exist in Vulkan 1.1. Exclude
                                                // physical storage buffer which doesn't support the legacy decorations.
                                                if (c.core11 &&
                                                    (c.coherent == 0 || c.syncType == ST_FENCE_ATOMIC ||
                                                     c.syncType == ST_ATOMIC_FENCE ||
                                                     c.syncType == ST_ATOMIC_ATOMIC ||
                                                     c.dataType == DATA_TYPE_UINT64 ||
                                                     c.dataType == DATA_TYPE_FLOAT64 ||
                                                     c.scope == SCOPE_QUEUEFAMILY || c.payloadSC == SC_PHYSBUFFER ||
                                                     c.guardSC == SC_PHYSBUFFER))
                                                {
                                                    continue;
                                                }

                                                if (c.stage != STAGE_COMPUTE && c.scope == SCOPE_WORKGROUP)
                                                {
                                                    continue;
                                                }

                                                // Don't exercise local and non-local for workgroup memory
                                                // Also don't exercise workgroup memory for non-compute stages
                                                if (c.payloadSC == SC_WORKGROUP &&
                                                    (c.payloadMemLocal != 0 || c.stage != STAGE_COMPUTE))
                                                {
                                                    continue;
                                                }

                                                if (c.guardSC == SC_WORKGROUP &&
                                                    (c.guardMemLocal != 0 || c.stage != STAGE_COMPUTE))
                                                {
                                                    continue;
                                                }

                                                // No support for control/split barrier with larger than workgroup scope, or non-compute stages
                                                if ((c.syncType == ST_CONTROL_BARRIER ||
                                                     c.syncType == ST_CONTROL_AND_MEMORY_BARRIER ||
                                                     isSplitBarrier(c.syncType)) &&
                                                    (c.scope == SCOPE_DEVICE || c.scope == SCOPE_QUEUEFAMILY ||
                                                     c.stage != STAGE_COMPUTE))
                                                {
                                                    continue;
                                                }

                                                if (c.syncType == ST_SPLIT_CONTROL_BARRIER_AND_BARRIER &&
                                                    c.scope != SCOPE_WORKGROUP)
                                                {
                                                    continue;
                                                }

                                                // Limit RMW atomics to ST_ATOMIC_ATOMIC, just to reduce # of test cases
                                                if (c.atomicRMW && c.syncType != ST_ATOMIC_ATOMIC)
                                                {
                                                    continue;
                                                }

                                                // uint64/float32/float64 testing is primarily for atomics, so only test it for ST_ATOMIC_ATOMIC
                                                const bool atomicTesting = (c.dataType == DATA_TYPE_UINT64 ||
                                                                            c.dataType == DATA_TYPE_FLOAT32 ||
                                                                            c.dataType == DATA_TYPE_FLOAT64);
                                                if (atomicTesting && c.syncType != ST_ATOMIC_ATOMIC)
                                                {
                                                    continue;
                                                }

                                                // No 64-bit image types, so skip tests with both payload and guard in image memory
                                                if (c.dataType == DATA_TYPE_UINT64 && c.payloadSC == SC_IMAGE &&
                                                    c.guardSC == SC_IMAGE)
                                                {
                                                    continue;
                                                }

                                                // No support for atomic operations on 64-bit floating point images
                                                if (c.dataType == DATA_TYPE_FLOAT64 &&
                                                    (c.payloadSC == SC_IMAGE || c.guardSC == SC_IMAGE))
                                                {
                                                    continue;
                                                }
                                                // Control barrier tests don't use a guard variable, so only run them with gsc,gl==0
                                                if ((c.syncType == ST_CONTROL_BARRIER ||
                                                     c.syncType == ST_CONTROL_AND_MEMORY_BARRIER) &&
                                                    (c.guardSC != 0 || c.guardMemLocal != 0))
                                                {
                                                    continue;
                                                }
                                                gscGroup->addChild(new TestCaseType(testCtx, stageCases[stageNdx].name, c));
                                            }
                                            glGroup->addChild(gscGroup.release());
                                        }
                                        pscGroup->addChild(glGroup.release());
                                    }
                                    plGroup->addChild(pscGroup.release());
                                }
                                scopeGroup->addChild(plGroup.release());
                            }
                            rmwGroup->addChild(scopeGroup.release());
                        }
                        stGroup->addChild(rmwGroup.release());
                    }
                    cohGroup->addChild(stGroup.release());
                }
                dtGroup->addChild(cohGroup.release());
            }
            core11Group->addChild(dtGroup.release());
        }
        ttGroup->addChild(core11Group.release());
    }

    return ttGroup.release();
}

} // namespace

tcu::TestCaseGroup *createLoadBufferTests(tcu::TestContext &testCtx)
{
    return createPairPatternTests<MemoryModelLoadBufferTestCase>(testCtx, "write_after_read");
}

tcu::TestCaseGroup *createMessagePassingTests(tcu::TestContext &testCtx)
{
    return createPairPatternTests<MemoryModelMessagePassingTestCase>(testCtx, "message_passing");
}

} // namespace vkt::MemoryModel
