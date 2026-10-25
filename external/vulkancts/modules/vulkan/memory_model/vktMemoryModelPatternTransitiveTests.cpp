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
 * \brief Memory Model Pattern Transitive Tests
 *//*--------------------------------------------------------------------*/

#include "vktMemoryModelPatternTransitiveTests.hpp"
#include "vktMemoryModelPatternTestCase.hpp"
#include "vktMemoryModelPatternTestInstance.hpp"

#include "deDefs.h"
#include "deSharedPtr.hpp"
#include "tcuTestCase.hpp"
#include "vkBuilderUtil.hpp"
#include "vktTestCase.hpp"

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
};

struct CaseDef
{
    bool payloadMemLocal;
    bool guardMemLocal;
    bool coherent;
    StorageClass payloadSC;
    StorageClass guardSC;
    SyncType syncType;
    DataType dataType;
};

class MemoryModelMessagePassingTransitiveTestCase : public MemoryModelGlTestCase
{
public:
    MemoryModelMessagePassingTransitiveTestCase(tcu::TestContext &context, const char *name, const CaseDef &data);
    void checkSupport(Context &context) const override;
    void initPrograms(SourceCollections &programCollection) const override;
    TestInstance *createInstance(Context &context) const override;

protected:
    virtual void initProgramBody(std::stringstream &css) const = 0;
    CaseDef m_data;
};

MemoryModelMessagePassingTransitiveTestCase::MemoryModelMessagePassingTransitiveTestCase(tcu::TestContext &context, const char *name, const CaseDef &data)
    : MemoryModelGlTestCase(context, name, {
          { "payload", data.payloadSC, data.dataType, (data.coherent ? MEM_WORKGROUP : MEM_NONPRIVATE), data.payloadMemLocal },
          { "guard", data.guardSC, data.dataType, MEM_NONE, data.guardMemLocal },
          { "fail", SC_BUFFER, DATA_TYPE_UINT, MEM_NONE, true },
      })
    , m_data(data)
{
}

void MemoryModelMessagePassingTransitiveTestCase::checkSupport(Context &context) const
{
    if (!context.contextSupports(vk::ApiVersion(0, 1, 1, 0)))
        TCU_THROW(NotSupportedError, "Vulkan 1.1 not supported");

    if (!context.getVulkanMemoryModelFeatures().vulkanMemoryModel)
        TCU_THROW(NotSupportedError, "vulkanMemoryModel not supported");

    if (!context.getVulkanMemoryModelFeatures().vulkanMemoryModelDeviceScope)
        TCU_THROW(NotSupportedError, "vulkanMemoryModelDeviceScope not supported");

    if (!context.getVulkanMemoryModelFeatures().vulkanMemoryModelAvailabilityVisibilityChains)
        TCU_THROW(NotSupportedError, "vulkanMemoryModelAvailabilityVisibilityChains not supported");

    checkSupportBuffers(context);
    checkSupportBufferAtomics(context, "guard");
}

void MemoryModelMessagePassingTransitiveTestCase::initPrograms(SourceCollections &programCollection) const
{
    std::stringstream css;
    css << "#version 450 core\n"
           "#pragma use_vulkan_memory_model\n"
           "#extension GL_KHR_shader_subgroup_basic : enable\n"
           "#extension GL_KHR_shader_subgroup_shuffle : enable\n"
           "#extension GL_KHR_shader_subgroup_ballot : enable\n"
           "#extension GL_KHR_memory_scope_semantics : enable\n"
           "#extension GL_ARB_gpu_shader_int64 : enable\n"
           "#extension GL_EXT_buffer_reference : enable\n"
           "// DIM/NUM_WORKGROUP_EACH_DIM overriden by spec constants\n"
           "layout(constant_id = 0) const int DIM = 1;\n"
           "layout(constant_id = 1) const int NUM_WORKGROUP_EACH_DIM = 1;\n"
           "shared bool sharedSkip;\n"
        << gl_layoutStructs()
        << "layout(local_size_x_id = 0, local_size_y_id = 0, local_size_z = 1) in;\n"
        << gl_layout()
        << "void main()\n"
           "{\n"
           "   bool pass = true;\n"
           "   bool skip = false;\n"
           "   sharedSkip = false;\n"
        << "   " << gl_type("payload") << " r = 0;\n"
        << gl_layoutReferences()
        << gl_layoutCoordinates(STAGE_COMPUTE, SCOPE_DEVICE)
        << "   ivec2 globalId00          = ivec2(DIM) * ivec2(gl_WorkGroupID.xy);\n"
           "   ivec2 partnerGlobalId00   = ivec2(DIM) * (ivec2(NUM_WORKGROUP_EACH_DIM-1) - ivec2(gl_WorkGroupID.xy));\n"
           "   uint bufferCoord00        = globalId00.y * DIM*NUM_WORKGROUP_EACH_DIM + globalId00.x;\n"
           "   uint partnerBufferCoord00 = partnerGlobalId00.y * DIM*NUM_WORKGROUP_EACH_DIM + partnerGlobalId00.x;\n"
           "   ivec2 imageCoord00        = globalId00;\n"
           "   ivec2 partnerImageCoord00 = partnerGlobalId00;\n"
        << gl_layoutBufferCoordinates()
        << gl_layoutBufferCoordinates00();

    initProgramBody(css);

    css << "}\n";

    const vk::ShaderBuildOptions buildOptions(programCollection.usedVulkanVersion, vk::SPIRV_VERSION_1_3, 0u);
    programCollection.glslSources.add("test") << glu::ComputeSource(css.str()) << buildOptions;
}

TestInstance *MemoryModelMessagePassingTransitiveTestCase::createInstance(Context &context) const
{
    return new MemoryModelTestInstance(context, STAGE_COMPUTE, m_buffers);
}

class MemoryModelMp3TestCase : public MemoryModelMessagePassingTransitiveTestCase
{
public:
    using MemoryModelMessagePassingTransitiveTestCase::MemoryModelMessagePassingTransitiveTestCase;

protected:
    void initProgramBody(std::stringstream &css) const override
    {
        const uint32_t semAv = m_data.coherent ? 0u : uint32_t(SEM_AVAILABLE);
        const uint32_t semPayload = getBufferSemantics("payload");
        const uint32_t semGuard = getBufferSemantics("guard");
        const uint32_t semRel = SEM_RELEASE | semPayload | SEM_AVAILABLE;
        const uint32_t semAcq = SEM_ACQUIRE | semPayload | SEM_VISIBLE;
        const uint32_t semAcqRel = SEM_ACQUIRE_RELEASE | semPayload;

        const DataType type = getBuffer("payload").type;
        const char *initValue = (type == DATA_TYPE_UINT || type == DATA_TYPE_UINT64) ?
                                "bufferCoord + (init0>>31)" :
                                "bufferCoord + (floatBitsToInt(float(init0))>>31)";

        css << "   " << gl_type("payload") << " init0 = " << gl_load("payload", "payloadPartnerCoord") << ";\n"
            << "   " << gl_store("payload", "payloadCoord", initValue) << ";\n"
            << "   " << gl_controlBarrier(SCOPE_WORKGROUP, SCOPE_WORKGROUP, semAcqRel | semAv) << ";\n"
            << "   if (all(equal(gl_LocalInvocationID.xy, ivec2(0,0)))) {\n";

        if (m_data.syncType == ST_ATOMIC_ATOMIC || m_data.syncType == ST_ATOMIC_FENCE)
            css << "       " << gl_atomicStore("guard", "guardCoord", "1u", SCOPE_DEVICE, semRel) << ";\n";
        else
            css << "       " << gl_memoryBarrier(SCOPE_DEVICE, semRel | semGuard) << ";\n"
                << "       " << gl_atomicStore("guard", "guardCoord", "1u", SCOPE_DEVICE, 0) << ";\n";

        css << "   }\n";

        if (m_data.syncType == ST_ATOMIC_ATOMIC || m_data.syncType == ST_FENCE_ATOMIC)
            css << "       skip = " << gl_atomicLoad("guard", "guardPartnerCoord00", SCOPE_DEVICE, semAcq) << " == 0;\n";
        else
            css << "       skip = " << gl_atomicLoad("guard", "guardPartnerCoord00", SCOPE_DEVICE, 0) << " == 0;\n"
                << "       " << gl_memoryBarrier(SCOPE_DEVICE, semAcq | semGuard) << ";\n";

        css << "   r = " << gl_load("payload", "payloadPartnerCoord") << ";\n"
            << "   if (!skip && r != " << gl_type("payload") << "(partnerBufferCoord)) { " << gl_store("fail", "bufferCoord", "1") << "; }\n";
    }
};

class MemoryModelMp4TestCase : public MemoryModelMessagePassingTransitiveTestCase
{
public:
    using MemoryModelMessagePassingTransitiveTestCase::MemoryModelMessagePassingTransitiveTestCase;

protected:
    void initProgramBody(std::stringstream &css) const override
    {
        const uint32_t semAv = m_data.coherent ? 0u : uint32_t(SEM_AVAILABLE);
        const uint32_t semVis = m_data.coherent ? 0u : uint32_t(SEM_VISIBLE);
        const uint32_t semPayload = getBufferSemantics("payload");
        const uint32_t semGuard = getBufferSemantics("guard");
        const uint32_t semRel     = SEM_RELEASE | semPayload | SEM_AVAILABLE;
        const uint32_t semAcq     = SEM_ACQUIRE | semPayload | SEM_VISIBLE;

        // Additional SEM_SHARED semantics orders accesses to sharedSkip
        const uint32_t semAcqRel = SEM_ACQUIRE_RELEASE | semPayload | SEM_SHARED;

        const DataType type = getBuffer("payload").type;
        const char *initValue = (type == DATA_TYPE_UINT || type == DATA_TYPE_UINT64) ?
                                "bufferCoord + (init0>>31)" :
                                "bufferCoord + (floatBitsToInt(float(init0))>>31)";

        css << "   " << gl_type("payload") << " init0 = " << gl_load("payload", "payloadPartnerCoord") << ";\n"
            << "   " << gl_store("payload", "payloadCoord", initValue) << ";\n"
            << "   " << gl_controlBarrier(SCOPE_WORKGROUP, SCOPE_WORKGROUP, semAcqRel | semAv) << ";\n"
            << "   if (all(equal(gl_LocalInvocationID.xy, ivec2(0,0)))) {\n";

        if (m_data.syncType == ST_ATOMIC_ATOMIC || m_data.syncType == ST_ATOMIC_FENCE)
            css << "       " << gl_atomicStore("guard", "guardCoord", "1u", SCOPE_DEVICE, semRel) << ";\n";
        else
            css << "       " << gl_memoryBarrier(SCOPE_DEVICE, semRel | semGuard) << ";\n"
                << "       " << gl_atomicStore("guard", "guardCoord", "1u", SCOPE_DEVICE, 0) << ";\n";

        if (m_data.syncType == ST_ATOMIC_ATOMIC || m_data.syncType == ST_FENCE_ATOMIC)
            css << "       skip = " << gl_atomicLoad("guard", "guardPartnerCoord00", SCOPE_DEVICE, semAcq) << " == 0;\n";
        else
            css << "       skip = " << gl_atomicLoad("guard", "guardPartnerCoord00", SCOPE_DEVICE, 0) << " == 0;\n"
                << "       " << gl_memoryBarrier(SCOPE_DEVICE, semAcq | semGuard) << ";\n";

        css << "       sharedSkip = skip;\n"
            << "   }\n"
            << "   " << gl_controlBarrier(SCOPE_WORKGROUP, SCOPE_WORKGROUP, semAcqRel | semVis) << ";\n"
            << "   skip = sharedSkip;\n"
            << "   r = " << gl_load("payload", "payloadPartnerCoord") << ";\n"
            << "   if (!skip && r != " << gl_type("payload") << "(partnerBufferCoord)) { " << gl_store("fail", "bufferCoord", "1") << "; }\n";
    }
};

} // namespace

tcu::TestCaseGroup *createTransitiveTests(tcu::TestContext &testCtx)
{
    struct TestGroupCase
    {
        const uint32_t value;
        const char *name;
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
        // guard variable in physical storage buffer memory
        {SC_PHYSBUFFER, "physbuffer"},
    };

    de::MovePtr<tcu::TestCaseGroup> transGroup(new tcu::TestCaseGroup(testCtx, "transitive"));
    for (int cohNdx = 0; cohNdx < DE_LENGTH_OF_ARRAY(cohCases); cohNdx++)
    {
        de::MovePtr<tcu::TestCaseGroup> cohGroup(new tcu::TestCaseGroup(testCtx, cohCases[cohNdx].name));
        for (int stNdx = 0; stNdx < DE_LENGTH_OF_ARRAY(stCases); stNdx++)
        {
            de::MovePtr<tcu::TestCaseGroup> stGroup(new tcu::TestCaseGroup(testCtx, stCases[stNdx].name));
            for (int plNdx = 0; plNdx < DE_LENGTH_OF_ARRAY(plCases); plNdx++)
            {
                de::MovePtr<tcu::TestCaseGroup> plGroup(new tcu::TestCaseGroup(testCtx, plCases[plNdx].name));
                for (int pscNdx = 0; pscNdx < DE_LENGTH_OF_ARRAY(pscCases); pscNdx++)
                {
                    de::MovePtr<tcu::TestCaseGroup> pscGroup(new tcu::TestCaseGroup(testCtx, pscCases[pscNdx].name));
                    for (int glNdx = 0; glNdx < DE_LENGTH_OF_ARRAY(glCases); glNdx++)
                    {
                        de::MovePtr<tcu::TestCaseGroup> glGroup(new tcu::TestCaseGroup(testCtx, glCases[glNdx].name));
                        for (int gscNdx = 0; gscNdx < DE_LENGTH_OF_ARRAY(gscCases); gscNdx++)
                        {
                            de::MovePtr<tcu::TestCaseGroup> gscGroup(
                                new tcu::TestCaseGroup(testCtx, gscCases[gscNdx].name));

                            CaseDef c = {
                                !!plCases[plNdx].value,               // bool payloadMemLocal;
                                !!glCases[glNdx].value,               // bool guardMemLocal;
                                !!cohCases[cohNdx].value,             // bool coherent;
                                (StorageClass)pscCases[pscNdx].value, // StorageClass payloadSC;
                                (StorageClass)gscCases[gscNdx].value, // StorageClass guardSC;
                                (SyncType)stCases[stNdx].value,       // SyncType syncType;
                                DATA_TYPE_UINT,                       // DataType dataType;
                            };

                            gscGroup->addChild(new MemoryModelMp3TestCase(testCtx, "nontransvis", c));
                            gscGroup->addChild(new MemoryModelMp4TestCase(testCtx, "transvis", c));

                            glGroup->addChild(gscGroup.release());
                        }
                        pscGroup->addChild(glGroup.release());
                    }
                    plGroup->addChild(pscGroup.release());
                }
                stGroup->addChild(plGroup.release());
            }
            cohGroup->addChild(stGroup.release());
        }
        transGroup->addChild(cohGroup.release());
    }

    return transGroup.release();
}

} // namespace vkt::MemoryModel
