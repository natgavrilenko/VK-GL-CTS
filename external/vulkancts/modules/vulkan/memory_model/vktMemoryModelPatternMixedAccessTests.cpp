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
 * \brief Memory Model Pattern Mixed Access Tests
 *//*--------------------------------------------------------------------*/

#include "vktMemoryModelPatternMixedAccessTests.hpp"
#include "vktMemoryModelPatternTestCase.hpp"
#include "vktMemoryModelPatternTestInstance.hpp"

#include "deDefs.h"
#include "deSharedPtr.hpp"
#include "tcuTestCase.hpp"
#include "vkBuilderUtil.hpp"
#include "vktTestCase.hpp"

#include <algorithm>
#include <sstream>
#include <vector>

namespace vkt::MemoryModel
{
namespace
{
using namespace vk;

struct CaseDef
{
    bool core11;
    bool payloadMemLocal;
    StorageClass payloadSC;
    DataType payloadType;
    Scope syncScope;
    Scope visScope;
    Stage stage;
};

class MemoryModelPatternMixedAccessTestCase : public MemoryModelGlTestCase
{
public:
    MemoryModelPatternMixedAccessTestCase(tcu::TestContext &context, const char *name, const CaseDef &data);
    void checkSupport(Context &context) const override;
    void initPrograms(SourceCollections &programCollection) const override;
    TestInstance *createInstance(Context &context) const override;

private:
    static std::vector<BufferDef> createBuffers(const CaseDef &data);
    static GlMemQualifier getMemQualifier(const CaseDef &data);
    Scope getMappingScope() const;
    CaseDef m_data;
};

MemoryModelPatternMixedAccessTestCase::MemoryModelPatternMixedAccessTestCase(tcu::TestContext &context, const char *name, const CaseDef &data)
    : MemoryModelGlTestCase(context, name, createBuffers(data)), m_data(data)
{
}

std::vector<BufferDef> MemoryModelPatternMixedAccessTestCase::createBuffers(const CaseDef &data)
{
    return std::vector<BufferDef>{
        { "payload", data.payloadSC, data.payloadType, getMemQualifier(data), data.payloadMemLocal },
        { "fail", SC_BUFFER, DATA_TYPE_UINT, MEM_NONE, true }
    };
}

GlMemQualifier MemoryModelPatternMixedAccessTestCase::getMemQualifier(const CaseDef &data)
{
    if (data.core11)
        return MEM_COHERENT;
    switch (data.visScope)
    {
        case SCOPE_DEVICE:
            return MEM_DEVICE;
        case SCOPE_QUEUEFAMILY:
            return MEM_QUEUEFAMILY;
        case SCOPE_WORKGROUP:
            return MEM_WORKGROUP;
        case SCOPE_SUBGROUP:
            return MEM_SUBGROUP;
        case SCOPE_INVOCATION:
            return MEM_NONPRIVATE;
        default:
            DE_ASSERT(0);
            return MEM_NONE;
    }
}

void MemoryModelPatternMixedAccessTestCase::checkSupport(Context &context) const
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
        checkSupportStageSubgroupFeatures(context, m_data.stage);

    checkSupportBuffers(context);
    checkSupportBufferAtomics(context, "payload");
    checkSupportStageAtomics(context, m_data.stage);
}

void MemoryModelPatternMixedAccessTestCase::initPrograms(SourceCollections &programCollection) const
{
    const Scope syncScope = m_data.syncScope;
    const Scope mappingScope = getMappingScope();
    const char *typeStr = gl_type("payload");
    const bool hasFloats = std::any_of(m_buffers.cbegin(), m_buffers.cend(), [](BufferDef b) {
        return b.type == DATA_TYPE_FLOAT32 || b.type == DATA_TYPE_FLOAT64;
    });

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
    css << "#extension GL_KHR_shader_subgroup_basic : enable\n"
           "#extension GL_KHR_shader_subgroup_shuffle : enable\n"
           "#extension GL_KHR_shader_subgroup_ballot : enable\n"
           "#extension GL_KHR_memory_scope_semantics : enable\n"
           "#extension GL_ARB_gpu_shader_int64 : enable\n"
           "#extension GL_EXT_buffer_reference : enable\n";

    if (!m_data.core11 && m_data.payloadSC == SC_WORKGROUP)
    {
        const bool hasVisibility = m_data.visScope != SCOPE_INVOCATION;
        const bool hasDeviceVisibility = m_data.visScope == SCOPE_DEVICE;
        css << "#extension GL_EXT_spirv_intrinsics : enable\n";
        css << "spirv_instruction(" << (hasDeviceVisibility ? "capabilities = [5346], " : "") << "id = 61) "
            << typeStr << " loadPayload(spirv_by_reference " << typeStr << " p, spirv_literal int memoryAccess"
            << (hasVisibility ? ", int scope" : "") << ");\n";
    }

    css << "// DIM/NUM_WORKGROUP_EACH_DIM overriden by spec constants\n"
           "layout(constant_id = 0) const int DIM = 1;\n"
           "layout(constant_id = 1) const int NUM_WORKGROUP_EACH_DIM = 1;\n"
        << gl_layoutStructs();

    if (m_data.stage == STAGE_COMPUTE)
    {
        css << "layout(local_size_x_id = 0, local_size_y_id = 0, local_size_z = 1) in;\n";
    }
    css << gl_layout();

    css << "void main()\n"
           "{\n"
        << "   " << typeStr << " r0, r1, init;\n"
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

        // Value `bufferCoord + 1` avoids matching the initial value if bufferCoord is zero
        const char *init = hasFloats ? "bufferCoord + 1 + (floatBitsToInt(float(init))>>31)" :
                           "bufferCoord + 1 + (init>>31)";
        std::string loadOperands = "0x20";
        if (m_data.visScope != SCOPE_INVOCATION)
            loadOperands = std::string("0x30, ") + gl_scope(m_data.visScope);

        if (!m_data.core11 && m_data.payloadSC == SC_WORKGROUP)
            css << "   init = loadPayload(payload.x[payloadPartnerCoord], " << loadOperands << ");\n";
        else
            css << "   init = " << gl_load("payload", "payloadPartnerCoord") << ";\n";

        css << "   " << gl_atomicStore("payload", "payloadCoord", init, syncScope, 0) << ";\n"
            << "   r0 = " << gl_atomicLoad("payload", "payloadPartnerCoord", syncScope, 0) << ";\n"
            << "   if (r0 == " << typeStr << "(partnerBufferCoord + 1)) {\n";

        if (!m_data.core11 && m_data.payloadSC == SC_WORKGROUP)
            css << "      r1 = loadPayload(payload.x[payloadPartnerCoord], " << loadOperands << ");\n";
        else
            css << "      r1 = " << gl_load("payload", "payloadPartnerCoord") << ";\n";

        css << "      if (r1 != " << typeStr << "(partnerBufferCoord + 1)) {\n"
            << "         " << gl_store("fail", "bufferCoord", "1") << ";\n"
            << "      }\n"
            << "   }\n"
            << "}\n";

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

TestInstance *MemoryModelPatternMixedAccessTestCase::createInstance(Context &context) const
{
    return new MemoryModelTestInstance(context, m_data.stage, m_buffers);
}

Scope MemoryModelPatternMixedAccessTestCase::getMappingScope() const
{
    const bool hasSharedBuffer = std::any_of(m_buffers.cbegin(), m_buffers.cend(),
                                             [](const BufferDef &buffer) { return buffer.sc == SC_WORKGROUP; });
    if ((m_data.syncScope == SCOPE_DEVICE || m_data.syncScope == SCOPE_QUEUEFAMILY) && hasSharedBuffer)
    {
        return SCOPE_WORKGROUP;
    }
    return m_data.syncScope;
}

} // namespace

tcu::TestCaseGroup *createMixedAccessTests(tcu::TestContext &testCtx)
{
    struct TestGroupCase
    {
        const uint32_t value;
        const char *name;
    };

    static const TestGroupCase core11Cases[] = {
        {1, "core11"},
        {0, "ext"},
    };

    static const TestGroupCase typeCases[] = {
        {DATA_TYPE_UINT, "u32"},
        {DATA_TYPE_UINT64, "u64"},
        {DATA_TYPE_FLOAT32, "f32"},
        {DATA_TYPE_FLOAT64, "f64"},
    };

    static const TestGroupCase localCases[] = {
        {0, "payload_nonlocal"},
        {1, "payload_local"},
    };

    static const TestGroupCase scCases[] = {
        {SC_BUFFER, "buffer"},
        {SC_IMAGE, "image"},
        {SC_WORKGROUP, "workgroup"},
        {SC_PHYSBUFFER, "physbuffer"},
    };

    static const TestGroupCase syncScopeCases[] = {
        {SCOPE_DEVICE, "device"},
        {SCOPE_QUEUEFAMILY, "queuefamily"},
        {SCOPE_WORKGROUP, "workgroup"},
        {SCOPE_SUBGROUP, "subgroup"},
    };

    static const TestGroupCase visScopeCases[] = {
        {SCOPE_INVOCATION, "nonprivate"},
        {SCOPE_SUBGROUP, "vissubgroup"},
        {SCOPE_WORKGROUP, "visworkgroup"},
        {SCOPE_QUEUEFAMILY, "visqueuefamily"},
        {SCOPE_DEVICE, "visdevice"},
    };

    static const TestGroupCase stageCases[] = {
        {STAGE_COMPUTE, "comp"},
        {STAGE_VERTEX, "vert"},
        {STAGE_FRAGMENT, "frag"},
    };

    de::MovePtr<tcu::TestCaseGroup> ttGroup(new tcu::TestCaseGroup(testCtx, "mixed_access"));

    for (int core11Ndx = 0; core11Ndx < DE_LENGTH_OF_ARRAY(core11Cases); core11Ndx++)
    {
        de::MovePtr<tcu::TestCaseGroup> core11Group(new tcu::TestCaseGroup(testCtx, core11Cases[core11Ndx].name));
        for (int typeNdx = 0; typeNdx < DE_LENGTH_OF_ARRAY(typeCases); typeNdx++)
        {
            de::MovePtr<tcu::TestCaseGroup> typeGroup(new tcu::TestCaseGroup(testCtx, typeCases[typeNdx].name));
            for (int localNdx = 0; localNdx < DE_LENGTH_OF_ARRAY(localCases); localNdx++)
            {
                de::MovePtr<tcu::TestCaseGroup> localGroup(new tcu::TestCaseGroup(testCtx, localCases[localNdx].name));
                for (int scNdx = 0; scNdx < DE_LENGTH_OF_ARRAY(scCases); scNdx++)
                {
                    de::MovePtr<tcu::TestCaseGroup> scGroup(new tcu::TestCaseGroup(testCtx, scCases[scNdx].name));
                    for (int syncScopeNdx = 0; syncScopeNdx < DE_LENGTH_OF_ARRAY(syncScopeCases); syncScopeNdx++)
                    {
                        de::MovePtr<tcu::TestCaseGroup> syncScopeGroup(
                            new tcu::TestCaseGroup(testCtx, syncScopeCases[syncScopeNdx].name));
                        for (int visScopeNdx = 0; visScopeNdx < DE_LENGTH_OF_ARRAY(visScopeCases); visScopeNdx++)
                        {
                            de::MovePtr<tcu::TestCaseGroup> visScopeGroup(
                                new tcu::TestCaseGroup(testCtx, visScopeCases[visScopeNdx].name));
                            for (int stageNdx = 0; stageNdx < DE_LENGTH_OF_ARRAY(stageCases); stageNdx++)
                            {
                                CaseDef c = {
                                    !!core11Cases[core11Ndx].value,             // bool core11;
                                    !!localCases[localNdx].value,               // bool payloadMemLocal;
                                    (StorageClass)scCases[scNdx].value,         // StorageClass payloadSC;
                                    (DataType)typeCases[typeNdx].value,         // DataType payloadType;
                                    (Scope)syncScopeCases[syncScopeNdx].value,  // Scope syncScope;
                                    (Scope)visScopeCases[visScopeNdx].value,    // Scope visScope;
                                    (Stage)stageCases[stageNdx].value,          // Stage stage;
                                };

                                // Mustpass11 tests should only exercise things we expect to work on
                                // existing implementations. Exclude noncoherent tests which require
                                // new extensions. Exclude queuefamily scope which doesn't exist in
                                // Vulkan 1.1. Exclude physical storage buffer which doesn't support
                                // the legacy decorations.
                                if (c.core11 &&
                                    (c.visScope != SCOPE_DEVICE || c.payloadType == DATA_TYPE_UINT64 ||
                                     c.payloadType == DATA_TYPE_FLOAT64 || c.syncScope == SCOPE_QUEUEFAMILY ||
                                     c.payloadSC == SC_PHYSBUFFER))
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
                                if (c.payloadSC == SC_IMAGE &&
                                    (c.payloadType == DATA_TYPE_UINT64 || c.payloadType == DATA_TYPE_FLOAT64))
                                {
                                    continue;
                                }
                                visScopeGroup->addChild(new MemoryModelPatternMixedAccessTestCase(testCtx, stageCases[stageNdx].name, c));
                            }
                            syncScopeGroup->addChild(visScopeGroup.release());
                        }
                        scGroup->addChild(syncScopeGroup.release());
                    }
                    localGroup->addChild(scGroup.release());
                }
                typeGroup->addChild(localGroup.release());
            }
            core11Group->addChild(typeGroup.release());
        }
        ttGroup->addChild(core11Group.release());
    }

    return ttGroup.release();
}

} // namespace vkt::MemoryModel
