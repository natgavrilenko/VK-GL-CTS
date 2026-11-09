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
 * \brief Memory Model Availability-Visibility Chains Tests
 *//*--------------------------------------------------------------------*/

#include "vktMemoryModelAvVisChainsTests.hpp"
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
    bool transitiveVis;
};

std::vector<BufferDef> createBuffers(const CaseDef &data)
{
    const MemQualifier payloadQualifier = data.coherent ? MEM_WORKGROUP : MEM_NONPRIVATE;
    const MemQualifier guardQualifier   = data.guardSC == SC_IMAGE ? payloadQualifier : MEM_NONE;
    return std::vector<BufferDef>{
        {"payload", payloadQualifier, data.payloadSC, data.dataType, data.payloadMemLocal},
        {"guard", guardQualifier, data.guardSC, data.dataType, data.guardMemLocal},
        {"fail", MEM_NONE, SC_BUFFER, DATA_TYPE_UINT, true},
    };
}

class AvVisChainsTestCase : public MemoryModelTestCase
{
public:
    AvVisChainsTestCase(tcu::TestContext &context, const char *name, const CaseDef &data);
    void checkSupport(Context &context) const override;

private:
    std::string declareSharedVariables() const override;
    std::string initProgram() const override;
    const CaseDef m_data;
};

AvVisChainsTestCase::AvVisChainsTestCase(tcu::TestContext &context, const char *name, const CaseDef &data)
    : MemoryModelTestCase(context, name, {false, STAGE_COMPUTE, SCOPE_DEVICE, 1}, createBuffers(data))
    , m_data(data)
{
}

void AvVisChainsTestCase::checkSupport(Context &context) const
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

std::string AvVisChainsTestCase::declareSharedVariables() const
{
    if (m_data.transitiveVis)
        return std::string{"shared bool sharedSkip;\n"};
    return {};
}

std::string AvVisChainsTestCase::initProgram() const
{
    const DataType type = getBuffer("payload").type;
    const char *typeStr = gl_type(type);

    const uint32_t semAv      = m_data.coherent ? 0u : uint32_t(SEM_AVAILABLE);
    const uint32_t semVis     = m_data.coherent ? 0u : uint32_t(SEM_VISIBLE);
    const uint32_t semPayload = getBufferSemantics("payload");
    const uint32_t semGuard   = getBufferSemantics("guard");
    const uint32_t semRel     = SEM_RELEASE | semPayload | SEM_AVAILABLE;
    const uint32_t semAcq     = SEM_ACQUIRE | semPayload | SEM_VISIBLE;

    // Additional SEM_SHARED semantics orders accesses to sharedSkip
    const uint32_t semAcqRel = SEM_ACQUIRE_RELEASE | semPayload | (m_data.transitiveVis ? SEM_SHARED : 0);

    std::ostringstream css;

    css << "   bool skip = false;\n"
        << "   uint value = invocationId;\n"
        << "   " << typeStr << " r = 0;\n"
        << "   " << typeStr << " expected = " << typeStr << "(invocationIdPartner1);\n";

    css << initLayoutWorkgroupCoordinates("guard");

    if (m_data.transitiveVis)
    {
        css << "   sharedSkip = false;\n";
    }

    // Prefetch the partner's payload; this adds 0 because every stored value is below 2^31
    css << "   value += " << gl_toUint(type, gl_load("payload", "payloadCoordPartner1")) << " >> 31;\n"
        << "   " << gl_store("payload", "payloadCoord", "value") << ";\n"
        << "   " << gl_controlBarrier(SCOPE_WORKGROUP, SCOPE_WORKGROUP, semAcqRel | semAv) << ";\n"
        << "   if (all(equal(gl_LocalInvocationID.xy, ivec2(0,0)))) {\n";

    if (m_data.syncType == ST_ATOMIC_ATOMIC || m_data.syncType == ST_ATOMIC_FENCE)
        css << "       " << gl_atomicStore("guard", "guardCoord", "1u", SCOPE_DEVICE, semRel) << ";\n";
    else
        css << "       " << gl_memoryBarrier(SCOPE_DEVICE, semRel | semGuard) << ";\n"
            << "       " << gl_atomicStore("guard", "guardCoord", "1u", SCOPE_DEVICE, 0) << ";\n";

    if (!m_data.transitiveVis)
    {
        css << "   }\n";
    }

    if (m_data.syncType == ST_ATOMIC_ATOMIC || m_data.syncType == ST_FENCE_ATOMIC)
        css << "       skip = " << gl_atomicLoad("guard", "guardGroupCoordPartner1", SCOPE_DEVICE, semAcq)
            << " == 0;\n";
    else
        css << "       skip = " << gl_atomicLoad("guard", "guardGroupCoordPartner1", SCOPE_DEVICE, 0) << " == 0;\n"
            << "       " << gl_memoryBarrier(SCOPE_DEVICE, semAcq | semGuard) << ";\n";

    if (m_data.transitiveVis)
    {
        css << "       sharedSkip = skip;\n"
            << "   }\n"
            << "   " << gl_controlBarrier(SCOPE_WORKGROUP, SCOPE_WORKGROUP, semAcqRel | semVis) << ";\n"
            << "   skip = sharedSkip;\n";
    }

    css << "   r = " << gl_load("payload", "payloadCoordPartner1") << ";\n"
        << "   if (!idle && !skip && r != expected) { " << gl_store("fail", "failCoord", "1") << "; }\n";

    return css.str();
}

} // namespace

tcu::TestCaseGroup *createAvVisChainsTests(tcu::TestContext &testCtx)
{
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
        // guard variable in physical storage buffer memory
        {SC_PHYSBUFFER, "physbuffer"},
    };

    static const NamedValue<bool> transVisCases[] = {
        // destination invocation acquires
        {false, "nontransvis"},
        // invocation 0,0 acquires
        {true, "transvis"},
    };

    de::MovePtr<tcu::TestCaseGroup> transGroup(new tcu::TestCaseGroup(testCtx, "transitive"));
    for (const auto &[coh, st, pl, psc, gl, gsc, vis] :
         Combinations(cohCases, stCases, plCases, pscCases, glCases, gscCases, transVisCases))
    {
        CaseDef c = {
            pl.value,       // bool payloadMemLocal;
            gl.value,       // bool guardMemLocal;
            coh.value,      // bool coherent;
            psc.value,      // StorageClass payloadSC;
            gsc.value,      // StorageClass guardSC;
            st.value,       // SyncType syncType;
            DATA_TYPE_UINT, // DataType dataType;
            vis.value,      // bool transitiveVis;
        };
        addTestCase(*transGroup, {coh.name, st.name, pl.name, psc.name, gl.name, gsc.name},
                    new AvVisChainsTestCase(testCtx, vis.name, c));
    }

    return transGroup.release();
}

} // namespace vkt::MemoryModel
