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
 * \brief Memory Model Chains With Barrier Tests
 *//*--------------------------------------------------------------------*/

#include "vktMemoryModelChainsWithBarrierTests.hpp"
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
};

std::vector<VarDef> createVars(const CaseDef &data)
{
    const MemQualifier qualifier = data.coherent ? MEM_WORKGROUP : MEM_NONPRIVATE;
    return std::vector<VarDef>{
        {"payload", qualifier, data.payloadSC, DATA_TYPE_UINT, data.payloadMemLocal},
        {"guard", MEM_NONE, data.guardSC, DATA_TYPE_UINT, data.guardMemLocal},
        {"fail", MEM_NONE, SC_BUFFER, DATA_TYPE_UINT, true},
    };
}

class ChainsWithBarrierTestCase : public MemoryModelTestCase
{
public:
    ChainsWithBarrierTestCase(tcu::TestContext &context, const char *name, const CaseDef &data);
    void checkSupport(Context &context) const override;

protected:
    const CaseDef m_data;
};

ChainsWithBarrierTestCase::ChainsWithBarrierTestCase(tcu::TestContext &context, const char *name, const CaseDef &data)
    : MemoryModelTestCase(context, name, {STAGE_COMPUTE, false}, createVars(data))
    , m_data(data)
{
}

void ChainsWithBarrierTestCase::checkSupport(Context &context) const
{
    if (!context.contextSupports(vk::ApiVersion(0, 1, 1, 0)))
        TCU_THROW(NotSupportedError, "Vulkan 1.1 not supported");

    if (!context.getVulkanMemoryModelFeatures().vulkanMemoryModel)
        TCU_THROW(NotSupportedError, "vulkanMemoryModel not supported");

    if (!context.getVulkanMemoryModelFeatures().vulkanMemoryModelAvailabilityVisibilityChains)
        TCU_THROW(NotSupportedError, "vulkanMemoryModelAvailabilityVisibilityChains not supported");

    if (!context.getVulkanMemoryModelFeatures().vulkanMemoryModelDeviceScope)
        TCU_THROW(NotSupportedError, "vulkanMemoryModelDeviceScope not supported");

    checkSupportVars(context);
    checkSupportVarAtomics(context, "guard");
}

class AvailabilityChainsTestCase : public ChainsWithBarrierTestCase
{
public:
    using ChainsWithBarrierTestCase::ChainsWithBarrierTestCase;

private:
    std::string initProgram() const override;
};

std::string AvailabilityChainsTestCase::initProgram() const
{
    const DataType type       = getVar("payload").type;
    const char *typeStr       = gl_type(type);
    const uint32_t semAv      = m_data.coherent ? 0u : uint32_t(SEM_AVAILABLE);
    const uint32_t semPayload = getVarSemantics("payload");
    const uint32_t semGuard   = getVarSemantics("guard");
    const uint32_t semRel     = SEM_RELEASE | semPayload | SEM_AVAILABLE;
    const uint32_t semAcq     = SEM_ACQUIRE | semPayload | SEM_VISIBLE;
    const uint32_t semAcqRel  = SEM_ACQUIRE_RELEASE | semPayload;

    std::ostringstream css;

    css << initPartnerCoordinates(SCOPE_DEVICE);
    css << "   ivec2 groupPosPartner = posPartner - posPartner % DIM;\n"
        << initVarCoordinates("groupPosPartner", "GroupCoordPartner");

    css << "   bool skip = false;\n"
        << "   uint value = invocationId;\n"
        << "   " << typeStr << " r = 0;\n"
        << "   " << typeStr << " expected = " << typeStr << "(invocationIdPartner);\n"
        << "   value += " << gl_toUint(type, gl_load("payload", "payloadCoordPartner")) << " >> 31;\n"
        << "   " << gl_store("payload", "payloadCoord", "value") << ";\n"
        << "   " << gl_controlBarrier(SCOPE_WORKGROUP, SCOPE_WORKGROUP, semAcqRel | semAv) << ";\n"
        << "   if (all(equal(gl_LocalInvocationID.xy, ivec2(0,0)))) {\n";

    if (m_data.syncType == ST_ATOMIC_ATOMIC || m_data.syncType == ST_ATOMIC_FENCE)
        css << "      " << gl_atomicStore("guard", "guardCoord", "1u", SCOPE_DEVICE, semRel) << ";\n";
    else
        css << "      " << gl_memoryBarrier(SCOPE_DEVICE, semRel | semGuard) << ";\n"
            << "      " << gl_atomicStore("guard", "guardCoord", "1u", SCOPE_DEVICE, 0) << ";\n";

    css << "   }\n";

    if (m_data.syncType == ST_ATOMIC_ATOMIC || m_data.syncType == ST_FENCE_ATOMIC)
        css << "   skip = " << gl_atomicLoad("guard", "guardGroupCoordPartner", SCOPE_DEVICE, semAcq) << " == 0;\n";
    else
        css << "   skip = " << gl_atomicLoad("guard", "guardGroupCoordPartner", SCOPE_DEVICE, 0) << " == 0;\n"
            << "   " << gl_memoryBarrier(SCOPE_DEVICE, semAcq | semGuard) << ";\n";

    css << "   r = " << gl_load("payload", "payloadCoordPartner") << ";\n"
        << "   if (!idle && !skip && r != expected) {\n"
        << "      " << gl_store("fail", "failCoord", "1") << ";\n"
        << "   }\n";

    return css.str();
}

class AvailabilityVisibilityChainsTestCase : public ChainsWithBarrierTestCase
{
public:
    using ChainsWithBarrierTestCase::ChainsWithBarrierTestCase;

private:
    std::string declareSharedVars() const override;
    std::string initProgram() const override;
};

std::string AvailabilityVisibilityChainsTestCase::declareSharedVars() const
{
    return std::string{"shared bool sharedSkip;\n"};
}

std::string AvailabilityVisibilityChainsTestCase::initProgram() const
{
    const DataType type       = getVar("payload").type;
    const char *typeStr       = gl_type(type);
    const uint32_t semAv      = m_data.coherent ? 0u : uint32_t(SEM_AVAILABLE);
    const uint32_t semVis     = m_data.coherent ? 0u : uint32_t(SEM_VISIBLE);
    const uint32_t semPayload = getVarSemantics("payload");
    const uint32_t semGuard   = getVarSemantics("guard");
    const uint32_t semRel     = SEM_RELEASE | semPayload | SEM_AVAILABLE;
    const uint32_t semAcq     = SEM_ACQUIRE | semPayload | SEM_VISIBLE;
    const uint32_t semAcqRel  = SEM_ACQUIRE_RELEASE | semPayload;

    std::ostringstream css;

    css << initPartnerCoordinates(SCOPE_DEVICE);
    css << "   ivec2 groupPosPartner = posPartner - posPartner % DIM;\n"
        << initVarCoordinates("groupPosPartner", "GroupCoordPartner");

    css << "   bool skip = false;\n"
        << "   uint value = invocationId;\n"
        << "   " << typeStr << " r = 0;\n"
        << "   " << typeStr << " expected = " << typeStr << "(invocationIdPartner);\n"
        << "   value += " << gl_toUint(type, gl_load("payload", "payloadCoordPartner")) << " >> 31;\n"
        << "   " << gl_store("payload", "payloadCoord", "value") << ";\n"
        << "   " << gl_controlBarrier(SCOPE_WORKGROUP, SCOPE_WORKGROUP, semAcqRel | semAv) << ";\n"
        << "   if (all(equal(gl_LocalInvocationID.xy, ivec2(0,0)))) {\n";

    if (m_data.syncType == ST_ATOMIC_ATOMIC || m_data.syncType == ST_ATOMIC_FENCE)
        css << "      " << gl_atomicStore("guard", "guardCoord", "1u", SCOPE_DEVICE, semRel) << ";\n";
    else
        css << "      " << gl_memoryBarrier(SCOPE_DEVICE, semRel | semGuard) << ";\n"
            << "      " << gl_atomicStore("guard", "guardCoord", "1u", SCOPE_DEVICE, 0) << ";\n";

    if (m_data.syncType == ST_ATOMIC_ATOMIC || m_data.syncType == ST_FENCE_ATOMIC)
        css << "      skip = " << gl_atomicLoad("guard", "guardGroupCoordPartner", SCOPE_DEVICE, semAcq) << " == 0;\n";
    else
        css << "      skip = " << gl_atomicLoad("guard", "guardGroupCoordPartner", SCOPE_DEVICE, 0) << " == 0;\n"
            << "      " << gl_memoryBarrier(SCOPE_DEVICE, semAcq | semGuard) << ";\n";

    // SEM_SHARED in gl_controlBarrier orders accesses to sharedSkip
    css << "      sharedSkip = skip;\n"
        << "   }\n"
        << "   " << gl_controlBarrier(SCOPE_WORKGROUP, SCOPE_WORKGROUP, SEM_SHARED | semAcqRel | semVis) << ";\n"
        << "   skip = sharedSkip;\n"
        << "   r = " << gl_load("payload", "payloadCoordPartner") << ";\n"
        << "   if (!idle && !skip && r != expected) {\n"
        << "      " << gl_store("fail", "failCoord", "1") << ";\n"
        << "   }\n";

    return css.str();
}

template <typename VariantTestCase>
void addChainsWithBarrierTests(tcu::TestCaseGroup &root, const char *variantName)
{
    static const NamedValue<StorageClass> scCases[] = {
        {SC_BUFFER, "buffer"},
        {SC_IMAGE, "image"},
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
    };

    static const NamedValue<bool> cohCases[] = {
        {true, "coherent"},
        {false, "noncoherent"},
    };

    const auto groupName = [](const char *prefix, const char *name) { return std::string(prefix) + "_" + name; };

    for (const auto &[local0, sc0, local1, sc1, sync, coh] :
         Combinations(localCases, scCases, localCases, scCases, syncCases, cohCases))
    {
        const CaseDef c = {
            local0.value, // bool payloadMemLocal;
            local1.value, // bool guardMemLocal;
            coh.value,    // bool coherent;
            sc0.value,    // StorageClass payloadSC;
            sc1.value,    // StorageClass guardSC;
            sync.value,   // SyncType syncType;
        };

        addTestCase(root,
                    {coh.name, sync.name, groupName("payload", local0.name), sc0.name, groupName("guard", local1.name),
                     sc1.name},
                    new VariantTestCase(root.getTestContext(), variantName, c));
    }
}

} // namespace

tcu::TestCaseGroup *createChainsWithBarrierTests(tcu::TestContext &testCtx)
{
    de::MovePtr<tcu::TestCaseGroup> group(new tcu::TestCaseGroup(testCtx, "transitive"));

    addChainsWithBarrierTests<AvailabilityChainsTestCase>(*group, "nontransvis");
    addChainsWithBarrierTests<AvailabilityVisibilityChainsTestCase>(*group, "transvis");

    return group.release();
}

} // namespace vkt::MemoryModel
