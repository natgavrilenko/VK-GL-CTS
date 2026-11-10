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
 * \brief Memory Model Repeated Access Tests
 *//*--------------------------------------------------------------------*/

#include "vktMemoryModelShadowedStoreIncrementTests.hpp"
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
    DataType payloadType;
    DataType guard1Type;
    DataType guard2Type;
    Scope scopeNear;
    Scope scopeFar;
    Stage stage;
};

// A helper function to compare scopes by the number of affected threads.
// The test instance doesn't map invocations across queue families,
// so queue family and device scopes are treated as equal.
int scopeWidth(Scope scope)
{
    switch (scope)
    {
    case SCOPE_SUBGROUP:
        return 0;
    case SCOPE_WORKGROUP:
        return 1;
    case SCOPE_QUEUEFAMILY:
    case SCOPE_DEVICE:
        return 2;
    default:
        DE_ASSERT(0);
        return -1;
    }
}

std::vector<BufferDef> createBuffers(const CaseDef &data)
{
    return std::vector<BufferDef>{{"payload", MEM_NONE, data.payloadSC, data.payloadType, data.payloadMemLocal},
                                  {"guard1", MEM_NONE, data.guard1SC, data.guard1Type, data.guard1MemLocal},
                                  {"guard2", MEM_NONE, data.guard2SC, data.guard2Type, data.guard2MemLocal},
                                  {"fail", MEM_NONE, SC_BUFFER, DATA_TYPE_UINT, true}};
}

class RepeatedAccessTestCase : public MemoryModelTestCase
{
public:
    RepeatedAccessTestCase(tcu::TestContext &context, const char *name, const CaseDef &data);
    void checkSupport(Context &context) const override;

protected:
    std::string declareSpvIntrinsics() const override;
    const CaseDef m_data;
};

RepeatedAccessTestCase::RepeatedAccessTestCase(tcu::TestContext &context, const char *name, const CaseDef &data)
    : MemoryModelTestCase(context, name, {data.stage, false}, createBuffers(data))
    , m_data(data)
{
}

void RepeatedAccessTestCase::checkSupport(Context &context) const
{
    if (!context.contextSupports(vk::ApiVersion(0, 1, 1, 0)))
        TCU_THROW(NotSupportedError, "Vulkan 1.1 not supported");

    if (!context.getVulkanMemoryModelFeatures().vulkanMemoryModel)
        TCU_THROW(NotSupportedError, "vulkanMemoryModel not supported");

    if (m_data.scopeFar == SCOPE_DEVICE && !context.getVulkanMemoryModelFeatures().vulkanMemoryModelDeviceScope)
    {
        TCU_THROW(NotSupportedError, "vulkanMemoryModelDeviceScope not supported");
    }

    if (m_data.scopeNear == SCOPE_SUBGROUP)
        checkSupportSubgroupFeatures(context);

    checkSupportBuffers(context);
    checkSupportBufferAtomics(context, "guard1");
    checkSupportBufferAtomics(context, "guard2");
    checkSupportStageAtomics(context);
}

std::string RepeatedAccessTestCase::declareSpvIntrinsics() const
{
    std::ostringstream css;
    css << "#extension GL_EXT_spirv_intrinsics : enable\n"
        << declareBufferSpvIntrinsics("payload", m_data.scopeFar == SCOPE_DEVICE);
    return css.str();
}

class ShadowedStoreIncTestCase : public RepeatedAccessTestCase
{
public:
    static constexpr bool nearFirst = true;
    using RepeatedAccessTestCase::RepeatedAccessTestCase;

private:
    std::string initProgram() const override;
};

std::string ShadowedStoreIncTestCase::initProgram() const
{
    const char *typeStr       = gl_type(getBuffer("payload").type);
    const Scope scopeNear     = m_data.scopeNear;
    const Scope scopeFar      = m_data.scopeFar;
    const uint32_t semPayload = getBufferSemantics("payload");
    const uint32_t semRel     = SEM_RELEASE | semPayload;
    const uint32_t semAcq     = SEM_ACQUIRE | semPayload;

    std::ostringstream css;

    css << initPartnerCoordinates(m_data.scopeNear, "Near")
        << initPartnerCoordinates(m_data.scopeFar, "Far");

    css << "   bool skip1;\n"
        << "   bool skip2;\n"
        << "   uint value = invocationId;\n"
        << "   " << typeStr << " r = 0;\n"
        << "   " << typeStr << " expected = " << typeStr << "(invocationIdPartnerFar);\n"
        << "   " << spv_store("payload", "payloadCoordPartnerNear", "value", scopeNear) << ";\n"
        << "   " << gl_atomicStore("guard1", "guard1CoordPartnerNear", "1u", scopeNear, semRel) << ";\n"
        << "   skip1 = " << gl_atomicLoad("guard1", "guard1Coord", scopeNear, semAcq) << " == 0;\n"
        << "   if (!skip1) {\n"
        << "      " << spv_store("payload", "payloadCoord", "value", scopeFar) << ";\n"
        << "      " << gl_atomicStore("guard2", "guard2Coord", "1u", scopeFar, semRel) << ";\n"
        << "   }\n"
        << "   skip2 = " << gl_atomicLoad("guard2", "guard2CoordPartnerFar", scopeFar, semAcq) << " == 0;\n"
        << "   if (!skip2) {\n"
        << "      r = " << spv_load("payload", "payloadCoordPartnerFar", scopeFar) << ";\n"
        << "      if (!idleNear && !idleFar && r != expected) {\n"
        << "         " << gl_store("fail", "failCoord", "1") << ";\n"
        << "      }\n"
        << "   }\n";

    return css.str();
}

class ShadowedStoreDecTestCase : public RepeatedAccessTestCase
{
public:
    static constexpr bool nearFirst = false;
    using RepeatedAccessTestCase::RepeatedAccessTestCase;

private:
    std::string initProgram() const override;
};

std::string ShadowedStoreDecTestCase::initProgram() const
{
    const char *typeStr       = gl_type(getBuffer("payload").type);
    const Scope scopeNear     = m_data.scopeNear;
    const Scope scopeFar      = m_data.scopeFar;
    const uint32_t semPayload = getBufferSemantics("payload");
    const uint32_t semRel     = SEM_RELEASE | semPayload;
    const uint32_t semAcq     = SEM_ACQUIRE | semPayload;

    std::ostringstream css;

    css << initPartnerCoordinates(m_data.scopeNear, "Near")
        << initPartnerCoordinates(m_data.scopeFar, "Far");

    css << "   bool skip1;\n"
        << "   bool skip2;\n"
        << "   uint value = invocationId;\n"
        << "   " << typeStr << " r = 0;\n"
        << "   " << typeStr << " expected = " << typeStr << "(invocationIdPartnerNear);\n"
        << "   " << spv_store("payload", "payloadCoordPartnerFar", "value", scopeFar) << ";\n"
        << "   " << gl_atomicStore("guard1", "guard1CoordPartnerFar", "1u", scopeFar, semRel) << ";\n"
        << "   skip1 = " << gl_atomicLoad("guard1", "guard1Coord", scopeFar, semAcq) << " == 0;\n"
        << "   if (!skip1) {\n"
        << "      " << spv_store("payload", "payloadCoord", "value", scopeNear) << ";\n"
        << "      " << gl_atomicStore("guard2", "guard2Coord", "1u", scopeNear, semRel) << ";\n"
        << "   }\n"
        << "   skip2 = " << gl_atomicLoad("guard2", "guard2CoordPartnerNear", scopeNear, semAcq) << " == 0;\n"
        << "   if (!skip2) {\n"
        << "      r = " << spv_load("payload", "payloadCoordPartnerNear", scopeNear) << ";\n"
        << "      if (!idleNear && !idleFar && r != expected) {\n"
        << "         " << gl_store("fail", "failCoord", "1") << ";\n"
        << "      }\n"
        << "   }\n";

    return css.str();
}

class ObservedLoadTestCase : public RepeatedAccessTestCase
{
public:
    static constexpr bool nearFirst = false;
    using RepeatedAccessTestCase::RepeatedAccessTestCase;

private:
    std::string initProgram() const override;
};

std::string ObservedLoadTestCase::initProgram() const
{
    const char *typeStr       = gl_type(getBuffer("payload").type);
    const Scope scopeNear     = m_data.scopeNear;
    const Scope scopeFar      = m_data.scopeFar;
    const uint32_t semPayload = getBufferSemantics("payload");
    const uint32_t semRel     = SEM_RELEASE | semPayload;
    const uint32_t semAcq     = SEM_ACQUIRE | semPayload;

    std::ostringstream css;

    css << initPartnerCoordinates(m_data.scopeNear, "Near")
        << initPartnerCoordinates(m_data.scopeFar, "Far");

    css << "   bool skip1;\n"
        << "   bool skip2;\n"
        << "   uint value = invocationIdPartnerFar;\n"
        << "   " << typeStr << " r1 = 0;\n"
        << "   " << typeStr << " r2 = 0;\n"
        << "   " << typeStr << " expected1 = " << typeStr << "(invocationId);\n"
        << "   " << typeStr << " expected2 = " << typeStr << "(invocationIdPartnerNear);\n"
        << "   " << spv_store("payload", "payloadCoordPartnerFar", "value", scopeFar) << ";\n"
        << "   " << gl_atomicStore("guard1", "guard1CoordPartnerFar", "1u", scopeFar, semRel) << ";\n"
        << "   skip1 = " << gl_atomicLoad("guard1", "guard1Coord", scopeFar, semAcq) << " == 0;\n"
        << "   if (!skip1) {\n"
        << "      r1 = " << spv_load("payload", "payloadCoord", scopeFar) << ";\n"
        << "      " << gl_atomicStore("guard2", "guard2Coord", "1u", scopeNear, semRel) << ";\n"
        << "   }\n"
        << "   skip2 = " << gl_atomicLoad("guard2", "guard2CoordPartnerNear", scopeNear, semAcq) << " == 0;\n"
        << "   if (!skip2) {\n"
        << "      r2 = " << spv_load("payload", "payloadCoordPartnerNear", scopeNear) << ";\n"
        << "   }\n"
        << "   if (!idleNear && !idleFar && ((!skip1 && r1 != expected1) || (!skip2 && r2 != expected2))) {\n"
        << "      " << gl_store("fail", "failCoord", "1") << ";\n"
        << "   }\n";

    return css.str();
}

template <typename PatternTestCase>
void addPatternTests(tcu::TestCaseGroup &root, const char *patternName)
{
    static const NamedValue<StorageClass> scCases[] = {
        {SC_BUFFER, "buffer"}, {SC_WORKGROUP, "workgroup"},
        {SC_PHYSBUFFER, "physbuffer"},
    };

    static const NamedValue<bool> localCases[] = {
        {false, "nonlocal"},
        {true, "local"},
    };

    static const NamedValue<DataType> typeCases[] = {
        {DATA_TYPE_UINT, "u32"},
        {DATA_TYPE_UINT64, "u64"},
        {DATA_TYPE_FLOAT32, "f32"},
        {DATA_TYPE_FLOAT64, "f64"},
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

    for (const auto &[sc0, sc1, sc2, local0, local1, local2, type0, type1, type2, scope0, scope1, stage] :
         Combinations(scCases, scCases, scCases, localCases, localCases, localCases, typeCases, typeCases, typeCases,
                      scopeCases, scopeCases, stageCases))
    {
        const CaseDef c = {
            local0.value, // bool payloadMemLocal;
            local1.value, // bool guard1MemLocal;
            local2.value, // bool guard2MemLocal;
            sc0.value,    // StorageClass payloadSC;
            sc1.value,    // StorageClass guard1SC;
            sc2.value,    // StorageClass guard2SC;
            type0.value,  // DataType payloadType;
            type1.value,  // DataType guard1Type;
            type2.value,  // DataType guard2Type;
            scope0.value, // Scope scopeNear;
            scope1.value, // Scope scopeFar;
            stage.value,  // Stage stage;
        };

        // Near scope must be strictly narrower than far scope
        if (scopeWidth(c.scopeNear) >= scopeWidth(c.scopeFar))
        {
            continue;
        }

        // Skip tests where a variable is in workgroup, but its accesses have a larger scopes.
        // The payload is shared at both scopes, each guard at the scope of its step.
        const Scope guard1Scope = PatternTestCase::nearFirst ? c.scopeNear : c.scopeFar;
        const Scope guard2Scope = PatternTestCase::nearFirst ? c.scopeFar : c.scopeNear;
        const int wgScopeWidth = scopeWidth(SCOPE_WORKGROUP);
        if ((c.payloadSC == SC_WORKGROUP && scopeWidth(c.scopeFar) > wgScopeWidth) ||
            (c.guard1SC == SC_WORKGROUP && scopeWidth(guard1Scope) > wgScopeWidth) ||
            (c.guard2SC == SC_WORKGROUP && scopeWidth(guard2Scope) > wgScopeWidth))
        {
            continue;
        }

        // Workgroup memory is only tested as non-local
        if ((c.payloadMemLocal && c.payloadSC == SC_WORKGROUP) || (c.guard1MemLocal && c.guard1SC == SC_WORKGROUP) ||
            (c.guard2MemLocal && c.guard2SC == SC_WORKGROUP))
        {
            continue;
        }

        // Workgroup scope requires compute stage
        if (c.stage != STAGE_COMPUTE && (c.scopeNear == SCOPE_WORKGROUP || c.scopeFar == SCOPE_WORKGROUP))
        {
            continue;
        }

        // Workgroup storage class requires compute stage
        if (c.stage != STAGE_COMPUTE &&
            (c.payloadSC == SC_WORKGROUP || c.guard1SC == SC_WORKGROUP || c.guard2SC == SC_WORKGROUP))
        {
            continue;
        }

        addTestCase(root,
                    {patternName, groupName("payload", sc0.name), groupName("guard1", sc1.name),
                     groupName("guard2", sc2.name), groupName("payload", local0.name), groupName("guard1", local1.name),
                     groupName("guard2", local2.name), groupName("payload", type0.name),
                     groupName("guard1", type1.name), groupName("guard2", type2.name), groupName("near", scope0.name),
                     groupName("far", scope1.name)},
                    new PatternTestCase(root.getTestContext(), stage.name, c));
    }
}

} // namespace

tcu::TestCaseGroup *createRepeatedAccessTests(tcu::TestContext &testCtx)
{
    de::MovePtr<tcu::TestCaseGroup> raGroup(new tcu::TestCaseGroup(testCtx, "repeated_access"));

    addPatternTests<ShadowedStoreIncTestCase>(*raGroup, "shadowed_store_inc");
    addPatternTests<ShadowedStoreDecTestCase>(*raGroup, "shadowed_store_dec");
    addPatternTests<ObservedLoadTestCase>(*raGroup, "observed_load");

    return raGroup.release();
}

} // namespace vkt::MemoryModel
