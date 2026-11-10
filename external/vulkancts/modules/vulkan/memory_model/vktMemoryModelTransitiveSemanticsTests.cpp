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

std::vector<BufferDef> createBuffers(const CaseDef &data)
{
    const MemQualifier qualifier = scopeToQualifier(data.scope);
    return std::vector<BufferDef>{{"payload", qualifier, data.payloadSC, data.payloadType, data.payloadMemLocal},
                                  {"guard1", qualifier, data.guard1SC, data.guard1Type, data.guard1MemLocal},
                                  {"guard2", qualifier, data.guard2SC, data.guard2Type, data.guard2MemLocal},
                                  {"fail", MEM_NONE, SC_BUFFER, DATA_TYPE_UINT, true}};
}

class TransitiveSemanticsTestCase : public MemoryModelTestCase
{
public:
    TransitiveSemanticsTestCase(tcu::TestContext &context, const char *name, const CaseDef &data);
    void checkSupport(Context &context) const override;

private:
    std::string initProgram() const override;
    const CaseDef m_data;
};

TransitiveSemanticsTestCase::TransitiveSemanticsTestCase(tcu::TestContext &context, const char *name,
                                                         const CaseDef &data)
    : MemoryModelTestCase(context, name, {STAGE_COMPUTE, false}, createBuffers(data))
    , m_data(data)
{
}

void TransitiveSemanticsTestCase::checkSupport(Context &context) const
{
    if (!context.contextSupports(vk::ApiVersion(0, 1, 1, 0)))
        TCU_THROW(NotSupportedError, "Vulkan 1.1 not supported");

    if (!context.getVulkanMemoryModelFeatures().vulkanMemoryModel)
        TCU_THROW(NotSupportedError, "vulkanMemoryModel not supported");

    if (m_data.scope == SCOPE_SUBGROUP)
        checkSupportSubgroupFeatures(context);

    checkSupportBuffers(context);
    checkSupportBufferAtomics(context, "guard1");
    checkSupportBufferAtomics(context, "guard2");
}

std::string TransitiveSemanticsTestCase::initProgram() const
{
    const Scope scope   = m_data.scope;
    const DataType type = getBuffer("payload").type;
    const char *typeStr = gl_type(type);

    const uint32_t semPayload = getBufferSemantics("payload");
    const uint32_t semGuard1  = getBufferSemantics("guard1");
    const uint32_t semGuard2  = getBufferSemantics("guard2");
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
        << "   " << typeStr << " expected = " << typeStr << "(invocationIdPartner1);\n";

    // Prefetch the partner's payload; this adds 0 because every stored value is below 2^31
    css << "   value += " << gl_toUint(type, gl_load("payload", "payloadCoordPartner1")) << " >> 31;\n"
        << "   " << gl_store("payload", "payloadCoord", "value") << ";\n"
        << "   " << gl_atomicStore("guard1", "guard1CoordPartner1", "1u", scope, semRel1) << ";\n"
        << "   skip1 = " << gl_atomicLoad("guard1", "guard1Coord", scope, semAcq1) << " == 0;\n"
        << "   if (!skip1) { " << gl_atomicStore("guard2", "guard2CoordPartner1", "1u", scope, semRel2) << "; }\n"
        << "   skip2 = " << gl_atomicLoad("guard2", "guard2Coord", scope, semAcq2) << " == 0;\n"
        << "   if (!skip2) {\n"
        << "      r = " << gl_load("payload", "payloadCoordPartner1") << ";\n"
        << "      if (!idle && r != expected) {\n"
        << "         " << gl_store("fail", "failCoord", "1") << ";\n"
        << "      }\n"
           "   }\n";

    return css.str();
}

} // namespace

tcu::TestCaseGroup *createTransitiveSemanticsTests(tcu::TestContext &testCtx)
{
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

    de::MovePtr<tcu::TestCaseGroup> ttGroup(new tcu::TestCaseGroup(testCtx, "transitive_semantics"));
    for (const auto &[sc0, sc1, sc2, local0, local1, local2, type0, type1, type2, scope] :
         Combinations(scCases, scCases, scCases, localCases, localCases, localCases, typeCases, typeCases, typeCases,
                      scopeCases))
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
            scope.value,  // Scope scope;
        };

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

        addTestCase(*ttGroup,
                    {groupName("payload", sc0.name), groupName("guard1", sc1.name), groupName("guard2", sc2.name),
                     groupName("payload", local0.name), groupName("guard1", local1.name),
                     groupName("guard2", local2.name), groupName("payload", type0.name),
                     groupName("guard1", type1.name), groupName("guard2", type2.name)},
                    new TransitiveSemanticsTestCase(testCtx, scope.name, c));
    }

    return ttGroup.release();
}

} // namespace vkt::MemoryModel
