#ifndef _VKTMEMORYMODELTESTCASE_HPP
#define _VKTMEMORYMODELTESTCASE_HPP
/*-------------------------------------------------------------------------
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
 * \brief Memory Model Test Case
 *//*--------------------------------------------------------------------*/

#include "tcuDefs.hpp"
#include "tcuTestCase.hpp"
#include "vktTestCase.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace vkt
{
namespace MemoryModel
{

enum DataType
{
    DATA_TYPE_UINT = 0,
    DATA_TYPE_UINT64,
    DATA_TYPE_FLOAT32,
    DATA_TYPE_FLOAT64,
};

enum StorageClass
{
    SC_BUFFER = 0,
    SC_IMAGE,
    SC_WORKGROUP,
    SC_PHYSBUFFER,
};

// From narrowest to widest, used to compare scopes
enum Scope
{
    SCOPE_INVOCATION = 0,
    SCOPE_SUBGROUP,
    SCOPE_WORKGROUP,
    SCOPE_QUEUEFAMILY,
    SCOPE_DEVICE,
};

enum Semantics
{
    SEM_NONE            = 0x0,
    SEM_ACQUIRE         = 0x2,
    SEM_RELEASE         = 0x4,
    SEM_ACQUIRE_RELEASE = 0x8,
    SEM_BUFFER          = 0x40,
    SEM_SHARED          = 0x100,
    SEM_IMAGE           = 0x800,
    SEM_AVAILABLE       = 0x2000,
    SEM_VISIBLE         = 0x4000,
};

enum Stage
{
    STAGE_COMPUTE = 0,
    STAGE_VERTEX,
    STAGE_FRAGMENT,
};

enum MemQualifier
{
    MEM_NONE = 0,
    MEM_NONPRIVATE,
    MEM_SUBGROUP,
    MEM_WORKGROUP,
    MEM_QUEUEFAMILY,
    MEM_DEVICE,
    MEM_COHERENT,
};

struct VarDef
{
    const char *name;
    MemQualifier qual;
    StorageClass sc;
    DataType type;
    bool local;
};

struct ParamsDef
{
    Stage stage;
    bool core11;
};

constexpr bool isIntType(const DataType type)
{
    return type == DATA_TYPE_UINT || type == DATA_TYPE_UINT64;
}

constexpr bool is64BitType(const DataType type)
{
    return type == DATA_TYPE_UINT64 || type == DATA_TYPE_FLOAT64;
}

constexpr Semantics scToSem(const StorageClass sc)
{
    switch (sc)
    {
    case SC_PHYSBUFFER:
    case SC_BUFFER:
        return SEM_BUFFER;
    case SC_IMAGE:
        return SEM_IMAGE;
    case SC_WORKGROUP:
        return SEM_SHARED;
    default:
        DE_ASSERT(0);
        return SEM_NONE;
    }
}

constexpr MemQualifier scopeToQualifier(const Scope scope)
{
    switch (scope)
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

class MemoryModelTestCase : public TestCase
{
public:
    void checkSupport(Context &context) const override = 0;
    void initPrograms(vk::SourceCollections &programCollection) const final;
    TestInstance *createInstance(Context &context) const final;

protected:
    MemoryModelTestCase(tcu::TestContext &context, const char *name, const ParamsDef &params, std::vector<VarDef> vars)
        : TestCase(context, name)
        , m_params(params)
        , m_vars(std::move(vars))
    {
    }

    // Hooks for initPrograms, initProgram must be overridden, the declarations are empty by default
    virtual std::string initProgram() const = 0;
    virtual std::string declareSharedVars() const
    {
        return {};
    }
    virtual std::string declareSpvIntrinsics() const
    {
        return {};
    }

    // Helper functions to check support
    void checkSupportVars(const Context &context) const;
    void checkSupportVarAtomics(const Context &context, const char *name) const;
    void checkSupportStageAtomics(const Context &context) const;
    void checkSupportSubgroupFeatures(const Context &context, uint32_t minSize = 2) const;
    void checkSupportSplitBarriers(const Context &context) const;

    // Helper functions to initialize partner and variable coordinates
    std::string initPartnerCoordinates(Scope scope, const char *suffix = "") const;
    std::string initMultiPartnerCoordinates(Scope scope, uint32_t numPartners) const;
    std::string initVarCoordinates(const char *pos, const char *suffix) const;

    // Helper functions to construct glsl instructions with respect to variable type
    static const char *gl_barrier();
    static std::string gl_controlBarrier(Scope scope1, Scope scope2, uint32_t sem);
    static std::string gl_controlBarrierArrive(Scope scope1, Scope scope2, uint32_t sem);
    static std::string gl_controlBarrierWait(Scope scope1, Scope scope2, uint32_t sem);
    static std::string gl_memoryBarrier(Scope scope, uint32_t sem);
    std::string gl_atomicLoad(const char *name, const char *idx, Scope scope, uint32_t sem) const;
    std::string gl_atomicStore(const char *name, const char *idx, const char *value, Scope scope, uint32_t sem) const;
    std::string gl_atomicExchange(const char *name, const char *idx, const char *value, Scope scope,
                                  uint32_t sem) const;
    std::string gl_load(const char *name, const char *idx) const;
    std::string gl_store(const char *name, const char *idx, const char *value) const;
    static const char *gl_type(DataType type);
    static const char *gl_scope(Scope scope);
    static std::string gl_toUint(DataType type, const std::string &expr);

    // Helper functions for SPIR-V intrinsics
    std::string declareVarSpvIntrinsics(const char *name, bool hasDeviceScope) const;
    std::string spv_load(const char *name, const char *idx) const;
    std::string spv_load(const char *name, const char *idx, Scope scope) const;
    std::string spv_store(const char *name, const char *idx, const char *value) const;
    std::string spv_store(const char *name, const char *idx, const char *value, Scope scope) const;

    // Helper functions for variables
    const VarDef &getVar(const char *name) const;
    uint32_t getVarSemantics(const char *name) const;

private:
    std::string declareVars() const;
    std::string initPartners(Scope scope, uint32_t numPartners, bool isIndexRotation, const char *suffix) const;
    bool hasStorageClass(StorageClass sc) const;
    bool hasDataType(DataType type) const;

    const ParamsDef m_params;
    const std::vector<VarDef> m_vars;
};

} // namespace MemoryModel
} // namespace vkt

#endif // _VKTMEMORYMODELTESTCASE_HPP
