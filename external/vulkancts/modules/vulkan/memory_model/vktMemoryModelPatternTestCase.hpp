#ifndef _VKTMEMORYMODELPATTERNTESTCASE_HPP
#define _VKTMEMORYMODELPATTERNTESTCASE_HPP
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
 * \brief Memory Model Pattern Test Case
 *//*--------------------------------------------------------------------*/

#include "tcuDefs.hpp"
#include "vktTestCase.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>
#include <string_view>
#include <unordered_map>
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

enum Scope
{
    SCOPE_DEVICE = 0,
    SCOPE_QUEUEFAMILY,
    SCOPE_WORKGROUP,
    SCOPE_SUBGROUP,
    SCOPE_INVOCATION,
};

enum Semantics
{
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

enum GlMemQualifier
{
    MEM_NONE = 0,
    MEM_NONPRIVATE,
    MEM_SUBGROUP,
    MEM_WORKGROUP,
    MEM_QUEUEFAMILY,
    MEM_DEVICE,
    MEM_COHERENT,
};

struct BufferDef
{
    const char *name;
    StorageClass sc;
    DataType type;
    GlMemQualifier memqual;
    bool local;
};

class MemoryModelTestCase : public TestCase
{
public:
    MemoryModelTestCase(tcu::TestContext &context, const char *name, std::vector<BufferDef> buffers)
        : TestCase(context, name)
        , m_buffers(std::move(buffers))
    {
        for (std::size_t i = 0; i < m_buffers.size(); ++i)
            m_bufferIndices[m_buffers[i].name] = i;
    }

protected:
    virtual void checkSupportBuffers(const Context &context) const;
    virtual void checkSupportBufferAtomics(const Context &context, const char *name) const;
    virtual void checkSupportStageAtomics(const Context &context, Stage stage) const;
    virtual void checkSupportStageSubgroupFeatures(const Context &context, Stage stage) const;
    virtual void checkSupportStageSplitBarriers(const Context &context, Stage stage) const;
    virtual const BufferDef &getBuffer(const char *name) const;
    virtual uint32_t getBufferSemantics(const char *name) const;
    std::vector<BufferDef> m_buffers;
    std::unordered_map<std::string_view, std::size_t> m_bufferIndices;
};

struct StreamOp
{
    std::function<void(std::ostream &)> fn;

    template <typename Callable>
    StreamOp(Callable &&c) : fn(std::forward<Callable>(c)) {}

    friend std::ostream &operator<<(std::ostream &os, const StreamOp &op)
    {
        op.fn(os);
        return os;
    }
};

class MemoryModelGlTestCase : public MemoryModelTestCase
{
public:
    using MemoryModelTestCase::MemoryModelTestCase;

protected:
    static StreamOp gl_layoutCoordinates(const Stage stage, Scope mappingScope);
    static StreamOp gl_layoutCoordinates3(const Stage stage, Scope mappingScope);
    static const char *gl_barrier();
    static StreamOp gl_controlBarrier(Scope scope1, Scope scope2, uint32_t sem);
    static StreamOp gl_controlBarrierArrive(Scope scope1, Scope scope2, uint32_t sem);
    static StreamOp gl_controlBarrierWait(Scope scope1, Scope scope2, uint32_t sem);
    static StreamOp gl_memoryBarrier(Scope scope, uint32_t sem);
    StreamOp gl_atomicLoad(const char *name, const char* idx, Scope scope, uint32_t sem) const;
    StreamOp gl_atomicStore(const char *name, const char* idx, const char* value, Scope scope, uint32_t sem) const;
    StreamOp gl_atomicExchange(const char *name, const char* idx, const char* value, Scope scope, uint32_t sem) const;
    StreamOp gl_load(const char *name, const char* idx) const;
    StreamOp gl_store(const char *name, const char* idx, const char* value) const;
    StreamOp gl_layout() const;
    StreamOp gl_layoutStructs() const;
    StreamOp gl_layoutReferences() const;
    StreamOp gl_layoutBufferCoordinates() const;
    StreamOp gl_layoutBufferCoordinates00() const;
    StreamOp gl_layoutInitSharedMemory(Scope scope) const;
    const char *gl_type(const char *name) const;
    static const char *gl_type(DataType type);
    static const char *gl_scope(Scope scope);

private:
    static constexpr uint32_t GL_SEM1 = SEM_BUFFER | SEM_IMAGE | SEM_SHARED;
    static constexpr uint32_t GL_SEM2 = SEM_ACQUIRE | SEM_RELEASE | SEM_ACQUIRE_RELEASE | SEM_AVAILABLE | SEM_VISIBLE;
    static const char *gl_memqual(GlMemQualifier qualifier);
    template <std::size_t N>
    static StreamOp formatSemantics(uint32_t value, const std::pair<uint32_t, const char *> (&mappings)[N]);
    static StreamOp gl_semantics1(uint32_t value);
    static StreamOp gl_semantics2(uint32_t value);
};

} // namespace MemoryModel
} // namespace vkt

#endif // _VKTMEMORYMODELPATTERNTESTCASE_HPP
