#ifndef _VKTMEMORYMODELPATTERNTESTINSTANCE_HPP
#define _VKTMEMORYMODELPATTERNTESTINSTANCE_HPP
/*-------------------------------------------------------------------------
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
 * \brief Memory Model Pattern Test Instance
 *//*--------------------------------------------------------------------*/

#include "vktMemoryModelPatternTestCase.hpp"

#include "vkBufferWithMemory.hpp"
#include "vkImageWithMemory.hpp"
#include "vkQueryUtil.hpp"
#include "vkBuilderUtil.hpp"
#include "vkCmdUtil.hpp"
#include "vkTypeUtil.hpp"
#include "vkObjUtil.hpp"

#include "tcuDefs.hpp"
#include "vktTestCase.hpp"

#include "deDefs.h"
#include "deMath.h"
#include "deSharedPtr.hpp"
#include "deString.h"

#include "tcuTestCase.hpp"
#include "tcuTestLog.hpp"

namespace vkt
{
namespace MemoryModel
{
using namespace vk;

namespace {
VkFlags getAllShaderStages(tcu::TestContext &testCtx)
{
    DE_UNREF(testCtx);
    return VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
}

VkFlags getAllPipelineStages(tcu::TestContext &testCtx)
{
    DE_UNREF(testCtx);
    return VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
}
}

struct InstanceCaseDef
{
    bool payloadMemLocal;
    bool guardMemLocal;
    StorageClass payloadSC;
    StorageClass guardSC;
    bool skipGuard;
    Stage stage;
    DataType dataType;
};

struct InstanceBufferDef
{
    StorageClass sc;
    DataType type;
    bool local;
    bool skip;
};

class MemoryModelTestInstance : public vkt::MultiQueueRunnerTestInstance
{
public:
    MemoryModelTestInstance(Context &context, const InstanceCaseDef &data);
    ~MemoryModelTestInstance(void) = default;
    tcu::TestStatus queuePass(const vkt::QueueData &queueData) override;

private:
    InstanceCaseDef m_data;
    std::array<InstanceBufferDef, 3> m_buffers;

    enum
    {
        WIDTH  = 256,
        HEIGHT = 256
    };
};

} // namespace MemoryModel
} // namespace vkt

#endif // _VKTMEMORYMODELPATTERNTESTINSTANCE_HPP
