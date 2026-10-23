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

#include "tcuDefs.hpp"
#include "tcuTestCase.hpp"
#include "vktTestCase.hpp"

#include <vector>

namespace vkt
{
namespace MemoryModel
{

class MemoryModelTestInstance : public vkt::MultiQueueRunnerTestInstance
{
public:
    MemoryModelTestInstance(Context &context, Stage stage, const std::vector<BufferDef> &buffers);
    ~MemoryModelTestInstance(void) = default;
    tcu::TestStatus queuePass(const vkt::QueueData &queueData) override;

private:
    const Stage m_stage;
    const std::vector<BufferDef> m_buffers;
};

} // namespace MemoryModel
} // namespace vkt

#endif // _VKTMEMORYMODELPATTERNTESTINSTANCE_HPP
