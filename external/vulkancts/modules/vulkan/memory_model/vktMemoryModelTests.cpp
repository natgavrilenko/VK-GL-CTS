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
 * \brief Memory Model Tests
 *//*--------------------------------------------------------------------*/

#include "vktMemoryModelTests.hpp"
#include "vktMemoryModelAmberTests.hpp"
#include "vktMemoryModelPaddingTests.hpp"
#include "vktMemoryModelPatternMessagePassingTests.hpp"
#include "vktMemoryModelPatternMixedAccessTests.hpp"
#include "vktMemoryModelPatternTransitiveTests.hpp"
#include "vktMemoryModelSharedLayoutTests.hpp"

#include "deUniquePtr.hpp"
#include "tcuTestCase.hpp"

namespace vkt
{
namespace MemoryModel
{

tcu::TestCaseGroup *createTests(tcu::TestContext &testCtx, const std::string &name)
{
    de::MovePtr<tcu::TestCaseGroup> group(new tcu::TestCaseGroup(testCtx, name.c_str()));

    group->addChild(createAmberTests(testCtx));
    group->addChild(createMessagePassingTests(testCtx));
    group->addChild(createLoadBufferTests(testCtx));
    group->addChild(createTransitiveTests(testCtx));
    group->addChild(createMixedAccessTests(testCtx));
    group->addChild(createPaddingTests(testCtx));
    group->addChild(createSharedMemoryLayoutTests(testCtx));

    return group.release();
}

} // namespace MemoryModel
} // namespace vkt
