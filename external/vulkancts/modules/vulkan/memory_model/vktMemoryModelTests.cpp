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
#include "vktMemoryModelAvVisChainsTests.hpp"
#include "vktMemoryModelMessagePassingTests.hpp"
#include "vktMemoryModelMixedAccessTests.hpp"
#include "vktMemoryModelPaddingTests.hpp"
#include "vktMemoryModelRepeatedAccessTests.hpp"
#include "vktMemoryModelSharedLayoutTests.hpp"
#include "vktMemoryModelTransitiveSemanticsTests.hpp"

#include "deUniquePtr.hpp"
#include "tcuTestCase.hpp"

namespace vkt
{
namespace MemoryModel
{

tcu::TestCaseGroup *createTests(tcu::TestContext &testCtx, const std::string &name)
{
    de::MovePtr<tcu::TestCaseGroup> group(new tcu::TestCaseGroup(testCtx, name.c_str()));

#ifndef CTS_USES_VULKANSC
    group->addChild(createAmberTests(testCtx));
#endif // CTS_USES_VULKANSC
    group->addChild(createAvVisChainsTests(testCtx));
    group->addChild(createMessagePassingTests(testCtx));
    group->addChild(createMixedAccessTests(testCtx));
    group->addChild(createPaddingTests(testCtx));
    group->addChild(createRepeatedAccessTests(testCtx));
    group->addChild(createSharedMemoryLayoutTests(testCtx));
    group->addChild(createTransitiveSemanticsTests(testCtx));
    group->addChild(createWriteAfterReadTests(testCtx));

    return group.release();
}

} // namespace MemoryModel
} // namespace vkt
