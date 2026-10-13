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
 * \brief Vulkan Memory Model Amber Tests
 *//*--------------------------------------------------------------------*/

#include "vktMemoryModelAmberTests.hpp"

#include "vktAmberTestCase.hpp"
#include "vktTestCase.hpp"

namespace vkt
{
namespace MemoryModel
{

namespace
{
#ifndef CTS_USES_VULKANSC
void checkPermutedIndexSupport(Context &context, std::string testName)
{
    DE_UNREF(testName);

    const vk::VkPhysicalDeviceProperties &properties = context.getDeviceProperties();

    const auto maxComputeWorkGroupCount = properties.limits.maxComputeWorkGroupCount;
    if (maxComputeWorkGroupCount[0] < 256u)
        TCU_THROW(NotSupportedError, "Minimum of 256 required for maxComputeWorkGroupCount.x");

    const auto maxComputeWorkGroupSize = properties.limits.maxComputeWorkGroupSize;
    if (maxComputeWorkGroupSize[0] < 256u)
        TCU_THROW(NotSupportedError, "Minimum of 256 required for maxComputeWorkGroupSize.x");

    const auto maxComputeWorkGroupInvocations = properties.limits.maxComputeWorkGroupInvocations;
    if (maxComputeWorkGroupInvocations < 256u)
        TCU_THROW(NotSupportedError, "Minimum of 256 required for maxComputeWorkGroupInvocations");
}
#endif // CTS_USES_VULKANSC

} // namespace

tcu::TestCaseGroup *createPermutedIndexTests(tcu::TestContext &testCtx)
{
    de::MovePtr<tcu::TestCaseGroup> group(new tcu::TestCaseGroup(testCtx, "permuted_index"));

#ifndef CTS_USES_VULKANSC
    const char *dataDir = "memory_model/permuted_index";

    const char *tests[] = {
        "barrier",
        "release_acquire",
        "release_acquire_atomic_payload"
    };

    for (const auto test : tests)
    {
        const std::string fileName = std::string(test) + ".amber";
        cts_amber::AmberTestCase *testCase = cts_amber::createAmberTestCase(testCtx, test, dataDir, fileName);
        testCase->setCheckSupportCallback(checkPermutedIndexSupport);
        group->addChild(testCase);
    }
#endif

    return group.release();
}

tcu::TestCaseGroup *createAmberTests(tcu::TestContext &testCtx)
{
    de::MovePtr<tcu::TestCaseGroup> group(new tcu::TestCaseGroup(testCtx, "amber"));
    group->addChild(createPermutedIndexTests(testCtx));
    return group.release();
}

} // namespace MemoryModel
} // namespace vkt
