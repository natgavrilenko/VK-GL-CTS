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
 * \brief Memory Model Utilities
 *//*--------------------------------------------------------------------*/

#include "vktMemoryModelUtil.hpp"

#include "deUniquePtr.hpp"
#include "tcuTestCase.hpp"

#include <algorithm>

namespace vkt
{
namespace MemoryModel
{

void addTestCase(tcu::TestCaseGroup &root, const std::vector<std::string> &path, tcu::TestNode *testCase)
{
    de::MovePtr<tcu::TestNode> child(testCase);
    tcu::TestNode *group = &root;
    std::vector<tcu::TestNode *> children;
    for (const auto &name : path)
    {
        group->getChildren(children);
        const auto it = std::find_if(children.begin(), children.end(),
                                     [&](const tcu::TestNode *c) { return name == c->getName(); });
        if (it != children.end())
        {
            group = *it;
            continue;
        }
        de::MovePtr<tcu::TestCaseGroup> newGroup(new tcu::TestCaseGroup(root.getTestContext(), name.c_str()));
        group->addChild(newGroup.get());
        group = newGroup.release();
    }
    group->addChild(child.get());
    child.release();
}

} // namespace MemoryModel
} // namespace vkt
