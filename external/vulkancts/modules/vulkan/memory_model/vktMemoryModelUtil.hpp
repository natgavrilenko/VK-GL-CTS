#ifndef _VKTMEMORYMODELUTIL_HPP
#define _VKTMEMORYMODELUTIL_HPP
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

#include "tcuTestCase.hpp"

#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkt
{
namespace MemoryModel
{

template <typename T>
struct NamedValue
{
    T value;
    const char *name;
};

template <typename... Tables>
class Combinations
{
    static_assert(((std::extent_v<Tables> > 0) && ...), "Tables must be non-empty arrays");

public:
    explicit Combinations(const Tables &...tables) : m_tables(tables...)
    {
    }

    class Iterator
    {
    public:
        explicit Iterator(const Combinations *combinations) : m_combinations(combinations)
        {
        }

        auto operator*() const
        {
            return get(std::index_sequence_for<Tables...>());
        }

        Iterator &operator++()
        {
            for (size_t i = sizeof...(Tables); i-- > 0;)
            {
                if (++m_indices[i] < kSizes[i])
                    return *this;
                m_indices[i] = 0;
            }
            m_combinations = nullptr;
            return *this;
        }

        bool operator!=(const Iterator &other) const
        {
            return m_combinations != other.m_combinations;
        }

    private:
        template <size_t... I>
        auto get(std::index_sequence<I...>) const
        {
            return std::tie(std::get<I>(m_combinations->m_tables)[m_indices[I]]...);
        }

        static constexpr size_t kSizes[] = {std::extent_v<Tables>...};
        const Combinations *m_combinations;
        size_t m_indices[sizeof...(Tables)] = {};
    };

    Iterator begin() const
    {
        return Iterator(this);
    }

    Iterator end() const
    {
        return Iterator(nullptr);
    }

private:
    const std::tuple<const Tables &...> m_tables;
};

void addTestCase(tcu::TestCaseGroup &root, const std::vector<std::string> &path, tcu::TestNode *testCase);

} // namespace MemoryModel
} // namespace vkt

#endif // _VKTMEMORYMODELUTIL_HPP
