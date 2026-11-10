include_directories(
	..
    ../amber
)

set(DEQP_VK_MEMORY_MODEL_SRCS
	vktMemoryModelAmberTests.cpp
	vktMemoryModelAmberTests.hpp
	vktMemoryModelAvVisChainsTests.cpp
	vktMemoryModelAvVisChainsTests.hpp
	vktMemoryModelMessagePassingTests.cpp
	vktMemoryModelMessagePassingTests.hpp
	vktMemoryModelMixedAccessTests.cpp
	vktMemoryModelMixedAccessTests.hpp
	vktMemoryModelPaddingTests.cpp
	vktMemoryModelPaddingTests.hpp
	vktMemoryModelRepeatedAccessTests.cpp
	vktMemoryModelRepeatedAccessTests.hpp
	vktMemoryModelSharedLayoutCase.cpp
	vktMemoryModelSharedLayoutCase.hpp
	vktMemoryModelSharedLayoutTests.cpp
	vktMemoryModelSharedLayoutTests.hpp
	vktMemoryModelTestCase.cpp
	vktMemoryModelTestCase.hpp
	vktMemoryModelTransitiveSemanticsTests.cpp
	vktMemoryModelTransitiveSemanticsTests.hpp
	vktMemoryModelTests.cpp
	vktMemoryModelTests.hpp
	vktMemoryModelUtil.cpp
	vktMemoryModelUtil.hpp
	../util/vktTypeComparisonUtil.hpp
	../util/vktTypeComparisonUtil.cpp
)

add_library(deqp-vk-memory-model STATIC ${DEQP_VK_MEMORY_MODEL_SRCS})
target_link_libraries(deqp-vk-memory-model tcutil vkutil)

add_library(deqp-vksc-memory-model STATIC ${DEQP_VK_MEMORY_MODEL_SRCS})
target_link_libraries(deqp-vksc-memory-model PUBLIC deqp-vksc-util tcutil vkscutil)
