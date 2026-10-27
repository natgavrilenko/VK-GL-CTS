include_directories(
	..
    ../amber
	${DEQP_INL_DIR}
)

set(DEQP_VK_MEMORY_MODEL_SRCS
	vktMemoryModelAmberTests.cpp
	vktMemoryModelAmberTests.hpp
	vktMemoryModelPaddingTests.hpp
	vktMemoryModelPaddingTests.cpp
	vktMemoryModelPatternMessagePassingTests.cpp
	vktMemoryModelPatternMessagePassingTests.hpp
	vktMemoryModelPatternMixedAccessTests.cpp
	vktMemoryModelPatternMixedAccessTests.hpp
	vktMemoryModelPatternTestCase.cpp
	vktMemoryModelPatternTestCase.hpp
	vktMemoryModelPatternTestInstance.cpp
	vktMemoryModelPatternTestInstance.hpp
	vktMemoryModelPatternTransitiveTests.cpp
	vktMemoryModelPatternTransitiveTests.hpp
	vktMemoryModelSharedLayoutCase.cpp
	vktMemoryModelSharedLayoutCase.hpp
	vktMemoryModelSharedLayoutTests.cpp
	vktMemoryModelSharedLayoutTests.hpp
	vktMemoryModelTests.cpp
	vktMemoryModelTests.hpp
	../util/vktTypeComparisonUtil.hpp
	../util/vktTypeComparisonUtil.cpp
)

add_library(deqp-vk-memory-model STATIC ${DEQP_VK_MEMORY_MODEL_SRCS})
target_link_libraries(deqp-vk-memory-model tcutil vkutil)

add_library(deqp-vksc-memory-model STATIC ${DEQP_VK_MEMORY_MODEL_SRCS})
target_link_libraries(deqp-vksc-memory-model PUBLIC deqp-vksc-util tcutil vkscutil)
