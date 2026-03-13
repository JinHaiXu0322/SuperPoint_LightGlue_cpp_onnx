cmake_minimum_required(VERSION 3.10)
project(sp_lg_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# -------- ONNX Runtime path --------
# 你可以在命令行传：-DORT_DIR=/path/to/onnxruntime
# 如果不传（例如 CLion），就使用默认路径
set(ORT_DIR "/opt/onnxruntime/onnxruntime-linux-x64-gpu-1.24.2" CACHE PATH "Path to ONNX Runtime root (contains include/ and lib/)")

# 安全检查：避免配置错路径还继续编译
if(NOT EXISTS "${ORT_DIR}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "onnxruntime_cxx_api.h not found. Please set ORT_DIR correctly. Current ORT_DIR=${ORT_DIR}")
endif()

# 有的发行包是 lib64，做个兼容
set(ORT_LIB_DIR "${ORT_DIR}/lib")
if(NOT EXISTS "${ORT_LIB_DIR}/libonnxruntime.so")
    set(ORT_LIB_DIR "${ORT_DIR}/lib64")
endif()
if(NOT EXISTS "${ORT_LIB_DIR}/libonnxruntime.so")
    message(FATAL_ERROR "libonnxruntime.so not found in ${ORT_DIR}/lib or ${ORT_DIR}/lib64. Please check ORT_DIR.")
endif()

# -------- targets --------

# 原始完整版 demo
add_executable(sp_lg_demo src/sp_lg_demo.cpp)

target_include_directories(sp_lg_demo PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${ORT_DIR}/include
)

target_link_libraries(sp_lg_demo PRIVATE
        ${OpenCV_LIBS}
        ${ORT_LIB_DIR}/libonnxruntime.so
)

set_target_properties(sp_lg_demo PROPERTIES
        BUILD_RPATH "${ORT_LIB_DIR}"
        INSTALL_RPATH "${ORT_LIB_DIR}"
)

# 精简版 demo（固定 I/O 名称和常见 shape，去掉大量兼容判断）
add_executable(sp_lg_demo_simple src/sp_lg_demo_simple.cpp)

target_include_directories(sp_lg_demo_simple PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${ORT_DIR}/include
)

target_link_libraries(sp_lg_demo_simple PRIVATE
        ${OpenCV_LIBS}
        ${ORT_LIB_DIR}/libonnxruntime.so
)

set_target_properties(sp_lg_demo_simple PROPERTIES
        BUILD_RPATH "${ORT_LIB_DIR}"
        INSTALL_RPATH "${ORT_LIB_DIR}"
)

# GPU 默认优先 demo（自动尝试 CUDA，不需要命令行指定）
add_executable(sp_lg_demo_gpu src/sp_lg_demo_gpu.cpp)

target_include_directories(sp_lg_demo_gpu PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${ORT_DIR}/include
)

target_link_libraries(sp_lg_demo_gpu PRIVATE
        ${OpenCV_LIBS}
        ${ORT_LIB_DIR}/libonnxruntime.so
)

set_target_properties(sp_lg_demo_gpu PROPERTIES
        BUILD_RPATH "${ORT_LIB_DIR}"
        INSTALL_RPATH "${ORT_LIB_DIR}"
)
