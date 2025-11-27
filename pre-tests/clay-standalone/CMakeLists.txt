cmake_minimum_required(VERSION 3.12)

project(clay_standalone_test)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/clay
    ${CMAKE_CURRENT_SOURCE_DIR}/jerasure
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

# 查找依赖
find_package(Boost REQUIRED COMPONENTS program_options)

# 查找 Ceph RADOS
find_library(RADOS_LIB rados PATHS /usr/lib /usr/local/lib)
find_path(RADOS_INCLUDE_DIR rados/buffer.h PATHS /usr/include /usr/local/include)

if(NOT RADOS_LIB OR NOT RADOS_INCLUDE_DIR)
    message(FATAL_ERROR "找不到 Ceph RADOS 库。请安装 librados-dev")
endif()

include_directories(${RADOS_INCLUDE_DIR})

# 查找并配置 Jerasure 和 gf-complete（使用项目中的源码）
set(LIB_EC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../lib/ec)
set(JERASURE_DIR ${LIB_EC_DIR}/jerasure)
set(GF_COMPLETE_DIR ${LIB_EC_DIR}/gf-complete)

if(EXISTS ${JERASURE_DIR} AND EXISTS ${GF_COMPLETE_DIR})
    message(STATUS "使用项目中的 Jerasure: ${JERASURE_DIR}")
    message(STATUS "使用项目中的 gf-complete: ${GF_COMPLETE_DIR}")
    
    # 添加包含路径
    include_directories(
        ${JERASURE_DIR}/include
        ${GF_COMPLETE_DIR}/include
    )
    
    # 编译 gf-complete 源文件
    set(GF_COMPLETE_SOURCES
        ${GF_COMPLETE_DIR}/src/gf_cpu.c
        ${GF_COMPLETE_DIR}/src/gf_wgen.c
        ${GF_COMPLETE_DIR}/src/gf_w16.c
        ${GF_COMPLETE_DIR}/src/gf.c
        ${GF_COMPLETE_DIR}/src/gf_w32.c
        ${GF_COMPLETE_DIR}/src/gf_w64.c
        ${GF_COMPLETE_DIR}/src/gf_w128.c
        ${GF_COMPLETE_DIR}/src/gf_general.c
        ${GF_COMPLETE_DIR}/src/gf_w4.c
        ${GF_COMPLETE_DIR}/src/gf_rand.c
        ${GF_COMPLETE_DIR}/src/gf_w8.c
    )
    
    # 编译 jerasure 源文件
    set(JERASURE_SOURCES
        ${JERASURE_DIR}/src/cauchy.c
        ${JERASURE_DIR}/src/galois.c
        ${JERASURE_DIR}/src/jerasure.c
        ${JERASURE_DIR}/src/liberation.c
        ${JERASURE_DIR}/src/reed_sol.c
    )
    
    # 创建静态库
    add_library(gf_complete_lib STATIC ${GF_COMPLETE_SOURCES})
    target_include_directories(gf_complete_lib PUBLIC ${GF_COMPLETE_DIR}/include)
    # 设置 C 标准
    set_target_properties(gf_complete_lib PROPERTIES
        C_STANDARD 99
        C_STANDARD_REQUIRED ON
    )
    
    add_library(jerasure_lib STATIC ${JERASURE_SOURCES})
    target_include_directories(jerasure_lib PUBLIC 
        ${JERASURE_DIR}/include
        ${GF_COMPLETE_DIR}/include
    )
    # Jerasure 依赖 gf-complete
    target_link_libraries(jerasure_lib gf_complete_lib)
    # 设置 C 标准
    set_target_properties(jerasure_lib PROPERTIES
        C_STANDARD 99
        C_STANDARD_REQUIRED ON
    )
    
    set(USE_PROJECT_LIBS TRUE)
else()
    # 尝试查找系统安装的库
    find_library(JERASURE_LIB jerasure PATHS /usr/lib /usr/local/lib)
    find_library(GF_COMPLETE_LIB gf_complete PATHS /usr/lib /usr/local/lib)
    
    if(NOT JERASURE_LIB OR NOT GF_COMPLETE_LIB)
        message(FATAL_ERROR "找不到 Jerasure 或 gf-complete 库。请确保 lib/ec 目录存在或系统已安装这些库")
    endif()
    
    set(USE_PROJECT_LIBS FALSE)
endif()

# 源文件
set(CLAY_SOURCES
    clay/erasure_code_clay.cc
    clay/erasure_code_clay_factory.cc
    jerasure/erasure_code_jerasure.cc
    jerasure/erasure_code_jerasure_factory.cc
    utils/erasure_code.cc
    utils/str_util.cc
)

# 测试程序
add_executable(test_clay test_clay.cc ${CLAY_SOURCES})

# 链接库 - 注意链接顺序：先链接依赖库，再链接主库
if(USE_PROJECT_LIBS)
    # 使用项目中的库
    target_link_libraries(test_clay
        jerasure_lib
        gf_complete_lib
        ${RADOS_LIB}
        ${Boost_LIBRARIES}
    )
else()
    # 使用系统库
    target_link_libraries(test_clay
        ${JERASURE_LIB}
        ${GF_COMPLETE_LIB}
        ${RADOS_LIB}
        ${Boost_LIBRARIES}
    )
endif()

# 编译选项
target_compile_options(test_clay PRIVATE -Wall -Wextra)

# 输出信息
message(STATUS "RADOS_LIB: ${RADOS_LIB}")
message(STATUS "RADOS_INCLUDE_DIR: ${RADOS_INCLUDE_DIR}")

