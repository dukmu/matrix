cmake_minimum_required(VERSION 3.8)
# Policy CMP0087
cmake_policy(SET CMP0087 NEW)
project(vo1)

add_definitions("-DENABLE_SSE")
set(SSE_FLAGS "-msse4 -mfma")
# set(AVX_FLAGS "-mavx2 -mfma -O3")
set(flags_gcc "-std=c++20 ${SSE_FLAGS} -fopenmp -static-libgcc -static-libstdc++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flags_gcc}")
# if Release then set -O3
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${flags_gcc} -O3")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/install)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# if windows
if (WIN32)
   set(OpenCV_DIR D:/project/ar/lib/opencv/install)
endif()

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
set(base_dir ${PROJECT_SOURCE_DIR})

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${base_dir}
        ${base_dir}/include
)

file(GLOB lib_src ${base_dir}/src/*.cpp)

set(deps_gcc ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
set(BUILD_SHARED_LIBS OFF)

add_library(lib ${lib_src})
target_compile_definitions(lib PRIVATE -DTIMEIT_DISABLE -DDEBUG_DO_DISABLE)
target_link_libraries(lib ${deps_gcc})

add_library(lib_debug ${lib_src})
target_compile_definitions(lib_debug PRIVATE -DTIMEIT_ENABLE -DDEBUG_DO_ENABLE)
target_link_libraries(lib_debug ${deps_gcc})

add_library(lib_timeit ${lib_src})
target_compile_definitions(lib_timeit PRIVATE -DTIMEIT_ENABLE -DDEBUG_DISABLE)
target_link_libraries(lib_timeit ${deps_gcc})

add_executable(main ${base_dir}/main.cpp) # with timeit and no debug output
target_compile_definitions(main PRIVATE -DTIMEIT_DISABLE -DDEBUG_DO_DISABLE)
target_link_libraries(main ${deps_gcc} lib)

add_executable(main_debug ${base_dir}/main.cpp) # with timeit and debug output
target_compile_definitions(main_debug PRIVATE -DTIMEIT_ENABLE -DDEBUG_DO_ENABLE)
target_link_libraries(main_debug ${deps_gcc} lib_debug)

add_executable(main_timeit ${base_dir}/main.cpp) # with timeit and no debug output
target_compile_definitions(main_timeit PRIVATE -DTIMEIT_ENABLE -DDEBUG_DO_DISABLE)
target_link_libraries(main_timeit ${deps_gcc} lib_timeit)

# install(TARGETS main
#         RUNTIME DESTINATION bin
#         LIBRARY DESTINATION lib
#         ARCHIVE DESTINATION lib
# )

# install runtime library, searching from mingw64 folder
# install(CODE [[
# file(GET_RUNTIME_DEPENDENCIES
#         RESOLVED_DEPENDENCIES_VAR _r_deps
#         UNRESOLVED_DEPENDENCIES_VAR _u_deps
#         EXECUTABLES $<TARGET_FILE:main> $<TARGET_FILE:diff_img> $<TARGET_FILE:speedtest> $<TARGET_FILE:guassian_mask>
#         DIRECTORIES D:/project/ar/lib/opencv/install/x64/mingw/bin D:/imp-applications/Exec/msys2/mingw64/bin
#         PRE_EXCLUDE_REGEXES "api-ms-win-*.dll"
#         PRE_EXCLUDE_REGEXES "system32"
#         POST_EXCLUDE_REGEXES "api-ms-win-*.dll"
#         POST_EXCLUDE_REGEXES "system32"
# )
# foreach(_file ${_r_deps})
#     message(STATUS "Copying ${_file}")
#     file(INSTALL ${_file} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
# endforeach()
# foreach(_file ${_u_deps})
#     message(STATUS "Unresolved ${_file}")
# endforeach()
# ]])

include (InstallRequiredSystemLibraries)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(CPACK_GENERATOR "ZIP")
include(CPack)
