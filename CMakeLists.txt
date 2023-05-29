cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 20)

add_compile_options(-Wall -O3)

project(md)

if(APPLE)
    link_directories(/usr/local/Cellar/openblas/0.3.21/lib/ ./lib/)
elseif(UNIX)
    link_directories(./lib/OpenBLAS/lib/ ./lib/)
endif()
include_directories(./include/)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

file(GLOB SRCFILES CONFIGURE_DEPENDS "src/*.h" "src/*.cpp")
# file(GLOB SRCFILES CONFIGURE_DEPENDS "src_bc/*.h" "src_bc/*.cpp")
add_executable(nnp.exe ${SRCFILES})
target_link_libraries(nnp.exe openblas tensorflow)
