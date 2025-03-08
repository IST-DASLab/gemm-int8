cmake_minimum_required(VERSION 3.18)
project(gemm_int8 LANGUAGES CXX)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Set output directories for all build artifacts
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find Python executable
if(NOT DEFINED Python3_EXECUTABLE)
    find_program(Python3_EXECUTABLE NAMES python3 python)
    if(NOT Python3_EXECUTABLE)
        message(FATAL_ERROR "Python3 executable not found. Please specify with -DPython3_EXECUTABLE=path/to/python")
    endif()
endif()
message(STATUS "Using Python executable: ${Python3_EXECUTABLE}")

# Find Python package
find_package(Python3 COMPONENTS Development REQUIRED)
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")

# Get Python include directories
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Python include directory: ${PYTHON_INCLUDE_DIR}")

# Find PyTorch
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    RESULT_VARIABLE PYTORCH_RESULT
    OUTPUT_VARIABLE TORCH_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT PYTORCH_RESULT EQUAL 0)
    message(FATAL_ERROR "PyTorch not found. Please install PyTorch first.")
endif()
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PREFIX_PATH})

# Enable CUDA
if(NOT DEFINED BUILD_CUDA)
    set(BUILD_CUDA ON)
endif()

if(BUILD_CUDA)
    # NVCC compatibility check for newer MSVC compilers
    if(MSVC AND MSVC_VERSION VERSION_GREATER_EQUAL 1940)
        string(APPEND CMAKE_CUDA_FLAGS " --allow-unsupported-compiler")
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Convert the CUDA version from X.Y.z to XY
    string(REGEX MATCH "^[0-9]+.[0-9]+" _CUDA_VERSION_FIRST_TWO "${CMAKE_CUDA_COMPILER_VERSION}")
    string(REPLACE "." "" CUDA_VERSION_SHORT "${_CUDA_VERSION_FIRST_TWO}")

    message(STATUS "CUDA Version: ${CUDA_VERSION_SHORT} (${CMAKE_CUDA_COMPILER_VERSION})")
    message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")

    # IMPORTANT: This is the key change - disable PyTorch's architecture detection
    set(TORCH_CUDA_ARCH_LIST "")

    # Default architectures if not provided
    if(NOT DEFINED COMPUTE_CAPABILITY)
        set(COMPUTE_CAPABILITY "70;75;80;86;89;90;90a" CACHE STRING "CUDA Compute Capabilities")
    endif()
    
    message(STATUS "CUDA Capabilities Selected: ${COMPUTE_CAPABILITY}")
    
    # Configure architectures for compilation - explicitly set with our choices
    set(CMAKE_CUDA_ARCHITECTURES ${COMPUTE_CAPABILITY})
    
    # Set explicit NVCC flags to override any auto-detection
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --use_fast_math")
    
    # Add explicit architecture flags to NVCC
    foreach(ARCH ${COMPUTE_CAPABILITY})
        string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_${ARCH},code=sm_${ARCH}")
    endforeach()
    
    # For the latest architecture, also add PTX
    list(GET COMPUTE_CAPABILITY -1 LATEST_ARCH)
    string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_${LATEST_ARCH},code=compute_${LATEST_ARCH}")
    
    message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")
    
    # Set C++ standard for CUDA
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Define that we're building with CUDA
    add_compile_definitions(BUILD_CUDA)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include CUTLASS headers (without building the entire library)
include_directories(${CMAKE_SOURCE_DIR}/cutlass/include)
include_directories(${CMAKE_SOURCE_DIR}/cutlass/tools/util/include)

# Setup include directories
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/csrc/kernels/include)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIR})

# Find PyTorch - IMPORTANT: Do this after setting TORCH_CUDA_ARCH_LIST
find_package(Torch REQUIRED)
message(STATUS "Found PyTorch: ${TORCH_INCLUDE_DIRS}")

# Create source files list
set(CPP_FILES csrc/kernels/bindings.cpp)
set(CUDA_FILES csrc/kernels/gemm.cu)

# Add source files based on backend
if(BUILD_CUDA)
    set(SRC_FILES ${CPP_FILES} ${CUDA_FILES})
    set(OUTPUT_NAME "gemm_int8_CUDA")
else()
    set(SRC_FILES ${CPP_FILES})
    set(OUTPUT_NAME "gemm_int8_CPU")
endif()

# Create the extension library
add_library(gemm_int8 SHARED ${SRC_FILES})

# Link dependencies
if(BUILD_CUDA)
    target_link_libraries(gemm_int8 PRIVATE 
        "${TORCH_LIBRARIES}" 
        Python3::Python 
        CUDA::cudart 
        CUDA::cublas
    )
else()
    target_link_libraries(gemm_int8 PRIVATE 
        "${TORCH_LIBRARIES}" 
        Python3::Python
    )
endif()

target_include_directories(gemm_int8 PRIVATE 
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIR}
)

# Set output properties
set_target_properties(gemm_int8 PROPERTIES
    OUTPUT_NAME "${OUTPUT_NAME}"
    PREFIX ""
)

# Configure output directories based on platform
if(WIN32)
    # Windows-specific settings
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    
    if(MSVC)
        set_target_properties(gemm_int8 PROPERTIES 
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/gemm_int8"
            RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/gemm_int8"
        )
    endif()
else()
    # Linux/macOS settings
    set_target_properties(gemm_int8 PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/gemm_int8"
    )
endif()

# Make a custom command to copy the built library to the Python package
add_custom_command(
    TARGET gemm_int8
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:gemm_int8>
    "${CMAKE_SOURCE_DIR}/gemm_int8/$<TARGET_FILE_NAME:gemm_int8>"
    COMMENT "Copying library to Python package directory"
)

# Debug info
message(STATUS "Source files: ${SRC_FILES}")
message(STATUS "Library will be copied to: ${CMAKE_SOURCE_DIR}/gemm_int8/$<TARGET_FILE_NAME:gemm_int8>")

# Print architecture settings again at the end to confirm
if(BUILD_CUDA)
    message(STATUS "Final CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "Final CUDA flags: ${CMAKE_CUDA_FLAGS}")
endif()