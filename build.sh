#!/bin/bash
set -euo pipefail

# Default values
build_arch=$(uname -m)
build_os=$(uname -s | tr '[:upper:]' '[:lower:]')
cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)

# Find Python executable
PYTHON_EXECUTABLE=$(which python3 || which python)
if [ -z "$PYTHON_EXECUTABLE" ]; then
  echo "ERROR: Could not find Python executable"
  exit 1
fi
echo "Using Python executable: $PYTHON_EXECUTABLE"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --cuda)
      cuda_version="$2"
      shift 2
      ;;
    --arch)
      build_arch="$2"
      shift 2
      ;;
    --os)
      build_os="$2"
      shift 2
      ;;
    --wheel)
      build_wheel=true
      shift
      ;;
    --python)
      PYTHON_EXECUTABLE="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [--cuda version] [--arch architecture] [--os operating_system] [--wheel] [--python python_executable]"
      echo "Example: $0 --cuda 12.1.0 --arch x86_64 --os ubuntu --python /usr/bin/python3.10"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Building for CUDA $cuda_version on $build_os/$build_arch"

# Get libtorch path using the specified Python executable
LIBTORCH_PATH=$("$PYTHON_EXECUTABLE" -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null || echo "")
if [ -z "$LIBTORCH_PATH" ]; then
  echo "ERROR: Could not find libtorch path. Is PyTorch installed?"
  exit 1
fi
echo "Found libtorch at: $LIBTORCH_PATH"

# Set appropriate compute capabilities based on CUDA version
# Default to common architectures
build_capability="80;86;89;90"

# Adjust for specific CUDA versions
if [[ "${cuda_version}" == 11.* ]]; then
    # For CUDA 11.x, include Ampere but exclude Hopper
    build_capability="70;75;80;86"
elif [[ "${cuda_version}" == 12.* ]]; then
    if [[ "${cuda_version}" == 12.[0-6].* ]]; then
        # CUDA 12.0-12.6: Include Ampere and Hopper
        build_capability="70;75;80;86;89;90"
    else
        # CUDA 12.7+: Include Blackwell
        build_capability="70;75;80;86;89;90;100"
    fi
fi

# Create build directory
mkdir -p build

# Check if ninja is available
if command -v ninja &> /dev/null; then
    echo "Ninja build system found, using it for faster builds"
    USE_NINJA=true
    GENERATOR="-GNinja"
else
    echo "Ninja not found, using default make generator"
    USE_NINJA=false
    GENERATOR=""
    
    # If on Windows without Ninja, use NMake
    if [[ "${build_os}" == "windows"* ]]; then
        GENERATOR="-G\"NMake Makefiles\""
    fi
fi

# Build process
echo "Starting build..."
cd build
cmake ${GENERATOR} \
    -DBUILD_CUDA=ON \
    -DCOMPUTE_CAPABILITY="${build_capability}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH" \
    -DPython3_EXECUTABLE="$PYTHON_EXECUTABLE" \
    ..

# Build with appropriate command
if $USE_NINJA; then
    ninja
elif [[ "${build_os}" == "windows"* ]]; then
    nmake
else
    make -j$(nproc)
fi
cd ..

# Create output directory structure
output_dir="output/${build_os}/${build_arch}"
mkdir -p "${output_dir}"

# Copy the built libraries to the output directory
echo "Copying libraries to ${output_dir}"
if [[ "${build_os}" == "windows"* ]]; then
    cp gemm_int8/*.dll "${output_dir}" 2>/dev/null || true
elif [[ "${build_os}" == "darwin" ]]; then
    cp gemm_int8/*.dylib "${output_dir}" 2>/dev/null || true
else
    cp gemm_int8/*.so "${output_dir}" 2>/dev/null || true
fi

# Check if any libraries were copied
if [ -z "$(ls -A ${output_dir})" ]; then
    echo "WARNING: No libraries were built or copied to output dir!"
    echo "Check if library exists in gemm_int8 directory:"
    ls -la gemm_int8/
    
    if [ -z "$(ls -A gemm_int8/*.{so,dll,dylib} 2>/dev/null)" ]; then
      echo "ERROR: No built libraries found! Build failed."
      exit 1
    else
      echo "Libraries found in gemm_int8/ but not copied to output."
    fi
else
    echo "Build successful! Libraries copied to ${output_dir}"
    ls -lah "${output_dir}"
fi

# Build Python wheel if requested
if [[ "${build_wheel:-false}" == true ]]; then
    echo "Building Python wheel..."
    "$PYTHON_EXECUTABLE" setup.py bdist_wheel
    echo "Wheel built in dist/ directory"
    ls -la dist/
fi 

cuobjdump -lelf gemm_int8/gemm_int8_CUDA.so
