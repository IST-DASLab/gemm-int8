#!/bin/bash
declare build_arch
declare build_os
declare cuda_version

set -xeuo pipefail
build_capability="80;86;89;90;100;120"
remove_for_11_8=";100;120"
remove_for_lt_12_7=";100;120"
[[ "${cuda_version}" == 11.8.* ]] && build_capability=$(sed 's|'"$remove_for_11_8"'||g' <<< "$build_capability")
[[ "${cuda_version}" < 12.7 ]] && build_capability=$(sed 's|'"$remove_for_lt_12_7"'||g; s|'"${remove_for_lt_12_7#;}"';||g' <<< "$build_capability")

if [ "${build_os:0:6}" == ubuntu ]; then
    image=nvidia/cuda:${cuda_version}-devel-ubuntu22.04
    echo "Using image $image"
    docker run --platform "linux/$build_arch" -i -w /src -v "$PWD:/src" "$image" sh -c \
        "apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cmake \
    && cmake -DPTXAS_VERBOSE=1 -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY=\"${build_capability}\" . \
    && cmake --build ."
else
    pip install cmake==3.28.3
    cmake -G Ninja -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY="${build_capability}" -DCMAKE_BUILD_TYPE=Release -S .
    cmake --build . --config Release
fi


output_dir="output/${build_os}/${build_arch}"
mkdir -p "${output_dir}"
(shopt -s nullglob && cp gemm_int8/*.{so,dylib,dll} "${output_dir}")
