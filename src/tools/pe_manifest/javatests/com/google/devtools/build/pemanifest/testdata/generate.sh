#!/usr/bin/env bash
# Copyright 2026 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Regenerates the minimal Windows executables used by PeManifestTest with the
# LLVM tools of the hermetic toolchain. Usage: generate.sh <llvm bin dir>
set -euo pipefail
bin=$1
cd "$(dirname "$0")"
cp ../../../../../../../../../main/cpp/bazel.ico .
trap 'rm -f bazel.ico res.res no_manifest.res stub64.o stub32.o no_rsrc.pdb' EXIT
"$bin/llvm-rc" /fo res.res res.rc
"$bin/llvm-rc" /fo no_manifest.res no_manifest.rc
"$bin/clang" -target x86_64-w64-windows-gnu -O2 -c stub.c -o stub64.o
"$bin/clang" -target i686-w64-windows-gnu -O2 -c stub.c -o stub32.o
# /release sets the PE checksum, which the test verifies.
common=(/entry:mainCRTStartup /subsystem:console /nodefaultlib /release /Brepro)
# .rsrc is followed by .reloc, which has to be moved when the manifest grows.
"$bin/lld-link" "${common[@]}" /out:with_reloc.exe stub64.o res.res
# .rsrc is the last section and can simply be extended.
"$bin/lld-link" "${common[@]}" /fixed /out:rsrc_last.exe stub64.o res.res
# 32-bit image with a PE32 optional header.
"$bin/lld-link" "${common[@]}" /machine:x86 /out:pe32.exe stub32.o res.res
# Has resources, but no manifest.
"$bin/lld-link" "${common[@]}" /out:no_manifest.exe stub64.o no_manifest.res
# Has no resource section at all, like a GraalVM native image. The debug directory keeps the
# section table tightly packed and references the CodeView record by file offset.
"$bin/lld-link" "${common[@]}" /debug /pdbaltpath:no_rsrc.pdb /merge:.rdata=.data \
  /out:no_rsrc.exe stub64.o
