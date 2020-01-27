// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>

#include "third_party/ijar/common.h"
#include "third_party/ijar/zlib_client.h"

namespace devtools_ijar {

u4 ComputeCrcChecksum(u1* buf, size_t length) { return 0; }

size_t TryDeflate(u1* buf, size_t length) { return 0; }

Decompressor::Decompressor() {}
Decompressor::~Decompressor() {}

DecompressedFile* Decompressor::UncompressFile(const u1* buffer,
                                                      size_t bytes_avail) {
  return NULL;
}

char* Decompressor::GetError() { return NULL; }

int Decompressor::error(const char* fmt, ...) { return 0; }
}  // namespace devtools_ijar
