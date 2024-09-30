// Copyright 2024 The Bazel Authors. All rights reserved.
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

#include <assert.h>
#include <stddef.h>
#include <sys/mman.h>
#include <unistd.h>

#include "src/tools/remote/src/main/cpp/testonly_output_service/memory.h"

size_t GetPageSize() {
  size_t result = getpagesize();
  return result;
}

void *ReserveMemory(size_t size) {
  void *result = mmap(0, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(result != MAP_FAILED && "Failed to reserve memory");
  return result;
}

void CommitMemory(void *ptr, size_t size) {
  bool result = mprotect(ptr, size, PROT_READ | PROT_WRITE) == 0;
  assert(result && "Failed to commit memory");
}

void ReleaseMemory(void *ptr, size_t size) {
  bool result = munmap(ptr, size) == 0;
  assert(result && "Failed to release memory");
}
