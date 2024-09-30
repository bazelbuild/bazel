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

#include "src/tools/remote/src/main/cpp/testonly_output_service/memory.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static inline size_t AlignSize(size_t addr, size_t align) {
  size_t mod = addr % align;
  return mod == 0 ? addr : (addr - mod + align);
}

static inline uint8_t *GetArenaBase(Arena *arena) {
  return (uint8_t *)(arena + 1);
}

Arena *AllocArena(size_t reserve_size) {
  size_t page_size = GetPageSize();
  size_t commit_size = AlignSize(sizeof(Arena), page_size);
  reserve_size = AlignSize(reserve_size, page_size);
  assert(sizeof(Arena) <= commit_size && commit_size <= reserve_size);
  Arena *arena = (Arena *)ReserveMemory(reserve_size);
  CommitMemory(arena, commit_size);
  arena->top = GetArenaBase(arena);
  arena->reserved = (uint8_t *)arena + reserve_size;
  arena->committed = (uint8_t *)arena + commit_size;
  return arena;
}

void FreeArena(Arena *arena) {
  size_t reserved_size = arena->reserved - (uint8_t *)arena;
  ReleaseMemory(arena, reserved_size);
}

void *PushArena(Arena *arena, size_t size) {
  size_t align = 8;

  uint8_t *result = (uint8_t *)AlignSize((size_t)arena->top, align);
  arena->top = result + size;
  assert(arena->top <= arena->reserved && "OOM");

  if (arena->top > arena->committed) {
    uint8_t *new_committed =
        (uint8_t *)AlignSize((size_t)arena->top, GetPageSize());
    if (new_committed > arena->reserved) {
      new_committed = arena->reserved;
    }
    size_t to_commit = new_committed - arena->committed;
    CommitMemory(arena->committed, to_commit);
    arena->committed = new_committed;
    assert(arena->top <= arena->committed);
  }

  memset(result, 0, size);
  return result;
}

void PopArena(Arena *arena, size_t size) {
  arena->top -= size;
  assert(arena->top >= GetArenaBase(arena));
}
