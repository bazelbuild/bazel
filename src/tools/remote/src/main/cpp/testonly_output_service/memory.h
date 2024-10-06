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

#ifndef BAZEL_SRC_TOOLS_REMOTE_SRC_MAIN_CPP_TESTONLY_OUTPUT_SERVICE_MEMORY_H_
#define BAZEL_SRC_TOOLS_REMOTE_SRC_MAIN_CPP_TESTONLY_OUTPUT_SERVICE_MEMORY_H_

#include <stddef.h>
#include <stdint.h>

static inline size_t KiB(size_t size) { return size * 1024; }
static inline size_t MiB(size_t size) { return KiB(size) * 1024; }
static inline size_t GiB(size_t size) { return MiB(size) * 1024; }

// Returns the number of bytes in a memory page.
size_t GetPageSize();

// Reserves `size` bytes of memory in the current process's memory space. The
// memory is not committed so cannot be accessed. In order to access (sub)region
// of the reserved memory, use CommitMemory().
void *ReserveMemory(size_t size);

// Commits a region of memory previously reserved with ReserveMemory(). The
// memory is accessible and can be read and written after this call.
void CommitMemory(void *ptr, size_t size);

// Releases a region of memory previously reserved with ReserveMemory(). The
// memory is no longer reserved and cannot be accessed.
void ReleaseMemory(void *ptr, size_t size);

// Arena is a memory allocator that allocates memory from a reserved memory
// region in a stack like manner. The memory is committed to OS on demand.
struct Arena {
  // The end of the reserved memory region.
  uint8_t *reserved;
  // The end of the committed memory region.
  uint8_t *committed;
  // The current top of the stack. This is the address of the first byte of
  // memory that is not yet allocated.
  //
  // Invariant: top <= committed <= reserved
  uint8_t *top;
  uint32_t temp_memory_count;
};

Arena *AllocArena(size_t reserve_size = GiB(1));
void FreeArena(Arena *arena);

// PushArena allocates a block of memory of the given size in the arena. The
// returned pointer is aligned to 8 bytes. The memory is zeroed.
void *PushArena(Arena *arena, size_t size);
void PopArena(Arena *arena, size_t size);

#define PushArray(arena, type, count) \
  (type *)PushArena(arena, sizeof(type) * (count))

struct TemporaryMemory {
  Arena *arena;
  uint8_t *top;
};

// Begins temporary memory allocations for the arena. The returned
// TemporaryMemory must be passed to EndTemporaryMemory() when the memory is no
// longer needed.
TemporaryMemory BeginTemporaryMemory(Arena *arena);
// Ends the temporary memory allocations. All memory allocated since
// BeginTemporaryMemory() was called is freed.
void EndTemporaryMempory(TemporaryMemory temp);

// Gets a thread-local arena for temporary use. Use `conflicts` to avoid
// returning the same arena as a previous call to GetScratchArena().
Arena *GetScratchArena(Arena **conflicts, size_t count);

// Begins the use of a thread-local scratch arena. The returned TemporaryMemory
// must be passed to EndScratch() when the memory is no longer needed.
static inline TemporaryMemory BeginScratch(Arena **conflicts, size_t count) {
  return BeginTemporaryMemory(GetScratchArena(conflicts, count));
}

static inline TemporaryMemory BeginScratch(Arena *conflict) {
  if (conflict) {
    return BeginTemporaryMemory(GetScratchArena(&conflict, 1));
  } else {
    return BeginTemporaryMemory(GetScratchArena(0, 0));
  }
}

// Ends the use of a thread-local scratch arena.
static inline void EndScratch(TemporaryMemory scratch) {
  EndTemporaryMempory(scratch);
}

#endif  // BAZEL_SRC_TOOLS_REMOTE_SRC_MAIN_CPP_TESTONLY_OUTPUT_SERVICE_MEMORY_H_
