// Copyright 2026 The Bazel Authors. All rights reserved.
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

#include "src/main/tools/seccomp_exec.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __linux__
#include <linux/seccomp.h>
#include <sys/syscall.h>
#endif  // __linux__

// Seccomp process blocking is supported on Linux x86_64 and aarch64.
#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__))

#include <err.h>
#include <errno.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <sched.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/prctl.h>
#include <unistd.h>

namespace {

// Bazel's hermetic Linux toolchain compiles against Linux 3.9.11 kernel headers
// and glibc 2.17 headers for portability, which predate AUDIT_ARCH_AARCH64
// (3.14), seccomp(2) (3.17), getrandom(2) (3.17 / glibc 2.25), execveat(2)
// (3.19), and clone3(2) (5.3). Provide fallback definitions for these frozen
// Linux UAPI constants and invoke them via syscall(2) so binaries built with
// the hermetic toolchain still work at runtime on modern Linux kernels.
#ifndef SECCOMP_SET_MODE_FILTER
#define SECCOMP_SET_MODE_FILTER 1
#endif
#ifndef __NR_clone3
#define __NR_clone3 435
#endif
#if defined(__x86_64__)
#ifndef __NR_seccomp
#define __NR_seccomp 317
#endif
#ifndef __NR_getrandom
#define __NR_getrandom 318
#endif
#ifndef __NR_execveat
#define __NR_execveat 322
#endif
#ifndef __X32_SYSCALL_BIT
#define __X32_SYSCALL_BIT 0x40000000
#endif
#elif defined(__aarch64__)
#ifndef AUDIT_ARCH_AARCH64
// EM_AARCH64 (183) | __AUDIT_ARCH_64BIT | __AUDIT_ARCH_LE.
#define AUDIT_ARCH_AARCH64 0xc00000b7U
#endif
#ifndef __NR_seccomp
#define __NR_seccomp 277
#endif
#ifndef __NR_getrandom
#define __NR_getrandom 278
#endif
#ifndef __NR_execveat
#define __NR_execveat 281
#endif
#endif

#if defined(__x86_64__)
constexpr unsigned int kExpectedAuditArch = AUDIT_ARCH_X86_64;
#elif defined(__aarch64__)
constexpr unsigned int kExpectedAuditArch = AUDIT_ARCH_AARCH64;
#endif

// Helpers for classic BPF (cBPF) seccomp filters.
// In cBPF:
// - A is the 32-bit accumulator register and X is the 32-bit index register.
// - BPF_LD | BPF_W | BPF_ABS loads a 32-bit word from `struct seccomp_data` at
//   an absolute byte offset into A.
// - BPF_K uses the immediate constant operand.
// - BPF_JUMP(op, k, jt, jf) skips `jt` instructions if the comparison is true
//   or `jf` instructions if false.

// Byte offsets of the low and high 32-bit halves of syscall argument `n`
// (little-endian).
#define ARG_LO(n) (offsetof(seccomp_data, args) + 8 * (n))
#define ARG_HI(n) (ARG_LO(n) + 4)
#define LOAD(offset) BPF_STMT(BPF_LD | BPF_W | BPF_ABS, (offset))
#define RETURN(action) BPF_STMT(BPF_RET | BPF_K, (action))
// If A (the loaded syscall number) == `nr`, return `action`; else skip 1.
#define ON_SYSCALL(nr, action) \
  BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, (nr), 0, 1), RETURN(action)

void FillRandomBytes(void* buf, size_t len) {
  auto* out = static_cast<unsigned char*>(buf);
  size_t filled = 0;
  while (filled < len) {
    ssize_t n = syscall(__NR_getrandom, out + filled, len - filled, 0);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      err(EXIT_FAILURE, "Failed to generate seccomp exec cookie");
    }
    filled += static_cast<size_t>(n);
  }
}

}  // namespace

// exec()s a given binary, which is then not allowed to exec() any other child
// processes anymore.
//
// This needs to be done in a slightly elaborate way because we do need to run
// execve() exactly once, but a cBPF filter cannot count. Instead, the filter
// only lets through an execve() whose unused fourth and fifth arguments (128
// bits total), which execve() ignores but the filter can see, match a true
// random cookie.
//
// The cookie does not leak to the exec'd process:
// - Registers: execve() zeroes all general-purpose and FPU/SIMD registers in
//   the kernel (start_thread()) before entering the new binary.
// - Memory: execve() replaces the entire address space (mm_struct), unmapping
//   this process's stack pages (which are zeroed before reuse).
// - Filter: the cBPF program lives in kernel memory and cannot be read back by
//   an unprivileged process (PTRACE_SECCOMP_GET_FILTER requires CAP_SYS_ADMIN).
// - Timing/caches: the filter compares all four 32-bit words in constant time
//   with immediate constants (no secret-dependent branches or memory accesses).
void ExecWithSeccompProcessBlocker(const char* exe, char* const* argv) {
  // Zero-initialize so MemorySanitizer treats `cookie` as initialized even
  // though FillRandomBytes() populates it via raw syscall(__NR_getrandom, ...).
  uint64_t cookie[2] = {};
  FillRandomBytes(cookie, sizeof(cookie));

  // The four 32-bit halves of the 128-bit cookie, in seccomp_data order.
  const uint32_t c[] = {
      static_cast<uint32_t>(cookie[0]),
      static_cast<uint32_t>(cookie[0] >> 32),
      static_cast<uint32_t>(cookie[1]),
      static_cast<uint32_t>(cookie[1] >> 32),
  };

  constexpr uint32_t kDenyEperm =
      SECCOMP_RET_ERRNO | (EPERM & SECCOMP_RET_DATA);
  constexpr uint32_t kReturnEnosys =
      SECCOMP_RET_ERRNO | (ENOSYS & SECCOMP_RET_DATA);

  // Classic BPF filter forbidding child process creation and exec() while
  // allowing new threads and the single execve() below that starts the Bazel
  // server.
  sock_filter filter[] = {
      // Load seccomp_data.arch; if it matches our ABI, skip 1 (past DENY).
      // Rejects 32-bit compat syscalls (e.g. int 0x80 on x86_64) whose syscall
      // numbers differ from our 64-bit table.
      LOAD(offsetof(seccomp_data, arch)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, kExpectedAuditArch, 1, 0),
      RETURN(kDenyEperm),

      // Load seccomp_data.nr into A for all subsequent syscall checks.
      LOAD(offsetof(seccomp_data, nr)),

#if defined(__x86_64__)
      // Reject x32 ABI syscalls (bit 30 set in nr) with EPERM.
      BPF_JUMP(BPF_JMP | BPF_JSET | BPF_K, __X32_SYSCALL_BIT, 0, 1),
      RETURN(kDenyEperm),
#endif

      // Deny clone3() with ENOSYS so libc falls back to clone(); clone3()
      // passes flags via a userspace pointer that cBPF cannot safely inspect.
      ON_SYSCALL(__NR_clone3, kReturnEnosys),

      // Deny execveat() with EPERM.
      ON_SYSCALL(__NR_execveat, kDenyEperm),

  // aarch64 has no fork() or vfork() syscalls; deny them where present.
#ifdef __NR_fork
      ON_SYSCALL(__NR_fork, kDenyEperm),
#endif
#ifdef __NR_vfork
      ON_SYSCALL(__NR_vfork, kDenyEperm),
#endif

      // clone(): allowed only when CLONE_THREAD is set in the low 32 bits of
      // args[0] (clone_flags on both x86_64 and aarch64). If A != __NR_clone,
      // skip the 4 instructions below to the next rule.
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_clone, 0, 4),
      // A = low 32 bits of seccomp_data.args[0].
      LOAD(ARG_LO(0)),
      // If (A & CLONE_THREAD) != 0, skip 1 to ALLOW; else fall through to DENY.
      BPF_JUMP(BPF_JMP | BPF_JSET | BPF_K, CLONE_THREAD, 1, 0),
      RETURN(kDenyEperm),
      RETURN(SECCOMP_RET_ALLOW),

      // Allow every syscall other than execve() (if A == __NR_execve, skip 1
      // past ALLOW into the cookie check below).
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_execve, 1, 0),
      RETURN(SECCOMP_RET_ALLOW),

      // execve(): allowed only if unused args 3 and 4 carry the 128-bit random
      // cookie. Constant-time: XOR each 32-bit half with its expected value,
      // OR all four differences together via X, and branch once at the end.
      // X = args[3]_lo ^ c[0].
      LOAD(ARG_LO(3)),
      BPF_STMT(BPF_ALU | BPF_XOR | BPF_K, c[0]),
      BPF_STMT(BPF_MISC | BPF_TAX, 0),
      // X |= args[3]_hi ^ c[1].
      LOAD(ARG_HI(3)),
      BPF_STMT(BPF_ALU | BPF_XOR | BPF_K, c[1]),
      BPF_STMT(BPF_ALU | BPF_OR | BPF_X, 0),
      BPF_STMT(BPF_MISC | BPF_TAX, 0),
      // X |= args[4]_lo ^ c[2].
      LOAD(ARG_LO(4)),
      BPF_STMT(BPF_ALU | BPF_XOR | BPF_K, c[2]),
      BPF_STMT(BPF_ALU | BPF_OR | BPF_X, 0),
      BPF_STMT(BPF_MISC | BPF_TAX, 0),
      // A = X | (args[4]_hi ^ c[3]).
      LOAD(ARG_HI(4)),
      BPF_STMT(BPF_ALU | BPF_XOR | BPF_K, c[3]),
      BPF_STMT(BPF_ALU | BPF_OR | BPF_X, 0),
      // If A == 0 (all 128 bits matched), fall through to ALLOW; else skip 1
      // to DENY.
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 0, 0, 1),
      RETURN(SECCOMP_RET_ALLOW),
      RETURN(kDenyEperm),
  };

  if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
    err(EXIT_FAILURE, "Failed to set PR_SET_NO_NEW_PRIVS");
  }

  sock_fprog prog = {
      .len = static_cast<unsigned short>(sizeof(filter) / sizeof(filter[0])),
      .filter = filter,
  };
  if (syscall(__NR_seccomp, SECCOMP_SET_MODE_FILTER, 0, &prog) != 0) {
    err(EXIT_FAILURE, "Failed to install seccomp process-blocking filter");
  }

  syscall(__NR_execve, exe, argv, environ, cookie[0], cookie[1]);
  err(EXIT_FAILURE, "Failed to execute %s", exe);
}

#else  // Linux x86_64/aarch64

void ExecWithSeccompProcessBlocker(const char* exe, char* const* argv) {
  (void)exe;
  (void)argv;
  fprintf(stderr,
          "Seccomp process blocker (--noallow_server_subprocesses) "
          "is only supported on Linux (x86_64 and aarch64).\n");
  exit(EXIT_FAILURE);
}

#endif  // Linux x86_64/aarch64
