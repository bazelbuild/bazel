// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util;

import static com.google.common.base.StandardSystemProperty.OS_ARCH;

import com.google.common.collect.ImmutableSet;

/**
 * Detects the CPU of the running JVM and returns a describing enum value.
 */
public enum CPU {
  X86_32("x86_32", ImmutableSet.of("i386", "i486", "i586", "i686", "i786", "x86")),
  X86_64("x86_64", ImmutableSet.of("amd64", "x86_64", "x64")),
  PPC("ppc", ImmutableSet.of("ppc", "ppc64", "ppc64le")),
  ARM("arm", ImmutableSet.of("arm", "armv7l")),
  AARCH64("aarch64", ImmutableSet.of("aarch64")),
  S390X("s390x", ImmutableSet.of("s390x", "s390")),
  MIPS64("mips64", ImmutableSet.of("mips64el", "mips64")),
  RISCV64("riscv64", ImmutableSet.of("riscv64")),
  SW64("sw_64", ImmutableSet.of("sw_64")),
  UNKNOWN("unknown", ImmutableSet.<String>of());

  private final String canonicalName;
  private final ImmutableSet<String> archs;

  CPU(String canonicalName, ImmutableSet<String> archs) {
    this.canonicalName = canonicalName;
    this.archs = archs;
  }

  public String getCanonicalName() {
    return canonicalName;
  }

  private static final CPU HOST_CPU = determineCurrentCpu();

  /**
   * The current CPU.
   */
  public static CPU getCurrent() {
    return HOST_CPU;
  }

  private static CPU determineCurrentCpu() {
    String currentArch = OS_ARCH.value();

    for (CPU cpu : CPU.values()) {
      if (cpu.archs.contains(currentArch)) {
        return cpu;
      }
    }

    return CPU.UNKNOWN;
  }
}
