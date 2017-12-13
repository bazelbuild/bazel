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
package com.google.devtools.build.android.desugar;

import org.objectweb.asm.Opcodes;

/**
 * Convenience method for working with {@code int} bitwise flags.
 */
class BitFlags {

  /**
   * Returns {@code true} iff <b>all</b> bits in {@code bitmask} are set in {@code flags}.
   * Trivially returns {@code true} if {@code bitmask} is 0.
   */
  public static boolean isSet(int flags, int bitmask) {
    return (flags & bitmask) == bitmask;
  }

  /**
   * Returns {@code true} iff <b>none</b> of the bits in {@code bitmask} are set in {@code flags}.
   * Trivially returns {@code true} if {@code bitmask} is 0.
   */
  public static boolean noneSet(int flags, int bitmask) {
    return (flags & bitmask) == 0;
  }

  public static boolean isInterface(int access) {
    return isSet(access, Opcodes.ACC_INTERFACE);
  }

  public static boolean isStatic(int access) {
    return isSet(access, Opcodes.ACC_STATIC);
  }

  public static boolean isSynthetic(int access) {
    return isSet(access, Opcodes.ACC_SYNTHETIC);
  }

  // Static methods only
  private BitFlags() {}
}
