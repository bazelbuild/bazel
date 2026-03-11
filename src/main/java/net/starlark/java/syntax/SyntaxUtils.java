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

package net.starlark.java.syntax;

/** Utility methods for Starlark syntax. */
public final class SyntaxUtils {

  private SyntaxUtils() {}

  /**
   * Returns the effective bound for a positive-stride slice operation from a user-supplied integer.
   * First, if the integer is negative, the length of the sequence is added to it, so an index of -1
   * represents the last element of the sequence. Then, the integer is "clamped" into the inclusive
   * interval [0, length].
   */
  public static int toSliceBound(int index, int length) {
    if (index < 0) {
      index += length;
    }

    if (index < 0) {
      return 0;
    } else if (index > length) {
      return length;
    } else {
      return index;
    }
  }

  /**
   * Returns the effective bound for a negative-stride slice operation from a user-supplied integer.
   * First, if the integer is negative, the length of the sequence is added to it, so an index of -1
   * represents the last element of the sequence. Then, the integer is "clamped" into the inclusive
   * interval [-1, length - 1].
   */
  public static int toReverseSliceBound(int index, int length) {
    if (index < 0) {
      index += length;
    }

    if (index < -1) {
      return -1;
    } else if (index >= length) {
      return length - 1;
    } else {
      return index;
    }
  }
}
