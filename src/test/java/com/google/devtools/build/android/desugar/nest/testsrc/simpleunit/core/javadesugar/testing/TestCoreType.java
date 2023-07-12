/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package javadesugar.testing;

/**
 * A fake core library class for testing core type desugaring. Related flags include,
 *
 * <p>--core_library, --desugar_supported_core_libs, --rewrite_core_library_prefix
 */
public class TestCoreType {

  /** Invocation entry point for testing to invoke private static methods in anther mate. */
  public static long twoSum(long x, long y) {
    return MateA.twoSum(x, y);
  }

  /** Invocation entry point for testing to invoke private instance methods in anther mate. */
  public static long twoSumWithBase(long base, long x, long y) {
    return new MateA(base).twoSumWithBase(x, y);
  }

  private TestCoreType() {}

  private static class MateA {

    private final long base;

    private MateA(long base) {
      this.base = base;
    }

    private static long twoSum(long x, long y) {
      return x + y;
    }

    private long twoSumWithBase(long x, long y) {
      return base + x + y;
    }
  }
}
