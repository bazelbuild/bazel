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

package com.google.devtools.build.android.desugar.nest.testsrc.simpleunit.method;

/**
 * Source code for testing the desugaring of invokevirtual at call the site of private instance
 * method within the same class.
 */
public final class NonNest {

  private final long base;

  private NonNest(long base) {
    this.base = base;
  }

  private long twoSum(long x, long y) {
    return base + x + y;
  }

  public static long invokeTwoSum(long base, long x, long y) {
    return new NonNest(base) // Expected invokespecial
        .twoSum(x, y); // Expected invokevirtual under javac 11.
  }
}
