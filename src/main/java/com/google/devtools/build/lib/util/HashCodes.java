// Copyright 2023 The Bazel Authors. All rights reserved.
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

import java.util.Objects;
import javax.annotation.Nullable;

/**
 * Helper library for hash codes that generates less garbage than {@link Objects#hash}.
 *
 * <p>This uses the same underlying algorithm as the {@link Objects#hash} but explicitly overloads
 * for different argument counts instead of using varargs. The benefit is that it generates no
 * garbage.
 *
 * <p>{@link Object#hashCode} implementations tend to be garbage hotspots in Bazel, especially with
 * hashmaps.
 */
public final class HashCodes {
  /**
   * Value to use when composing two hash codes.
   *
   * <p>31 is prime so modulo-multiplication is invertible (it spans the full range of values).
   * Multiplication by 31 also efficient because {@code 31 * x} optimizes down to {@code (x << 5) -
   * x}.
   */
  public static final int MULTIPLIER = 31;

  private HashCodes() {}

  /** Returns the hashCode of a given object if it is non-null and 0 otherwise. */
  public static int hashObject(@Nullable Object obj) {
    return obj == null ? 0 : obj.hashCode();
  }

  public static int hashObjects(@Nullable Object o1, @Nullable Object o2) {
    int h1 = hashObject(o1);
    int h2 = hashObject(o2);
    return h2 + (MULTIPLIER * (h1 + MULTIPLIER));
  }

  public static int hashObjects(@Nullable Object o1, @Nullable Object o2, @Nullable Object o3) {
    int h1 = hashObject(o1);
    int h2 = hashObject(o2);
    int h3 = hashObject(o3);
    return h3 + MULTIPLIER * (h2 + (MULTIPLIER * (h1 + MULTIPLIER)));
  }

  public static int hashObjects(
      @Nullable Object o1, @Nullable Object o2, @Nullable Object o3, @Nullable Object o4) {
    int h1 = hashObject(o1);
    int h2 = hashObject(o2);
    int h3 = hashObject(o3);
    int h4 = hashObject(o4);
    return h4 + MULTIPLIER * (h3 + MULTIPLIER * (h2 + (MULTIPLIER * (h1 + MULTIPLIER))));
  }

  public static int hashObjects(
      @Nullable Object o1,
      @Nullable Object o2,
      @Nullable Object o3,
      @Nullable Object o4,
      @Nullable Object o5) {
    int h1 = hashObject(o1);
    int h2 = hashObject(o2);
    int h3 = hashObject(o3);
    int h4 = hashObject(o4);
    int h5 = hashObject(o5);
    return h5
        + MULTIPLIER
            * (h4 + MULTIPLIER * (h3 + MULTIPLIER * (h2 + (MULTIPLIER * (h1 + MULTIPLIER)))));
  }

  public static int hashObjects(
      @Nullable Object o1,
      @Nullable Object o2,
      @Nullable Object o3,
      @Nullable Object o4,
      @Nullable Object o5,
      @Nullable Object o6) {
    int h1 = hashObject(o1);
    int h2 = hashObject(o2);
    int h3 = hashObject(o3);
    int h4 = hashObject(o4);
    int h5 = hashObject(o5);
    int h6 = hashObject(o6);
    return h6
        + MULTIPLIER
            * (h5
                + MULTIPLIER
                    * (h4
                        + MULTIPLIER
                            * (h3 + MULTIPLIER * (h2 + (MULTIPLIER * (h1 + MULTIPLIER))))));
  }

  public static int hashObjects(
      @Nullable Object o1,
      @Nullable Object o2,
      @Nullable Object o3,
      @Nullable Object o4,
      @Nullable Object o5,
      @Nullable Object o6,
      @Nullable Object o7) {
    int h1 = hashObject(o1);
    int h2 = hashObject(o2);
    int h3 = hashObject(o3);
    int h4 = hashObject(o4);
    int h5 = hashObject(o5);
    int h6 = hashObject(o6);
    int h7 = hashObject(o7);
    return h7
        + MULTIPLIER
            * (h6
                + MULTIPLIER
                    * (h5
                        + MULTIPLIER
                            * (h4
                                + MULTIPLIER
                                    * (h3
                                        + MULTIPLIER * (h2 + (MULTIPLIER * (h1 + MULTIPLIER)))))));
  }

  public static int hashObjects(
      @Nullable Object o1,
      @Nullable Object o2,
      @Nullable Object o3,
      @Nullable Object o4,
      @Nullable Object o5,
      @Nullable Object o6,
      @Nullable Object o7,
      @Nullable Object o8) {
    int h1 = hashObject(o1);
    int h2 = hashObject(o2);
    int h3 = hashObject(o3);
    int h4 = hashObject(o4);
    int h5 = hashObject(o5);
    int h6 = hashObject(o6);
    int h7 = hashObject(o7);
    int h8 = hashObject(o8);
    return h8
        + MULTIPLIER
            * (h7
                + MULTIPLIER
                    * (h6
                        + MULTIPLIER
                            * (h5
                                + MULTIPLIER
                                    * (h4
                                        + MULTIPLIER
                                            * (h3
                                                + MULTIPLIER
                                                    * (h2 + (MULTIPLIER * (h1 + MULTIPLIER))))))));
  }
}
