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

package com.google.devtools.build.android.desugar.typeadapter.javadesugar.testing;

/**
 * Simulated a manually-configured type converter between platform built-in types and
 * desugar-mirrored types.
 */
public abstract class CuboidConverter {

  public static jd$.testing.Cuboid from(javadesugar.testing.Cuboid cuboid) {
    return new jd$.testing.Cuboid()
        .setDimensions(cuboid.getWidth(), cuboid.getLength(), cuboid.getHeight());
  }

  public static javadesugar.testing.Cuboid to(jd$.testing.Cuboid cuboid) {
    return new javadesugar.testing.Cuboid()
        .setDimensions(cuboid.getWidth(), cuboid.getLength(), cuboid.getHeight());
  }

  private CuboidConverter() {}
}
