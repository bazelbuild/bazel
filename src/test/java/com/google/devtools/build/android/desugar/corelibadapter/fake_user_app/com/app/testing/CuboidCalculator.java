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

package com.app.testing;

import android.testing.CuboidInflater;
import javadesugar.testing.Cuboid;

/** Simulates application-level code that's subject to desugar. */
public final class CuboidCalculator {

  public static long invokeConstructor(long width, long length, long height) {
    Cuboid cuboid = new Cuboid(width, length, height);

    // Simulated Android API.
    CuboidInflater inflater = new CuboidInflater(cuboid);

    return inflater.getVolume();
  }

  public static long invokeDerivedClassConstructorWithEmbeddedInstance(
      long width, long length, long height) {
    Cuboid cuboid = new Cuboid(width, length, height);

    // Derived
    MyCuboidInflater inflater = new MyCuboidInflater(cuboid);

    return inflater.getVolume();
  }

  public static long invokeDerivedClassConstructorWithDimensions(
      long width, long length, long height) {

    // Derived
    MyCuboidInflater inflater = new MyCuboidInflater(width, length, height);

    return inflater.getVolume();
  }

  public static long invokeStaticMethod(
      long width,
      long length,
      long height,
      long widthMultiplier,
      long lengthMultiplier,
      long heightMultiplier) {
    Cuboid cuboid = new Cuboid(width, length, height);

    // Simulated Android API.
    cuboid =
        CuboidInflater.inflateStatic(cuboid, widthMultiplier, lengthMultiplier, heightMultiplier);

    return cuboid.getVolume();
  }

  public static long invokeInstanceMethod(
      long width,
      long length,
      long height,
      long widthMultiplier,
      long lengthMultiplier,
      long heightMultiplier) {
    CuboidInflater cuboidInflater = createCuboidInflater(width, length, height);

    // Simulated Android API.
    Cuboid cuboid =
        cuboidInflater.inflateInstance(widthMultiplier, lengthMultiplier, heightMultiplier);

    return cuboid.getVolume();
  }

  public static long invokeOverridableMethod(long width, long length, long height) {
    Cuboid base = new Cuboid(width, length, height);
    Cuboid other = new Cuboid(2 * width, 3 * length, 4 * height);

    // Derived
    CuboidInflater inflater = new MyCuboidInflater(base);
    Cuboid combined = inflater.onCombine(10L, other);

    return combined.getVolume();
  }

  private static CuboidInflater createCuboidInflater(long width, long length, long height) {
    return new CuboidInflater(new Cuboid(width, length, height));
  }

  private CuboidCalculator() {}
}
