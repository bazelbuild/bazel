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

package android.testing;

import javadesugar.testing.Cuboid;

/**
 * A simulated Android platform SDK class for desugar testing. The class should not be desugared.
 */
public class CuboidInflater {

  private final Cuboid cuboid;

  public CuboidInflater(Cuboid cuboid) {
    this.cuboid = cuboid;
  }

  public static Cuboid inflateStatic(
      Cuboid cuboid, long widthMultiplier, long lengthMultiplier, long heightMultiplier) {
    return new Cuboid(
        widthMultiplier * cuboid.getWidth(),
        lengthMultiplier * cuboid.getLength(),
        heightMultiplier * cuboid.getHeight());
  }

  public Cuboid inflateInstance(
      long widthMultiplier, long lengthMultiplier, long heightMultiplier) {
    return new Cuboid()
        .setDimensions(
            widthMultiplier * cuboid.getWidth(),
            lengthMultiplier * cuboid.getLength(),
            heightMultiplier * cuboid.getHeight());
  }

  public Cuboid onCombine(long tag, Cuboid other) {
    return new Cuboid(
        tag * (cuboid.getWidth() + other.getWidth()),
        tag * (cuboid.getLength() + other.getLength()),
        tag * (cuboid.getHeight() + other.getHeight()));
  }

  public long getVolume() {
    return cuboid.getVolume();
  }
}
