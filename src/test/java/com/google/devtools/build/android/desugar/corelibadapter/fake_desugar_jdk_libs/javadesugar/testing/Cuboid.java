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

/** A simulated built-in JDK class for desugar testing. */
public final class Cuboid {

  private long width;
  private long length;
  private long height;

  public Cuboid(long width, long length, long height) {
    this.width = width;
    this.length = length;
    this.height = height;
  }

  public Cuboid() {
    this(0, 0, 0);
  }

  public Cuboid(Cuboid cuboid) {
    this(cuboid.width, cuboid.length, cuboid.height);
  }

  public Cuboid setDimensions(long width, long length, long height) {
    this.width = width;
    this.length = length;
    this.height = height;
    return this;
  }

  public long getWidth() {
    return width;
  }

  public long getLength() {
    return length;
  }

  public long getHeight() {
    return height;
  }

  public long getVolume() {
    return width * length * height;
  }
}
