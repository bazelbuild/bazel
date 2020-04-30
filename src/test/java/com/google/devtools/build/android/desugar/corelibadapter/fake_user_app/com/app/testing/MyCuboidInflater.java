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

/**
 * Simulates an Android SDK-derived class in user application for testing constructor desugaring.
 */
public class MyCuboidInflater extends CuboidInflater {

  public MyCuboidInflater(Object cuboid) {
    // Simulates the invocation of Android SDK constructor API.
    super(cuboid instanceof Cuboid ? (Cuboid) cuboid : new Cuboid(2, 4, 8));
  }

  public MyCuboidInflater(long w, long l, long h) {
    // Simulates the invocation of Android SDK constructor API.
    super(w * l * h < 64 ? new Cuboid(2, 4, 8) : new Cuboid(w, l, h));
  }

  @Override
  public Cuboid onCombine(long tag, Cuboid other) {
    Cuboid cuboid = super.onCombine(tag, other);
    return new Cuboid(tag + cuboid.getWidth(), tag + cuboid.getLength(), tag + cuboid.getHeight());
  }
}
