// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.skyframe;

import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

/**
 * A SkyValue with no attached data. Preferable to a specialized empty value class to minimize
 * bloat.
 */
public final class EmptySkyValue implements SkyValue {
  @AutoCodec public static final EmptySkyValue INSTANCE = new EmptySkyValue();

  private EmptySkyValue() {}

  @Override
  public int hashCode() {
    return 422;
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof EmptySkyValue;
  }

  // SkyValue implements Serializable, however we don't want to support Java serialization.
  @SuppressWarnings("unused")
  private void writeObject(ObjectOutputStream unused) {
    throw new UnsupportedOperationException("Java serialization not supported");
  }

  @SuppressWarnings("unused")
  private void readObject(ObjectInputStream unused) {
    throw new UnsupportedOperationException("Java serialization not supported");
  }
}
