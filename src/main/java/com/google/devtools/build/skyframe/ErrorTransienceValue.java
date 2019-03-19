// Copyright 2014 The Bazel Authors. All rights reserved.
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
 * A value that represents "error transience", i.e. anything which may have caused an unexpected
 * failure. Is not equal to anything, including itself, in order to force re-evaluation.
 */
public final class ErrorTransienceValue implements SkyValue {
  private static final SkyFunctionName FUNCTION_NAME =
      SkyFunctionName.create(
          "ERROR_TRANSIENCE", ShareabilityOfValue.NEVER, FunctionHermeticity.NONHERMETIC);
  @AutoCodec public static final SkyKey KEY = () -> FUNCTION_NAME;
  @AutoCodec public static final ErrorTransienceValue INSTANCE = new ErrorTransienceValue();

  private ErrorTransienceValue() {}

  @Override
  public boolean dataIsShareable() {
    return false;
  }

  @Override
  public int hashCode() {
    // Not the prettiest, but since we always return false for equals throw exception here to
    // catch any errors related to hash-based collections quickly.
    throw new UnsupportedOperationException();
  }

  @Override
  public boolean equals(Object other) {
    return false;
  }

  @Override
  public String toString() {
    return "ErrorTransienceValue";
  }

  @SuppressWarnings("unused")
  private void writeObject(ObjectOutputStream unused) {
    throw new UnsupportedOperationException("Java serialization not supported");
  }

  @SuppressWarnings("unused")
  private void readObject(ObjectInputStream unused) {
    throw new UnsupportedOperationException("Java serialization not supported");
  }
}
