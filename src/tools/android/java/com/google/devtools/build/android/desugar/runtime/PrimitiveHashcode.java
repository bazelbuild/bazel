/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package com.google.devtools.build.android.desugar.runtime;

/** Backward-compatibility implementation for the hash code of primitive types. */
public class PrimitiveHashcode {

  public static int hashCode(float value) {
    return Float.floatToIntBits(value);
  }

  public static int hashCode(boolean value) {
    return value ? 1231 : 1237;
  }

  public static int hashCode(long value) {
    return (int) (value ^ value >>> 32);
  }

  public static int hashCode(double value) {
    long bits = Double.doubleToLongBits(value);
    return (int) (bits ^ (bits >>> 32));
  }

  public static int identityAsHashCode(int value) {
    return value;
  }

  public static int identityAsHashCode(short value) {
    return value;
  }

  public static int identityAsHashCode(byte value) {
    return value;
  }

  public static int identityAsHashCode(char value) {
    return value;
  }

  private PrimitiveHashcode() {}
}
