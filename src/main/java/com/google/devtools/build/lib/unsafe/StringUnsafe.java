// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.unsafe;

import com.google.common.base.Preconditions;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.util.Arrays;
import javax.annotation.Nullable;
import sun.misc.Unsafe;

/**
 * Provides direct access to the string implementation used by JDK9.
 *
 * <p>Under JDK9, a string is two fields: <code>byte coder</code>, and <code>byte[] value</code>.
 * The <code>coder</code> field has value 0 if the encoding is LATIN-1, and 2 if the encoding is
 * UTF-16 (the classic JDK8 encoding).
 *
 * <p>The <code>value</code> field contains the actual bytes.
 */
public class StringUnsafe {
  // Fields corresponding to the coder
  public static final byte LATIN1 = 0;
  public static final byte UTF16 = 1;

  private static final StringUnsafe INSTANCE = initInstance();
  private final Unsafe unsafe;
  private final Constructor<String> constructor;
  private final long valueOffset;
  private final long coderOffset;

  public static boolean canUse() {
    return RuntimeVersion.isAtLeast9();
  }

  @Nullable
  public static StringUnsafe getInstance() {
    return Preconditions.checkNotNull(INSTANCE);
  }

  private static StringUnsafe initInstance() {
    if (!canUse()) {
      return null;
    }
    return new StringUnsafe();
  }

  private StringUnsafe() {
    unsafe = UnsafeProvider.getInstance();
    Field valueField;
    Field coderField;
    try {
      this.constructor = String.class.getDeclaredConstructor(byte[].class, byte.class);
      valueField = String.class.getDeclaredField("value");
      coderField = String.class.getDeclaredField("coder");
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException(
          "Bad fields/constructor: "
              + Arrays.toString(String.class.getDeclaredConstructors())
              + ", "
              + Arrays.toString(String.class.getDeclaredFields()),
          e);
    }
    this.constructor.setAccessible(true);
    valueField.setAccessible(true);
    valueOffset = UnsafeProvider.getInstance().objectFieldOffset(valueField);
    coderField.setAccessible(true);
    coderOffset = UnsafeProvider.getInstance().objectFieldOffset(coderField);
  }

  /** Returns the coder used for this string. See {@link #LATIN1} and {@link #UTF16}. */
  public byte getCoder(String obj) {
    return unsafe.getByte(obj, coderOffset);
  }

  /**
   * Returns the internal byte array, encoded according to {@link #getCoder}.
   *
   * <p>Use of this is unsafe. The representation may change from one JDK version to the next.
   * Ensure you do not mutate this byte array in any way.
   */
  public byte[] getByteArray(String obj) {
    return (byte[]) unsafe.getObject(obj, valueOffset);
  }

  /** Constructs a new string from a byte array and coder. */
  public String newInstance(byte[] bytes, byte coder) throws ReflectiveOperationException {
    return constructor.newInstance(bytes, coder);
  }
}
