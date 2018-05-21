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

package com.google.devtools.build.lib.skyframe.serialization;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.UnsafeProvider;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.util.Arrays;

/**
 * A high-performance {@link ObjectCodec} for {@link String} objects specialized for Strings in
 * JDK9, where a String can be represented as a byte array together with a single byte (0 or 1) for
 * Latin-1 or UTF16 encoding.
 */
public class UnsafeJdk9StringCodec implements ObjectCodec<String> {
  @VisibleForTesting
  public static boolean canUseUnsafeCodec() {
    return getVersion() > 1.8;
  }

  private static double getVersion() {
    String version = System.getProperty("java.version");
    int pos = version.indexOf('.');
    pos = version.indexOf('.', pos + 1);
    return Double.parseDouble(version.substring(0, pos));
  }

  private final Constructor<String> constructor;
  private final long valueOffset;
  private final long coderOffset;

  public UnsafeJdk9StringCodec() {
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

  @Override
  public Class<? extends String> getEncodedClass() {
    return String.class;
  }

  @Override
  public MemoizationStrategy getStrategy() {
    // Don't memoize strings inside memoizing serialization, to preserve current behavior.
    // TODO(janakr,brandjon,michajlo): Is it actually a problem to memoize strings? Doubt there
    // would be much performance impact from increasing the size of the identity map, and we
    // could potentially drop our string tables in the future.
    return MemoizationStrategy.DO_NOT_MEMOIZE;
  }

  @Override
  public void serialize(SerializationContext context, String obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    byte coder = UnsafeProvider.getInstance().getByte(obj, coderOffset);
    byte[] value = (byte[]) UnsafeProvider.getInstance().getObject(obj, valueOffset);
    // Optimize for the case that coder == 0, in which case we can just write the length here,
    // potentially using just one byte. If coder != 0, we'll use 4 bytes, but that's vanishingly
    // rare.
    if (coder == 0) {
      codedOut.writeInt32NoTag(value.length);
    } else if (coder == 1) {
      codedOut.writeInt32NoTag(-value.length);
    } else {
      throw new SerializationException("Unexpected coder value: " + coder + " for " + obj);
    }
    codedOut.writeRawBytes(value);
  }

  @Override
  public String deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int length = codedIn.readInt32();
    byte coder;
    if (length >= 0) {
      coder = 0;
    } else {
      coder = 1;
      length = -length;
    }
    byte[] value = codedIn.readRawBytes(length);
    try {
      return constructor.newInstance(value, coder);
    } catch (ReflectiveOperationException e) {
      throw new SerializationException(
          "Could not instantiate string: " + Arrays.toString(value) + ", " + coder, e);
    }
  }
}
