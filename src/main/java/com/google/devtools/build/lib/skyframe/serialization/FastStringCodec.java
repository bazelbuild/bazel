// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.security.AccessController;
import java.security.PrivilegedActionException;
import java.security.PrivilegedExceptionAction;
import sun.misc.Unsafe;

/**
 * Similar to {@link StringCodec}, except with deserialization optimized for ascii data. It can
 * still handle UTF-8, though less efficiently than {@link StringCodec}. Should be used when the
 * majority of the data passing through will be ascii.
 */
public class FastStringCodec implements ObjectCodec<String> {
  public static final FastStringCodec INSTANCE = new FastStringCodec();

  private static final Unsafe theUnsafe;
  private static final long STRING_VALUE_OFFSET;

  private static final String EMPTY_STRING = "";

  static {
    theUnsafe = getUnsafe();
    try {
      // String's 'value' field stores its char[]. If this field changes name or type then the
      // reflective check below will fail. We can reasonably expect our approach to be stable for
      // now, but things are likely to change in java 9, hopefully in a way which obsoletes this
      // optimization.
      Field valueField = String.class.getDeclaredField("value");
      Class<?> valueFieldType = valueField.getType();
      if (!valueFieldType.equals(char[].class)) {
        throw new AssertionError(
            "Expected String's value field to be char[], but was " + valueFieldType);
      }
      STRING_VALUE_OFFSET = theUnsafe.objectFieldOffset(valueField);
    } catch (NoSuchFieldException | SecurityException e) {
      throw new AssertionError("Failed to find String's 'value' offset", e);
    }
  }

  @Override
  public Class<String> getEncodedClass() {
    return String.class;
  }

  @Override
  public void serialize(String string, CodedOutputStream codedOut) throws IOException {
    codedOut.writeStringNoTag(string);
  }

  @Override
  public String deserialize(CodedInputStream codedIn) throws IOException {
    int length = codedIn.readInt32();
    if (length == 0) {
      return EMPTY_STRING;
    }

    char[] maybeDecoded = new char[length];
    for (int i = 0; i < length; i++) {
      // Read one byte at a time to avoid creating a new ByteString/copy of the underlying array.
      byte b = codedIn.readRawByte();
      // Check highest order bit, if it's set we've crossed into extended ascii/utf8.
      if ((b & 0x80) == 0) {
        maybeDecoded[i] = (char) b;
      } else {
        // Fail, we encountered a non-ascii byte. Copy what we have so far plus and then the rest
        // of the data into a buffer and let String's constructor do the UTF-8 decoding work.
        byte[] decodeFrom = new byte[length];
        for (int j = 0; j < i; j++) {
          decodeFrom[j] = (byte) maybeDecoded[j];
        }
        decodeFrom[i] = b;
        for (int j = i + 1; j < length; j++) {
          decodeFrom[j] = codedIn.readRawByte();
        }
        return new String(decodeFrom, StandardCharsets.UTF_8);
      }
    }

    try {
      String result = (String) theUnsafe.allocateInstance(String.class);
      theUnsafe.putObject(result, STRING_VALUE_OFFSET, maybeDecoded);
      return result;
    } catch (Exception e) {
      // This should only catch InstantiationException, but that makes IntelliJ unhappy for
      // some reason; it insists that that exception cannot be thrown from here, even though it
      // is set to JDK 8
      throw new IllegalStateException("Could not create string", e);
    }
  }

  /**
   * Get a reference to {@link sun.misc.Unsafe} or throw an {@link AssertionError} if failing to do
   * so. Failure is highly unlikely, but possible if the underlying VM stores unsafe in an
   * unexpected location.
   */
  private static Unsafe getUnsafe() {
    try {
      // sun.misc.Unsafe is intentionally difficult to get a hold of - it gives us the power to
      // do things like access raw memory and segfault the JVM.
      return AccessController.doPrivileged(
          new PrivilegedExceptionAction<Unsafe>() {
            @Override
            public Unsafe run() throws Exception {
              Class<Unsafe> unsafeClass = Unsafe.class;
              // Unsafe usually exists in the field 'theUnsafe', however check all fields
              // in case it's somewhere else in this VM's version of Unsafe.
              for (Field f : unsafeClass.getDeclaredFields()) {
                f.setAccessible(true);
                Object fieldValue = f.get(null);
                if (unsafeClass.isInstance(fieldValue)) {
                  return unsafeClass.cast(fieldValue);
                }
              }
              throw new AssertionError("Failed to find sun.misc.Unsafe instance");
            }
          });
    } catch (PrivilegedActionException pae) {
      throw new AssertionError("Unable to get sun.misc.Unsafe", pae);
    }
  }
}
