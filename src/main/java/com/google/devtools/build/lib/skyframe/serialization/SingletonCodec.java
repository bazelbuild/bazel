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

import com.google.common.base.Preconditions;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

/**
 * Specialized {@link ObjectCodec} for storing singleton values. Values serialize to a supplied
 * representation, which is useful for debugging and is used to verify the serialized representation
 * during deserialization.
 */
public class SingletonCodec<T> implements ObjectCodec<T> {

  /**
   * Create instance wrapping the singleton {@code value}. Will serialize to the byte array
   * representation of {@code mnemonic}. On deserialization if {@code mnemonic} matches the
   * serialized data then {@code value} is returned.
   */
  public static <T> SingletonCodec<T> of(T value, String mnemonic) {
    return new SingletonCodec<T>(value, mnemonic);
  }

  private final T value;
  private final byte[] mnemonic;

  private SingletonCodec(T value, String mnemonic) {
    this.value = Preconditions.checkNotNull(value, "SingletonCodec cannot represent null");
    this.mnemonic = mnemonic.getBytes(StandardCharsets.UTF_8);
  }

  @SuppressWarnings("unchecked")
  @Override
  public Class<T> getEncodedClass() {
    return (Class<T>) value.getClass();
  }

  @Override
  public void serialize(SerializationContext context, T t, CodedOutputStream codedOut)
      throws IOException {
    codedOut.writeByteArrayNoTag(mnemonic);
  }

  @Override
  public T deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    // Get ByteBuffer instead of raw bytes, as it may be a direct view of the data and not a copy,
    // which is much more efficient.
    ByteBuffer readMnemonic = codedIn.readByteBuffer();
    if (!bytesEqual(mnemonic, readMnemonic)) {
      throw new SerializationException(
          "Failed to decode singleton " + value + " expected " + Arrays.toString(mnemonic));
    }
    return value;
  }

  private static boolean bytesEqual(byte[] expected, ByteBuffer buffer) {
    if (buffer.remaining() != expected.length) {
      return false;
    }

    for (int i = 0; i < expected.length; i++) {
      if (expected[i] != buffer.get(i)) {
        return false;
      }
    }

    return true;
  }
}

