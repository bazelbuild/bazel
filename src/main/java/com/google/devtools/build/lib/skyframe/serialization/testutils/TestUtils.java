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

package com.google.devtools.build.lib.skyframe.serialization.testutils;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.syntax.Environment.Frame;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

/** Helpers for serialization tests. */
public class TestUtils {

  private TestUtils() {}

  /** Serialize a value to a new byte array. */
  public static <T> byte[] toBytes(ObjectCodec<T> codec, T value)
      throws IOException, SerializationException {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(bytes);
    codec.serialize(value, codedOut);
    codedOut.flush();
    return bytes.toByteArray();
  }

  /** Deserialize a value from a byte array. */
  public static <T> T fromBytes(ObjectCodec<T> codec, byte[] bytes)
      throws SerializationException, IOException {
    return codec.deserialize(CodedInputStream.newInstance(bytes));
  }

  /**
   * Asserts that two {@link Frame}s have the same structure. Needed because {@link Frame} doesn't
   * override {@link Object#equals}.
   */
  public static void assertFramesEqual(Frame frame1, Frame frame2) {
    assertThat(frame1.mutability().getAnnotation())
        .isEqualTo(frame2.mutability().getAnnotation());
    assertThat(frame1.getLabel()).isEqualTo(frame2.getLabel());
    assertThat(frame1.getTransitiveBindings())
        .containsExactlyEntriesIn(frame2.getTransitiveBindings()).inOrder();
    if (frame1.getParent() == null || frame2.getParent() == null) {
      assertThat(frame1.getParent()).isNull();
      assertThat(frame2.getParent()).isNull();
    } else {
      assertFramesEqual(frame1.getParent(), frame2.getParent());
    }
  }
}
