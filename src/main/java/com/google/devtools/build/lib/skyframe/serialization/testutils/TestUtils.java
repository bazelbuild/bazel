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

import com.google.devtools.build.lib.skyframe.serialization.CodecRegisterer;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.strings.StringCodecs;
import com.google.devtools.build.lib.syntax.Environment.GlobalFrame;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

/** Helpers for serialization tests. */
public class TestUtils {

  private TestUtils() {}

  /** Serialize a value to a new byte array. */
  public static <T> byte[] toBytes(SerializationContext context, ObjectCodec<T> codec, T value)
      throws IOException, SerializationException {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(bytes);
    codec.serialize(context, value, codedOut);
    codedOut.flush();
    return bytes.toByteArray();
  }

  /** Deserialize a value from a byte array. */
  public static <T> T fromBytes(DeserializationContext context, ObjectCodec<T> codec, byte[] bytes)
      throws SerializationException, IOException {
    return codec.deserialize(context, CodedInputStream.newInstance(bytes));
  }

  /**
   * Asserts that two {@link GlobalFrame}s have the same structure. Needed because
   * {@link GlobalFrame} doesn't override {@link Object#equals}.
   */
  public static void assertGlobalFramesEqual(GlobalFrame frame1, GlobalFrame frame2) {
    assertThat(frame1.mutability().getAnnotation())
        .isEqualTo(frame2.mutability().getAnnotation());
    assertThat(frame1.getLabel()).isEqualTo(frame2.getLabel());
    assertThat(frame1.getTransitiveBindings())
        .containsExactlyEntriesIn(frame2.getTransitiveBindings()).inOrder();
    if (frame1.getParent() == null || frame2.getParent() == null) {
      assertThat(frame1.getParent()).isNull();
      assertThat(frame2.getParent()).isNull();
    } else {
      assertGlobalFramesEqual(frame1.getParent(), frame2.getParent());
    }
  }

  /**
   * Fake string codec that replaces all input and output string values with the constant "dummy".
   */
  public static class ConstantStringCodec implements ObjectCodec<String> {

    private static final ObjectCodec<String> stringCodec = StringCodecs.simple();

    @Override
    public Class<String> getEncodedClass() {
      return String.class;
    }

    @Override
    public void serialize(SerializationContext context, String value, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      stringCodec.serialize(context, "dummy", codedOut);
    }

    @Override
    public String deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      stringCodec.deserialize(context, codedIn);
      return "dummy";
    }

    /** Disables auto-registration of ConstantStringCodec. */
    private static class ConstantStringCodecRegisterer
        implements CodecRegisterer<ConstantStringCodec> {}
  }
}
