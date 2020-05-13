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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skyframe.serialization.AutoRegistry;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecRegistry;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.syntax.Module;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import javax.annotation.Nullable;

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

  public static <T> ByteString toBytes(T value, ImmutableMap<Class<?>, Object> dependencies)
      throws IOException, SerializationException {
    return toBytes(new SerializationContext(dependencies), value);
  }

  public static <T> ByteString toBytes(SerializationContext serializationContext, T value)
      throws IOException, SerializationException {
    ByteString.Output output = ByteString.newOutput();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(output);
    serializationContext.serialize(value, codedOut);
    codedOut.flush();
    return output.toByteString();
  }

  public static Object fromBytes(DeserializationContext deserializationContext, ByteString bytes)
      throws IOException, SerializationException {
    return deserializationContext.deserialize(bytes.newCodedInput());
  }

  /** Deserialize a value from a byte array. */
  public static <T> T fromBytes(DeserializationContext context, ObjectCodec<T> codec, byte[] bytes)
      throws SerializationException, IOException {
    return codec.deserialize(context, CodedInputStream.newInstance(bytes));
  }

  public static <T> T roundTrip(T value, ObjectCodecRegistry registry)
      throws IOException, SerializationException {
    return new DeserializationContext(registry, ImmutableMap.of())
        .deserialize(
            toBytes(new SerializationContext(registry, ImmutableMap.of()), value).newCodedInput());
  }

  public static <T> T roundTrip(T value, ImmutableMap<Class<?>, Object> dependencies)
      throws IOException, SerializationException {
    ObjectCodecRegistry.Builder builder = AutoRegistry.get().getBuilder();
    for (Object constant : dependencies.values()) {
      builder.addReferenceConstant(constant);
    }
    ObjectCodecRegistry registry = builder.build();
    return new DeserializationContext(registry, dependencies)
        .deserialize(
            toBytes(new SerializationContext(registry, dependencies), value).newCodedInput());
  }

  public static <T> T roundTrip(T value) throws IOException, SerializationException {
    return TestUtils.roundTrip(value, ImmutableMap.of());
  }

  public static void assertFramesEqual(Module frame1, Module frame2) {
    assertThat(frame1.getTransitiveBindings())
        .containsExactlyEntriesIn(frame2.getTransitiveBindings())
        .inOrder();
  }

  /**
   * Asserts that two {@link Module}s have the same structure. Needed because {@link Module} doesn't
   * override {@link Object#equals}.
   */
  public static void assertModulesEqual(Module frame1, Module frame2) {
    assertThat(frame1.mutability().getAnnotation())
        .isEqualTo(frame2.mutability().getAnnotation());
    assertThat(frame1.getClientData()).isEqualTo(frame2.getClientData());
    assertThat(frame1.getTransitiveBindings())
        .containsExactlyEntriesIn(frame2.getTransitiveBindings()).inOrder();
    if (frame1.getParent() == null || frame2.getParent() == null) {
      assertThat(frame1.getParent()).isNull();
      assertThat(frame2.getParent()).isNull();
    } else {
      assertFramesEqual(frame1.getParent(), frame2.getParent());
    }
    assertThat(frame1.getExportedBindings())
        .containsExactlyEntriesIn(frame2.getExportedBindings())
        .inOrder();
  }

  public static ByteString toBytesMemoized(Object original, ObjectCodecRegistry registry)
      throws IOException, SerializationException {
    ByteString.Output output = ByteString.newOutput();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(output);
    new ObjectCodecs(registry).serializeMemoized(original, codedOut);
    codedOut.flush();
    return output.toByteString();
  }

  public static Object fromBytesMemoized(ByteString bytes, ObjectCodecRegistry registry)
      throws IOException, SerializationException {
    return new ObjectCodecs(registry).deserializeMemoized(bytes.newCodedInput());
  }

  @SuppressWarnings("unchecked")
  public static <T> T roundTripMemoized(T original, ObjectCodecRegistry registry)
      throws IOException, SerializationException {
    return (T)
        new ObjectCodecs(registry)
            .deserializeMemoized(toBytesMemoized(original, registry).newCodedInput());
  }

  public static <T> T roundTripMemoized(T original, ObjectCodec<?>... codecs)
      throws IOException, SerializationException {
    return roundTripMemoized(original, getBuilderWithAdditionalCodecs(codecs).build());
  }

  public static <T> T roundTripMemoized(
      T original, @Nullable Mutability mutability, ObjectCodec<?>... codecs)
      throws IOException, SerializationException {
    return roundTripMemoized(original, getBuilderWithAdditionalCodecs(codecs).build());
  }

  public static ObjectCodecRegistry.Builder getBuilderWithAdditionalCodecs(
      ObjectCodec<?>... codecs) {
    ObjectCodecRegistry.Builder builder = AutoRegistry.get().getBuilder();
    for (ObjectCodec<?> codec : codecs) {
      builder.add(codec);
    }
    return builder;
  }
}
