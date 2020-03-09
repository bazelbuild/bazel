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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.spy;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ObjectCodecs}. */
@RunWith(JUnit4.class)
public class ObjectCodecsTest {

  /** Dummy ObjectCodec implementation so we can verify nice type system interaction. */
  private static class IntegerCodec implements ObjectCodec<Integer> {
    @Override
    public Class<Integer> getEncodedClass() {
      return Integer.class;
    }

    // We have to override the default explicitly here because Mockito can't delegate to default
    // methods on interfaces.
    @Override
    public List<Class<? extends Integer>> additionalEncodedClasses() {
      return ImmutableList.of();
    }

    @Override
    public void serialize(SerializationContext context, Integer obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      codedOut.writeInt32NoTag(obj);
    }

    @Override
    public Integer deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      return codedIn.readInt32();
    }

    /** Disables auto-registration. */
    private static class IntegerCodecRegisterer implements CodecRegisterer<IntegerCodec> {}
  }

  private ObjectCodec<Integer> spyObjectCodec;

  private ObjectCodecs underTest;

  @Before
  public final void setup() {
    spyObjectCodec = spy(new IntegerCodec());
    this.underTest =
        new ObjectCodecs(
            ObjectCodecRegistry.newBuilder().add(spyObjectCodec).build(), ImmutableMap.of());
  }

  @Test
  public void testSerializeDeserializeUsesCustomLogicWhenAvailable() throws Exception {
    Integer original = 12345;

    doAnswer(
            invocation -> {
              CodedOutputStream codedOutArg = (CodedOutputStream) invocation.getArguments()[2];
              codedOutArg.writeInt32NoTag(42);
              return null;
            })
        .when(spyObjectCodec)
        .serialize(any(SerializationContext.class), eq(original), any(CodedOutputStream.class));
    AtomicInteger readInteger = new AtomicInteger(0);
    doAnswer(
            invocation -> {
              readInteger.set(((CodedInputStream) invocation.getArguments()[1]).readInt32());
              return original;
            })
        .when(spyObjectCodec)
        .deserialize(any(DeserializationContext.class), any(CodedInputStream.class));

    ByteString serialized = underTest.serialize(original);
    Object deserialized = underTest.deserialize(serialized);
    assertThat(deserialized).isEqualTo(original);

    assertThat(readInteger.get()).isEqualTo(42);
  }

  @Test
  public void tooManyBytesCausesFailure() throws Exception {
    doReturn(1)
        .when(spyObjectCodec)
        .deserialize(any(DeserializationContext.class), any(CodedInputStream.class));
    doAnswer(
            invocation -> {
              ((CodedOutputStream) invocation.getArguments()[2]).writeInt64NoTag(0xAAAAAA);
              return null;
            })
        .when(spyObjectCodec)
        .serialize(any(SerializationContext.class), eq(1), any(CodedOutputStream.class));
    SerializationException e =
        assertThrows(
            SerializationException.class, () -> underTest.deserialize(underTest.serialize(1)));
    assertThat(e).hasMessageThat().isEqualTo("input stream not exhausted after deserializing 1");
  }

  @Test
  public void testSerializePropagatesSerializationExceptionFromCustomCodec() throws Exception {
    Integer original = Integer.valueOf(12345);

    SerializationException staged = new SerializationException("BECAUSE FAIL");
    doThrow(staged)
        .when(spyObjectCodec)
        .serialize(any(SerializationContext.class), eq(original), any(CodedOutputStream.class));
    SerializationException e =
        assertThrows(SerializationException.class, () -> underTest.serialize(original));
    assertThat(e).isSameInstanceAs(staged);
  }

  @Test
  public void testSerializePropagatesIOExceptionFromCustomCodecsAsSerializationException()
      throws Exception {
    Integer original = Integer.valueOf(12345);

    IOException staged = new IOException("BECAUSE FAIL");
    doThrow(staged)
        .when(spyObjectCodec)
        .serialize(any(SerializationContext.class), eq(original), any(CodedOutputStream.class));
    SerializationException e =
        assertThrows(SerializationException.class, () -> underTest.serialize(original));
    assertThat(e).hasCauseThat().isSameInstanceAs(staged);
  }

  @Test
  public void testDeserializePropagatesSerializationExceptionFromCustomCodec() throws Exception {
    SerializationException staged = new SerializationException("BECAUSE FAIL");
    doThrow(staged)
        .when(spyObjectCodec)
        .deserialize(any(DeserializationContext.class), any(CodedInputStream.class));
    SerializationException thrown =
        assertThrows(
            SerializationException.class, () -> underTest.deserialize(underTest.serialize(1)));
    assertThat(thrown).isSameInstanceAs(staged);
  }

  @Test
  public void testDeserializePropagatesIOExceptionFromCustomCodecAsSerializationException()
      throws Exception {
    IOException staged = new IOException("BECAUSE FAIL");
    doThrow(staged)
        .when(spyObjectCodec)
        .deserialize(any(DeserializationContext.class), any(CodedInputStream.class));
    SerializationException e =
        assertThrows(
            SerializationException.class, () -> underTest.deserialize(underTest.serialize(1)));
    assertThat(e).hasCauseThat().isSameInstanceAs(staged);
  }

  @Test
  public void testDeserializePropagatesSerializationExceptionFromDefaultCodec() {
    ByteString serialized = ByteString.copyFromUtf8("probably not serialized anything");

    assertThrows(SerializationException.class, () -> underTest.deserialize(serialized));
  }

  @Test
  public void testSerializeFailsWhenNoCustomCodecAndFallbackDisabled() {
    ObjectCodecs underTest =
        new ObjectCodecs(
            ObjectCodecRegistry.newBuilder().setAllowDefaultCodec(false).build(),
            ImmutableMap.of());
    SerializationException.NoCodecException expected =
        assertThrows(SerializationException.NoCodecException.class, () -> underTest.serialize("Y"));
    assertThat(expected)
        .hasMessageThat()
        .isEqualTo("No codec available for class java.lang.String and default fallback disabled");
  }

  @Test
  public void testDeserializeFailsWithNoCodecs() throws Exception {
    ByteString serialized = underTest.serialize(1);
    ObjectCodecs underTest =
        new ObjectCodecs(
            ObjectCodecRegistry.newBuilder().setAllowDefaultCodec(false).build(),
            ImmutableMap.of());
    assertThrows(
        SerializationException.NoCodecException.class, () -> underTest.deserialize(serialized));
  }

  @Test
  public void testSerializeDeserialize() throws Exception {
    ObjectCodecs underTest = new ObjectCodecs(AutoRegistry.get(), ImmutableMap.of());
    assertThat((String) underTest.deserialize(underTest.serialize("hello"))).isEqualTo("hello");
    assertThat(underTest.deserialize(underTest.serialize(null))).isNull();
  }

  private static class MyException extends Exception {}

  @Test
  public void exception() throws SerializationException {
    MyException exception = new MyException();
    // Force initialization of stack trace.
    StackTraceElement[] stackTrace = exception.getStackTrace();
    ObjectCodecs underTest = new ObjectCodecs(AutoRegistry.get(), ImmutableMap.of());
    assertThat(
            ((MyException) underTest.deserializeMemoized(underTest.serializeMemoized(exception)))
                .getStackTrace())
        .isEqualTo(stackTrace);
  }
}
