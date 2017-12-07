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
import static org.junit.Assert.fail;
import static org.mockito.Matchers.any;
import static org.mockito.Matchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;

import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.NotSerializableException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentCaptor;
import org.mockito.invocation.InvocationOnMock;
import org.mockito.stubbing.Answer;

/** Tests for {@link ObjectCodecs}. */
@RunWith(JUnit4.class)
public class ObjectCodecsTest {

  /** Dummy ObjectCodec implementation so we can verify nice type system interaction. */
  private static class IntegerCodec implements ObjectCodec<Integer> {
    @Override
    public Class<Integer> getEncodedClass() {
      return Integer.class;
    }

    @Override
    public void serialize(Integer obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      codedOut.writeInt32NoTag(obj);
    }

    @Override
    public Integer deserialize(CodedInputStream codedIn)
        throws SerializationException, IOException {
      return codedIn.readInt32();
    }
  }

  private static final String KNOWN_CLASSIFIER = "KNOWN_CLASSIFIER";
  private static final ByteString KNOWN_CLASSIFIER_BYTES =
      ByteString.copyFromUtf8("KNOWN_CLASSIFIER");

  private static final String UNKNOWN_CLASSIFIER = "UNKNOWN_CLASSIFIER";
  private static final ByteString UNKNOWN_CLASSIFIER_BYTES =
      ByteString.copyFromUtf8("UNKNOWN_CLASSIFIER");

  private ObjectCodec<Integer> spyObjectCodec;

  private ObjectCodecs underTest;

  @Before
  public final void setup() {
    spyObjectCodec = spy(new IntegerCodec());
    this.underTest = ObjectCodecs.newBuilder().add(KNOWN_CLASSIFIER, spyObjectCodec).build();
  }

  @Test
  public void testSerializeDeserializeUsesCustomLogicWhenAvailable() throws Exception {
    Integer original = Integer.valueOf(12345);

    doAnswer(
            new Answer<Void>() {
              @Override
              public Void answer(InvocationOnMock invocation) throws IOException {
                CodedOutputStream codedOutArg = (CodedOutputStream) invocation.getArguments()[1];
                codedOutArg.writeInt32NoTag(42);
                return null;
              }
            })
        .when(spyObjectCodec)
        .serialize(eq(original), any(CodedOutputStream.class));
    ArgumentCaptor<CodedInputStream> captor = ArgumentCaptor.forClass(CodedInputStream.class);
    doReturn(original).when(spyObjectCodec).deserialize(captor.capture());

    ByteString serialized = underTest.serialize(KNOWN_CLASSIFIER, original);
    Object deserialized = underTest.deserialize(KNOWN_CLASSIFIER_BYTES, serialized);
    assertThat(deserialized).isEqualTo(original);

    assertThat(captor.getValue().readInt32()).isEqualTo(42);
  }

  @Test
  public void testMismatchedArgRaisesException() throws Exception {
    Long notRight = Long.valueOf(123456789);
    try {
      underTest.serialize(KNOWN_CLASSIFIER, notRight);
      fail("Expected exception");
    } catch (SerializationException expected) {
    }
  }

  @Test
  public void testSerializeDeserializeWhenNocustomLogicAvailable() throws Exception {
    Integer original = Integer.valueOf(12345);

    ByteString serialized = underTest.serialize(UNKNOWN_CLASSIFIER, original);
    Object deserialized = underTest.deserialize(UNKNOWN_CLASSIFIER_BYTES, serialized);
    assertThat(deserialized).isEqualTo(original);

    verify(spyObjectCodec, never()).serialize(any(Integer.class), any(CodedOutputStream.class));
    verify(spyObjectCodec, never()).deserialize(any(CodedInputStream.class));
  }

  @Test
  public void testSerializePropagatesSerializationExceptionFromCustomCodec() throws Exception {
    Integer original = Integer.valueOf(12345);

    SerializationException staged = new SerializationException("BECAUSE FAIL");
    doThrow(staged).when(spyObjectCodec).serialize(eq(original), any(CodedOutputStream.class));
    try {
      underTest.serialize(KNOWN_CLASSIFIER, original);
      fail("Expected exception");
    } catch (SerializationException e) {
      assertThat(e).isSameAs(staged);
    }
  }

  @Test
  public void testSerializePropagatesIOExceptionFromCustomCodecsAsSerializationException()
      throws Exception {
    Integer original = Integer.valueOf(12345);

    IOException staged = new IOException("BECAUSE FAIL");
    doThrow(staged).when(spyObjectCodec).serialize(eq(original), any(CodedOutputStream.class));
    try {
      underTest.serialize(KNOWN_CLASSIFIER, original);
      fail("Expected exception");
    } catch (SerializationException e) {
      assertThat(e).hasCauseThat().isSameAs(staged);
    }
  }

  @Test
  public void testDeserializePropagatesSerializationExceptionFromCustomCodec() throws Exception {
    SerializationException staged = new SerializationException("BECAUSE FAIL");
    doThrow(staged).when(spyObjectCodec).deserialize(any(CodedInputStream.class));
    try {
      underTest.deserialize(KNOWN_CLASSIFIER_BYTES, ByteString.EMPTY);
      fail("Expected exception");
    } catch (SerializationException e) {
      assertThat(e).isSameAs(staged);
    }
  }

  @Test
  public void testDeserializePropagatesIOExceptionFromCustomCodecAsSerializationException()
      throws Exception {
    IOException staged = new IOException("BECAUSE FAIL");
    doThrow(staged).when(spyObjectCodec).deserialize(any(CodedInputStream.class));
    try {
      underTest.deserialize(KNOWN_CLASSIFIER_BYTES, ByteString.EMPTY);
      fail("Expected exception");
    } catch (SerializationException e) {
      assertThat(e).hasCauseThat().isSameAs(staged);
    }
  }

  @Test
  public void testSerializePropagatesSerializationExceptionFromDefaultCodec() throws Exception {
    Object nonSerializable = new Object();

    try {
      underTest.serialize(UNKNOWN_CLASSIFIER, nonSerializable);
      fail("Expected exception");
    } catch (SerializationException e) {
      assertThat(e).hasCauseThat().isInstanceOf(NotSerializableException.class);
    }
  }

  @Test
  public void testDeserializePropagatesSerializationExceptionFromDefaultCodec() throws Exception {
    ByteString serialized = ByteString.copyFromUtf8("probably not serialized anything");

    try {
      underTest.deserialize(UNKNOWN_CLASSIFIER_BYTES, serialized);
      fail("Expected exception");
    } catch (SerializationException expected) {
    }
  }

  @Test
  public void testSerializeFailsWhenNoCustomCodecAndFallbackDisabled() throws Exception {
    try {
      ObjectCodecs.newBuilder().setAllowDefaultCodec(false).build().serialize("X", "Y");
      fail("Expected exception");
    } catch (SerializationException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo("No codec available for X and default fallback disabled");
    }
  }

  @Test
  public void testDeserializeFailsWhenNoCustomCodecAndFallbackDisabled() throws Exception {
    ByteString serialized = ByteString.copyFromUtf8("doesn't matter");
    try {
      ObjectCodecs.newBuilder()
          .setAllowDefaultCodec(false)
          .build()
          .deserialize(ByteString.copyFromUtf8("X"), serialized);
      fail("Expected exception");
    } catch (SerializationException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo("No codec available for X and default fallback disabled");
    }
  }

  @Test
  public void testSerializeDeserializeInBulk() throws Exception {
    Integer value1 = 12345;
    Integer value2 = 67890;
    Integer value3 = 42;

    ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(bytesOut);
    underTest.serialize(KNOWN_CLASSIFIER, value1, codedOut);
    underTest.serialize(KNOWN_CLASSIFIER, value2, codedOut);
    underTest.serialize(KNOWN_CLASSIFIER, value3, codedOut);
    codedOut.flush();

    CodedInputStream codedIn = CodedInputStream.newInstance(bytesOut.toByteArray());
    assertThat(underTest.deserialize(KNOWN_CLASSIFIER_BYTES, codedIn)).isEqualTo(value1);
    assertThat(underTest.deserialize(KNOWN_CLASSIFIER_BYTES, codedIn)).isEqualTo(value2);
    assertThat(underTest.deserialize(KNOWN_CLASSIFIER_BYTES, codedIn)).isEqualTo(value3);
  }
}
