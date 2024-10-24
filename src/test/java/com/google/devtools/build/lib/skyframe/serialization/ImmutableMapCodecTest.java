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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException.NoCodecException;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.skyframe.serialization.testutils.RoundTripping;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester.VerificationFunction;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.util.Comparator;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ImmutableMapCodec}. */
@RunWith(JUnit4.class)
public class ImmutableMapCodecTest {

  @SuppressWarnings("unused")
  @SerializationConstant
  @VisibleForSerialization
  static final Comparator<?> ORDERING_REVERSE_NATURAL = Ordering.natural().reverse();

  @SerializationConstant @VisibleForSerialization
  static final Comparator<String> HELLO_FIRST_COMPARATOR = selectedFirstComparator("hello");

  @Test
  public void smoke() throws Exception {
    new SerializationTester(
            ImmutableMap.of(),
            ImmutableMap.of("A", "//foo:A"),
            ImmutableMap.of("B", "//foo:B"),
            ImmutableSortedMap.of(),
            ImmutableSortedMap.of("A", "//foo:A"),
            ImmutableSortedMap.of("B", "//foo:B"),
            ImmutableSortedMap.reverseOrder().put("a", "b").put("c", "d").buildOrThrow())
        // Check for order.
        .setVerificationFunction(
            (VerificationFunction<ImmutableMap<?, ?>>)
                (deserialized, subject) -> {
                  assertThat(deserialized).isEqualTo(subject);
                  assertThat(deserialized).containsExactlyEntriesIn(subject).inOrder();
                })
        .runTests();
  }

  @Test
  public void immutableSortedMapRoundTripsWithTheSameComparator() throws Exception {
    ImmutableSortedMap<?, ?> deserialized =
        RoundTripping.roundTrip(
            ImmutableSortedMap.orderedBy(HELLO_FIRST_COMPARATOR)
                .put("a", "b")
                .put("hello", "there")
                .buildOrThrow());

    assertThat(deserialized).containsExactly("hello", "there", "a", "b");
    assertThat(deserialized.comparator()).isSameInstanceAs(HELLO_FIRST_COMPARATOR);
  }

  @Test
  public void immutableSortedMapUnserializableComparatorFails() {
    Comparator<String> comparator = selectedFirstComparator("c");

    NoCodecException thrown =
        assertThrows(
            NoCodecException.class,
            () ->
                RoundTripping.roundTrip(
                    ImmutableSortedMap.<String, String>orderedBy(comparator)
                        .put("a", "b")
                        .put("c", "d")
                        .buildOrThrow()));
    assertThat(thrown)
        .hasMessageThat()
        .startsWith("No default codec available for " + comparator.getClass().getName());
  }

  @Test
  public void serializingErrorIncludesKeyStringAndValueClass() {
    SerializationException expected =
        assertThrows(
            SerializationException.class,
            () ->
                RoundTripping.toBytesMemoized(
                    ImmutableMap.of("a", new Dummy()),
                    AutoRegistry.get()
                        .getBuilder()
                        .add(new DummyThrowingCodec(/* throwsOnSerialization= */ true))
                        .build()));
    assertThat(expected)
        .hasMessageThat()
        .containsMatch("Exception while serializing value of type .*\\$Dummy for key 'a'");
  }

  @Test
  public void deserializingErrorIncludesKeyString() throws Exception {
    ObjectCodecRegistry registry =
        AutoRegistry.get()
            .getBuilder()
            .add(new DummyThrowingCodec(/*throwsOnSerialization=*/ false))
            .build();
    ObjectCodecs codecs = new ObjectCodecs(registry);
    ByteString data = codecs.serialize(ImmutableMap.of("a", new Dummy()));
    SerializationException expected =
        assertThrows(SerializationException.class, () -> codecs.deserialize(data));
    assertThat(expected)
        .hasMessageThat()
        .contains("Exception while deserializing value for key 'a'");
  }

  private static Comparator<String> selectedFirstComparator(String first) {
    return (a, b) -> {
      if (a.equals(b)) {
        return 0;
      }
      if (a.equals(first)) {
        return -1;
      }
      if (b.equals(first)) {
        return 1;
      }
      return a.compareTo(b);
    };
  }

  private static class Dummy {}

  private static class DummyThrowingCodec implements ObjectCodec<Dummy> {
    private final boolean throwsOnSerialization;

    private DummyThrowingCodec(boolean throwsOnSerialization) {
      this.throwsOnSerialization = throwsOnSerialization;
    }

    @Override
    public Class<Dummy> getEncodedClass() {
      return Dummy.class;
    }

    @Override
    public void serialize(SerializationContext context, Dummy value, CodedOutputStream codedOut)
        throws SerializationException {
      if (throwsOnSerialization) {
        throw new SerializationException("Expected failure");
      }
    }

    @Override
    public Dummy deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException {
      Preconditions.checkState(!throwsOnSerialization);
      throw new SerializationException("Expected failure");
    }
  }
}
