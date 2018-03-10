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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecRegistry.CodecDescriptor;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException.NoCodecException;
import com.google.protobuf.ByteString;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ObjectCodecRegistry}. */
@RunWith(JUnit4.class)
public class ObjectCodecRegistryTest {

  @Test
  public void testDescriptorLookups() throws NoCodecException {
    SingletonCodec<String> codec1 = SingletonCodec.of("value1", "mnemonic1");
    SingletonCodec<String> codec2 = SingletonCodec.of("value2", "mnemonic2");

    ObjectCodecRegistry underTest = ObjectCodecRegistry.newBuilder()
        .setAllowDefaultCodec(false)
        .add("foo", codec1)
        .add("bar", codec2)
        .build();

    CodecDescriptor fooDescriptor = underTest.getCodecDescriptor("foo");
    assertThat(fooDescriptor.getCodec()).isSameAs(codec1);
    assertThat(underTest.getCodecDescriptor(ByteString.copyFromUtf8("foo")))
        .isSameAs(fooDescriptor);
    assertThat(underTest.getCodecDescriptorByTag(fooDescriptor.getTag())).isSameAs(fooDescriptor);

    CodecDescriptor barDescriptor = underTest.getCodecDescriptor("bar");
    assertThat(barDescriptor.getCodec()).isSameAs(codec2);
    assertThat(underTest.getCodecDescriptor(ByteString.copyFromUtf8("bar")))
        .isSameAs(barDescriptor);
    assertThat(underTest.getCodecDescriptorByTag(barDescriptor.getTag())).isSameAs(barDescriptor);

    assertThat(barDescriptor.getTag()).isNotEqualTo(fooDescriptor.getTag());

    assertThrows(NoCodecException.class, () -> underTest.getCodecDescriptor("baz"));
    assertThrows(
        NoCodecException.class,
        () -> underTest.getCodecDescriptor(ByteString.copyFromUtf8("baz")));
    assertThrows(NoCodecException.class, () -> underTest.getCodecDescriptorByTag(42));
  }

  @Test
  public void testDefaultCodecFallback() throws NoCodecException {
    SingletonCodec<String> codec = SingletonCodec.of("value1", "mnemonic1");

    ObjectCodecRegistry underTest = ObjectCodecRegistry.newBuilder()
        .setAllowDefaultCodec(true)
        .add("foo", codec)
        .build();

    CodecDescriptor fooDescriptor = underTest.getCodecDescriptor("foo");
    assertThat(fooDescriptor.getCodec()).isSameAs(codec);

    CodecDescriptor barDefaultDescriptor = underTest.getCodecDescriptor("bar");
    assertThat(barDefaultDescriptor.getCodec()).isNotSameAs(codec);
    assertThat(barDefaultDescriptor.getTag()).isNotEqualTo(fooDescriptor.getTag());
    assertThat(underTest.getCodecDescriptorByTag(barDefaultDescriptor.getTag()))
        .isSameAs(barDefaultDescriptor);

    assertThat(underTest.getCodecDescriptor("baz")).isSameAs(barDefaultDescriptor);

    // Bogus tags still throw.
    assertThrows(NoCodecException.class, () -> underTest.getCodecDescriptorByTag(42));
  }

  @Test
  public void testStableTagOrdering() throws NoCodecException {
    SingletonCodec<String> codec1 = SingletonCodec.of("value1", "mnemonic1");
    SingletonCodec<String> codec2 = SingletonCodec.of("value2", "mnemonic2");

    ObjectCodecRegistry underTest1 = ObjectCodecRegistry.newBuilder()
        .setAllowDefaultCodec(true)
        .add("foo", codec1)
        .add("bar", codec2)
        .build();

    ObjectCodecRegistry underTest2 = ObjectCodecRegistry.newBuilder()
        .setAllowDefaultCodec(true)
        .add("bar", codec2)
        .add("foo", codec1)
        .build();

    assertThat(underTest1.getCodecDescriptor("foo").getTag())
        .isEqualTo(underTest2.getCodecDescriptor("foo").getTag());
    assertThat(underTest1.getCodecDescriptor("bar").getTag())
        .isEqualTo(underTest2.getCodecDescriptor("bar").getTag());
    // Default codec.
    assertThat(underTest1.getCodecDescriptor("baz").getTag())
        .isEqualTo(underTest2.getCodecDescriptor("baz").getTag());
  }

  @Test
  public void constantsOrderedByLastOccurrenceInIteration() {
    Object constant1 = new Object();
    Object constant2 = new Object();
    ObjectCodecRegistry underTest1 =
        ObjectCodecRegistry.newBuilder()
            .setAllowDefaultCodec(false)
            .addConstant(constant1)
            .addConstant(constant2)
            .addConstant(constant1)
            .build();
    ObjectCodecRegistry underTest2 =
        ObjectCodecRegistry.newBuilder()
            .setAllowDefaultCodec(false)
            .addConstant(constant1)
            .addConstant(constant2)
            .build();
    assertThat(underTest1.maybeGetTagForConstant(constant1)).isEqualTo(3);
    assertThat(underTest1.maybeGetTagForConstant(constant2)).isEqualTo(2);
    assertThat(underTest2.maybeGetTagForConstant(constant1)).isEqualTo(1);
    assertThat(underTest2.maybeGetTagForConstant(constant2)).isEqualTo(2);
  }

  @Test
  public void memoizingOrderedByClassName() {
    Memoizer.MemoizingCodec<String> memoizingCodec1 =
        new Memoizer.ObjectCodecAdaptor<>(SingletonCodec.of("value1", "mnemonic1"));
    Memoizer.MemoizingCodec<Object> memoizingCodec2 =
        new Memoizer.ObjectCodecAdaptor<>(SingletonCodec.of(new Object(), "mnemonic2"));
    ObjectCodecRegistry underTest =
        ObjectCodecRegistry.newBuilder()
            .setAllowDefaultCodec(false)
            .addMemoizing(memoizingCodec1)
            .addMemoizing(memoizingCodec2)
            .build();

    assertThat(underTest.getMemoizingCodecDescriptor(Object.class).getMemoizingCodec())
        .isEqualTo(memoizingCodec2);
    assertThat(underTest.getMemoizingCodecDescriptor(String.class).getMemoizingCodec())
        .isEqualTo(memoizingCodec1);
    assertThat(underTest.maybeGetMemoizingCodecByTag(1)).isEqualTo(memoizingCodec2);
    assertThat(underTest.maybeGetMemoizingCodecByTag(2)).isEqualTo(memoizingCodec1);
  }

  @Test
  public void testGetBuilder() throws NoCodecException {
    SingletonCodec<String> codec1 = SingletonCodec.of("value1", "mnemonic1");
    SingletonCodec<String> codec2 = SingletonCodec.of("value2", "mnemonic2");
    Object constant = new Object();
    Memoizer.MemoizingCodec<String> memoizingCodec = new Memoizer.ObjectCodecAdaptor<>(codec1);

    ObjectCodecRegistry underTest =
        ObjectCodecRegistry.newBuilder()
            .setAllowDefaultCodec(false)
            .add("foo", codec1)
            .add("bar", codec2)
            .addConstant(constant)
            .addMemoizing(memoizingCodec)
            .build();

    ObjectCodecRegistry copy = underTest.getBuilder().build();
    assertThat(copy.getCodecDescriptor("bar").getTag()).isEqualTo(1);
    assertThat(copy.getCodecDescriptor("foo").getTag()).isEqualTo(2);
    assertThat(copy.getMemoizingCodecDescriptor(String.class).getMemoizingCodec())
        .isEqualTo(memoizingCodec);
    assertThat(copy.maybeGetTagForConstant(constant)).isNotNull();
    assertThrows(NoCodecException.class, () -> copy.getCodecDescriptor("baz"));
  }
}
