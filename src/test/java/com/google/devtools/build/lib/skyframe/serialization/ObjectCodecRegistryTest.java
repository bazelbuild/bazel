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
import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecRegistry.CodecDescriptor;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException.NoCodecException;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ObjectCodecRegistry}. */
@RunWith(JUnit4.class)
public class ObjectCodecRegistryTest {

  @Test
  public void testDescriptorLookups() throws NoCodecException {
    SingletonCodec<String> codec1 = SingletonCodec.of("value1", "mnemonic1");
    SingletonCodec<Integer> codec2 = SingletonCodec.of(1, "mnemonic2");

    ObjectCodecRegistry underTest =
        ObjectCodecRegistry.newBuilder()
            .setAllowDefaultCodec(false)
            .add(codec1)
            .add(codec2)
            .build();

    CodecDescriptor fooDescriptor = underTest.getCodecDescriptorForObject("hello");
    assertThat(fooDescriptor.codec()).isSameInstanceAs(codec1);
    assertThat(underTest.getCodecDescriptorByTag(fooDescriptor.tag()))
        .isSameInstanceAs(fooDescriptor);

    CodecDescriptor barDescriptor = underTest.getCodecDescriptorForObject(1);
    assertThat(barDescriptor.codec()).isSameInstanceAs(codec2);
    assertThat(underTest.getCodecDescriptorByTag(barDescriptor.tag()))
        .isSameInstanceAs(barDescriptor);

    assertThat(barDescriptor.tag()).isNotEqualTo(fooDescriptor.tag());

    assertThrows(NoCodecException.class, () -> underTest.getCodecDescriptorForObject((byte) 1));
    assertThrows(NoCodecException.class, () -> underTest.getCodecDescriptorByTag(42));
  }

  @Test
  public void testDefaultCodecFallback() throws NoCodecException {
    SingletonCodec<String> codec = SingletonCodec.of("value1", "mnemonic1");

    ObjectCodecRegistry underTest =
        ObjectCodecRegistry.newBuilder()
            .setAllowDefaultCodec(true)
            .add(codec)
            .addClassName(Byte.class.getName())
            .addClassName(Integer.class.getName())
            .build();

    CodecDescriptor fooDescriptor = underTest.getCodecDescriptorForObject("value1");
    assertThat(fooDescriptor.codec()).isSameInstanceAs(codec);

    CodecDescriptor barDefaultDescriptor = underTest.getCodecDescriptorForObject(15);
    assertThat(barDefaultDescriptor.codec()).isNotSameInstanceAs(codec);
    assertThat(barDefaultDescriptor.tag()).isNotEqualTo(fooDescriptor.tag());
    assertThat(underTest.getCodecDescriptorByTag(barDefaultDescriptor.tag()))
        .isSameInstanceAs(barDefaultDescriptor);

    assertThat(underTest.getCodecDescriptorForObject((byte) 9).codec().getClass())
        .isSameInstanceAs(barDefaultDescriptor.codec().getClass());

    // Bogus tags still throw.
    assertThrows(NoCodecException.class, () -> underTest.getCodecDescriptorByTag(42));
  }

  @Test
  public void testStableTagOrdering() throws NoCodecException {
    SingletonCodec<String> codec1 = SingletonCodec.of("value1", "mnemonic1");
    SingletonCodec<Integer> codec2 = SingletonCodec.of(1, "mnemonic2");

    ObjectCodecRegistry underTest1 =
        ObjectCodecRegistry.newBuilder()
            .setAllowDefaultCodec(true)
            .add(codec1)
            .add(codec2)
            .addClassName(Byte.class.getName())
            .build();

    ObjectCodecRegistry underTest2 =
        ObjectCodecRegistry.newBuilder()
            .setAllowDefaultCodec(true)
            .add(codec2)
            .add(codec1)
            .addClassName(Byte.class.getName())
            .build();

    assertThat(underTest1.getCodecDescriptorForObject("value1").tag())
        .isEqualTo(underTest2.getCodecDescriptorForObject("value1").tag());
    assertThat(underTest1.getCodecDescriptorForObject(5).tag())
        .isEqualTo(underTest2.getCodecDescriptorForObject(5).tag());
    // Default codec.
    assertThat(underTest1.getCodecDescriptorForObject((byte) 10).tag())
        .isEqualTo(underTest2.getCodecDescriptorForObject((byte) 10).tag());
  }

  @Test
  public void constantsOrderedByLastOccurrenceInIteration() {
    Object constant1 = new Object();
    Object constant2 = new Object();
    ObjectCodecRegistry underTest1 =
        ObjectCodecRegistry.newBuilder()
            .setAllowDefaultCodec(false)
            .addReferenceConstant(constant1)
            .addReferenceConstant(constant2)
            .addReferenceConstant(constant1)
            .build();
    ObjectCodecRegistry underTest2 =
        ObjectCodecRegistry.newBuilder()
            .setAllowDefaultCodec(false)
            .addReferenceConstant(constant1)
            .addReferenceConstant(constant2)
            .build();
    assertThat(underTest1.maybeGetTagForConstant(constant1)).isEqualTo(3);
    assertThat(underTest1.maybeGetTagForConstant(constant2)).isEqualTo(2);
    assertThat(underTest2.maybeGetTagForConstant(constant1)).isEqualTo(1);
    assertThat(underTest2.maybeGetTagForConstant(constant2)).isEqualTo(2);
  }

  private static ObjectCodecRegistry.Builder builderWithThisClass() {
    return ObjectCodecRegistry.newBuilder().addClassName(ObjectCodecRegistryTest.class.getName());
  }

  @Test
  public void excludingPrefix() throws NoCodecException {
    ObjectCodecRegistry underTest = builderWithThisClass().build();
    CodecDescriptor descriptor = underTest.getCodecDescriptorForObject(this);
    assertThat(descriptor).isNotNull();
    assertThat(descriptor.codec()).isInstanceOf(DynamicCodec.class);
    ObjectCodecRegistry underTestWithExcludeList =
        builderWithThisClass()
            .excludeClassNamePrefix(this.getClass().getPackage().getName())
            .build();
    assertThrows(
        NoCodecException.class, () -> underTestWithExcludeList.getCodecDescriptorForObject(this));
    ObjectCodecRegistry underTestWithWideExcludeList =
        builderWithThisClass().excludeClassNamePrefix("com").build();
    assertThrows(
        NoCodecException.class,
        () -> underTestWithWideExcludeList.getCodecDescriptorForObject(this));
  }

  @Test
  public void testGetBuilder() throws NoCodecException {
    SingletonCodec<String> codec1 = SingletonCodec.of("value1", "mnemonic1");
    SingletonCodec<Integer> codec2 = SingletonCodec.of(1, "mnemonic2");
    Object constant = new Object();

    ObjectCodecRegistry underTest =
        ObjectCodecRegistry.newBuilder()
            .setAllowDefaultCodec(false)
            .add(codec1)
            .add(codec2)
            .addReferenceConstant(constant)
            .build();

    ObjectCodecRegistry copy = underTest.getBuilder().build();
    assertThat(copy.getCodecDescriptorForObject(12).tag()).isEqualTo(1);
    assertThat(copy.getCodecDescriptorForObject("value1").tag()).isEqualTo(2);
    assertThat(copy.maybeGetTagForConstant(constant)).isNotNull();
    assertThrows(NoCodecException.class, () -> copy.getCodecDescriptorForObject((byte) 5));
  }

  private enum TestEnum {
    ONE {
      @Override
      public int val() {
        return 1;
      }
    },
    TWO {
      @Override
      public int val() {
        return 2;
      }
    },
    THREE {
      @Override
      public int val() {
        return 3;
      }
    };

    @SuppressWarnings("unused")
    public abstract int val();
  }

  @SuppressWarnings("GetClassOnEnum")
  @Test
  public void testDefaultEnum() throws NoCodecException {
    assertThat(TestEnum.ONE.getClass()).isNotEqualTo(TestEnum.class);
    assertThat(TestEnum.ONE.getDeclaringClass()).isEqualTo(TestEnum.class);
    assertThat(TestEnum.ONE.getClass()).isNotEqualTo(TestEnum.TWO.getClass());

    ObjectCodecRegistry underTest =
        ObjectCodecRegistry.newBuilder()
            .setAllowDefaultCodec(true)
            .addClassName(TestEnum.class.getName())
            .addClassName(TestEnum.ONE.getClass().getName())
            .addClassName(TestEnum.TWO.getClass().getName())
            .addClassName(TestEnum.THREE.getClass().getName())
            .build();

    CodecDescriptor oneDescriptor = underTest.getCodecDescriptorForObject(TestEnum.ONE);
    CodecDescriptor twoDescriptor = underTest.getCodecDescriptorForObject(TestEnum.TWO);
    assertThat(oneDescriptor).isEqualTo(twoDescriptor);

    assertThat(oneDescriptor.codec().getEncodedClass()).isEqualTo(TestEnum.class);
  }

  @Test
  public void checksum_nullIfNotComputed() {
    ObjectCodecRegistry registry =
        ObjectCodecRegistry.newBuilder()
            .setAllowDefaultCodec(true)
            .add(SingletonCodec.of("value", "mnemonic"))
            .addClassName(Byte.class.getName())
            .addReferenceConstant(new Object())
            .computeChecksum(false)
            .build();
    assertThat(registry.getChecksum()).isNull();
  }

  @Test
  public void checksum_deterministic() {
    ObjectCodec<String> codec = SingletonCodec.of("value", "mnemonic");
    Object constant = new Object();
    byte[] checksum1 =
        ObjectCodecRegistry.newBuilder()
            .setAllowDefaultCodec(true)
            .add(codec)
            .addClassName(Byte.class.getName())
            .addReferenceConstant(constant)
            .computeChecksum(true)
            .build()
            .getChecksum();
    byte[] checksum2 =
        ObjectCodecRegistry.newBuilder()
            .setAllowDefaultCodec(true)
            .add(codec)
            .addClassName(Byte.class.getName())
            .addReferenceConstant(constant)
            .computeChecksum(true)
            .build()
            .getChecksum();
    assertThat(checksum1).isNotNull();
    assertThat(checksum1).isEqualTo(checksum2);
  }

  @Test
  public void checksum_sensitiveToChangeInAllowDefaultCodec() {
    ObjectCodec<String> codec = SingletonCodec.of("value", "mnemonic");
    Object constant = new Object();
    byte[] checksum1 =
        ObjectCodecRegistry.newBuilder()
            .setAllowDefaultCodec(true)
            .add(codec)
            .addClassName(Byte.class.getName())
            .addReferenceConstant(constant)
            .computeChecksum(true)
            .build()
            .getChecksum();
    byte[] checksum2 =
        ObjectCodecRegistry.newBuilder()
            .setAllowDefaultCodec(false)
            .add(codec)
            .addClassName(Byte.class.getName())
            .addReferenceConstant(constant)
            .computeChecksum(true)
            .build()
            .getChecksum();
    assertThat(checksum1).isNotNull();
    assertThat(checksum2).isNotNull();
    assertThat(checksum1).isNotEqualTo(checksum2);
  }

  @Test
  public void checksum_sensitiveToChangeInCodecs() {
    ObjectCodec<String> codec = SingletonCodec.of("value", "mnemonic");
    Object constant = new Object();
    byte[] checksum1 =
        ObjectCodecRegistry.newBuilder()
            .setAllowDefaultCodec(true)
            .add(codec)
            .addClassName(Byte.class.getName())
            .addReferenceConstant(constant)
            .computeChecksum(true)
            .build()
            .getChecksum();
    byte[] checksum2 =
        ObjectCodecRegistry.newBuilder()
            .setAllowDefaultCodec(false)
            .add(codec)
            .add(SingletonCodec.of("extra", "extra_mnemonic"))
            .addClassName(Byte.class.getName())
            .addReferenceConstant(constant)
            .computeChecksum(true)
            .build()
            .getChecksum();
    assertThat(checksum1).isNotNull();
    assertThat(checksum2).isNotNull();
    assertThat(checksum1).isNotEqualTo(checksum2);
  }

  @Test
  public void checksum_sensitiveToChangeInClassNames() {
    ObjectCodec<String> codec = SingletonCodec.of("value", "mnemonic");
    Object constant = new Object();
    byte[] checksum1 =
        ObjectCodecRegistry.newBuilder()
            .setAllowDefaultCodec(true)
            .add(codec)
            .addClassName(Byte.class.getName())
            .addReferenceConstant(constant)
            .computeChecksum(true)
            .build()
            .getChecksum();
    byte[] checksum2 =
        ObjectCodecRegistry.newBuilder()
            .setAllowDefaultCodec(false)
            .add(codec)
            .addClassName(Integer.class.getName())
            .addReferenceConstant(constant)
            .computeChecksum(true)
            .build()
            .getChecksum();
    assertThat(checksum1).isNotNull();
    assertThat(checksum2).isNotNull();
    assertThat(checksum1).isNotEqualTo(checksum2);
  }

  @Test
  public void checksum_sensitiveToChangeInReferenceConstants() {
    ObjectCodec<String> codec = SingletonCodec.of("value", "mnemonic");
    Object constant = new Object();
    byte[] checksum1 =
        ObjectCodecRegistry.newBuilder()
            .setAllowDefaultCodec(true)
            .add(codec)
            .addClassName(Byte.class.getName())
            .addReferenceConstant(constant)
            .computeChecksum(true)
            .build()
            .getChecksum();
    byte[] checksum2 =
        ObjectCodecRegistry.newBuilder()
            .setAllowDefaultCodec(false)
            .add(codec)
            .addClassName(Byte.class.getName())
            .addClassName(Integer.class.getName())
            .addReferenceConstant(constant)
            .addReferenceConstant("another constant")
            .computeChecksum(true)
            .build()
            .getChecksum();
    assertThat(checksum1).isNotNull();
    assertThat(checksum2).isNotNull();
    assertThat(checksum1).isNotEqualTo(checksum2);
  }

  sealed interface TestIntf permits ClassA, ClassB, ClassC {}

  @AutoCodec
  static final class ClassA implements TestIntf {}

  static final class ClassB implements TestIntf {

    static class Codec extends DeferredObjectCodec<TestIntf> {

      @Override
      public boolean autoRegister() {
        // Will be registered by the test explicitly.
        return false;
      }

      @Override
      public Class<? extends TestIntf> getEncodedClass() {
        return ClassB.class;
      }

      @Override
      public ImmutableSet<Class<? extends TestIntf>> additionalEncodedClasses() {
        return ImmutableSet.of(ClassA.class, ClassC.class);
      }

      @Override
      public void serialize(SerializationContext context, TestIntf obj, CodedOutputStream codedOut)
          throws SerializationException, IOException {
        // unused
      }

      @Override
      public DeferredValue<? extends TestIntf> deserializeDeferred(
          AsyncDeserializationContext context, CodedInputStream codedIn)
          throws SerializationException, IOException {
        return ClassB::new;
      }
    }
  }

  @AutoCodec
  static final class ClassC implements TestIntf {}

  @Test
  public void testDescriptorLookups_respectsInsertionOrder() throws Exception {
    var a = new ClassA();
    var b = new ClassB();
    var c = new ClassC();
    var tester = new SerializationTester(a, b, c);

    // The AutoCodecs for ClassA and ClassC will be added first.
    tester.addCodec(new ClassB.Codec());

    tester.setVerificationFunction(
        (in, out) ->
            // Expect that all three objects use ClassB.Codec() as it's registered as one of the
            // last.
            //
            // If there is implicit alphabetical ordering when processing codec descriptors,
            // then:
            // - a will be deserialized as an instance of A.class, or
            // - c will be deserialized as an instance of C.class.
            assertWithMessage("incorrect codec look up for %s", in)
                .that(out)
                .isInstanceOf(ClassB.class));
    tester.runTests();
  }
}
