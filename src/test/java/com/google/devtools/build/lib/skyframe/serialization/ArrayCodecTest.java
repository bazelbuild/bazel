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

import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.skyframe.serialization.testutils.TestUtils;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.math.BigInteger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ArrayCodec}. */
@RunWith(JUnit4.class)
public final class ArrayCodecTest {

  @Test
  public void objectArray() throws Exception {
    Object[] instance = new Object[2];
    instance[0] = "hi";
    Object[] inner = new Object[2];
    inner[0] = "inner1";
    inner[1] = null;
    instance[1] = inner;
    new SerializationTester(new Object[0], instance)
        .setVerificationFunction(ArrayCodecTest::verifyDeserialized)
        .runTests();
  }

  @Test
  public void typedArray() throws Exception {
    new SerializationTester(
            new BigInteger[] {},
            new BigInteger[] {BigInteger.ZERO},
            new BigInteger[] {BigInteger.ZERO, BigInteger.ONE, BigInteger.TWO})
        .addCodec(ArrayCodec.forComponentType(BigInteger.class))
        .runTests();
  }

  @Test
  public void stackOverflowTransformedIntoSerializationException() {
    class Foo {}

    Foo foo = new Foo();

    class FooCodec implements ObjectCodec<Foo> {
      @Override
      public Class<? extends Foo> getEncodedClass() {
        return Foo.class;
      }

      @Override
      public void serialize(SerializationContext context, Foo obj, CodedOutputStream codedOut) {
        if (obj == foo) {
          throw new StackOverflowError();
        }
      }

      @Override
      public Foo deserialize(DeserializationContext context, CodedInputStream codedIn) {
        throw new UnsupportedOperationException();
      }
    }

    SerializationContext serializationContext =
        new SerializationContext(
            ObjectCodecRegistry.newBuilder().add(new FooCodec()).build(),
            ImmutableClassToInstanceMap.of());
    // Serialize an array containing a special object of a special class for which the code always
    // will throw a StackOverflowError. This way we exercise the catch block in ArrayCodec without
    // having to cause a real StackOverflowError to organically occur.
    //
    // We used to take that approach of trying to get a real StackOverflowError to occur, by
    // serializing an array-of-array-of-... nested thousands of times. But this approach was
    // brittle: On different machines/architectures that have a lot of stack memory, the JVM
    // wouldn't organically throw a StackOverflowError, and when we increased the nesting depth that
    // caused segfaults on machines/architectures with less stack memory.
    Object[] array = {foo};
    assertThrows(
        SerializationException.class, () -> TestUtils.toBytes(serializationContext, array));
  }

  private static void verifyDeserialized(Object[] original, Object[] deserialized) {
    assertThat(deserialized).hasLength(original.length);
    for (int i = 0; i < deserialized.length; i++) {
      if (original[i] instanceof Object[]) {
        assertThat(deserialized[i]).isInstanceOf(Object[].class);
        verifyDeserialized((Object[]) original[i], (Object[]) deserialized[i]);
      } else {
        assertThat(deserialized[i]).isEqualTo(original[i]);
      }
    }
  }
}
