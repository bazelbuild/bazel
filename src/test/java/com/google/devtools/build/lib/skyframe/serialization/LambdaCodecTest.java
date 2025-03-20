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

import com.google.devtools.build.lib.skyframe.serialization.testutils.RoundTripping;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link LambdaCodec}. */
@RunWith(JUnit4.class)
public class LambdaCodecTest {
  private interface MyInterface {
    boolean func(String arg);
  }

  @Test
  public void smoke() throws Exception {
    List<Boolean> returnValue = new ArrayList<>();
    new SerializationTester(
            (Supplier<Void> & Serializable) () -> null,
            (Function<Object, String> & Serializable) Object::toString,
            (MyInterface & Serializable) arg -> "hello".equals(arg),
            (MyInterface & Serializable) "hello"::equals,
            (MyInterface & Serializable) (arg) -> !returnValue.isEmpty())
        // We can't compare lambdas for equality, just make sure they get deserialized.
        .setVerificationFunction((original, deserialized) -> assertThat(deserialized).isNotNull())
        .runTests();
  }

  @Test
  public void lambdaBehaviorPreserved() throws Exception {
    List<Boolean> returnValue = new ArrayList<>();
    MyInterface lambda = (MyInterface & Serializable) (arg) -> returnValue.isEmpty();
    MyInterface deserializedLambda = RoundTripping.roundTrip(lambda);
    assertThat(lambda.func("any")).isTrue();
    assertThat(deserializedLambda.func("any")).isTrue();
    returnValue.add(true);
    assertThat(lambda.func("any")).isFalse();
    // Deserialized object's list is not the same as original's. Changes to original aren't seen.
    assertThat(deserializedLambda.func("any")).isTrue();
  }

  @Test
  public void onlySerializableWorks() {
    MyInterface unserializableLambda = (arg) -> true;
    assertThrows(
        SerializationException.class,
        () -> RoundTripping.toBytesMemoized(unserializableLambda, AutoRegistry.get()));
  }
}
