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
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ArrayCodec}. */
@RunWith(JUnit4.class)
public class ArrayCodecTest {
  @Test
  public void smoke() throws Exception {
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
  public void stackOverflowTransformedIntoSerializationException() {
    int depth = 4000;
    Object[] obj = new Object[1];
    Object[] cur = obj;
    for (int i = 0; i < depth; i++) {
      cur[0] = new Object[1];
      cur = (Object[]) cur[0];
    }
    assertThrows(
        SerializationException.class,
        () -> TestUtils.toBytes(new SerializationContext(ImmutableClassToInstanceMap.of()), obj));
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
