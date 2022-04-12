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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.skyframe.serialization.testutils.TestUtils;
import java.util.EnumMap;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link EnumMapCodec}. */
@RunWith(JUnit4.class)
public class EnumMapCodecTest {
  @Test
  public void smoke() throws Exception {
    new SerializationTester(
            new EnumMap<>(
                ImmutableMap.of(
                    TestEnum.FIRST, "first", TestEnum.THIRD, "third", TestEnum.SECOND, "second")),
            new EnumMap<>(TestEnum.class),
            new EnumMap<>(EmptyEnum.class))
        .runTests();
  }

  @Test
  public void throwsOnSubclass() {
    SerializationException exception =
        assertThrows(
            SerializationException.class,
            () ->
                TestUtils.toBytes(
                    new SerializationContext(ImmutableClassToInstanceMap.of()),
                    new SubEnum<>(TestEnum.class)));
    assertThat(exception).hasMessageThat().contains("Cannot serialize subclasses of EnumMap");
  }

  private enum TestEnum {
    FIRST,
    SECOND,
    THIRD
  }

  private enum EmptyEnum {}

  private static class SubEnum<E extends Enum<E>, V> extends EnumMap<E, V> {
    public SubEnum(Class<E> keyType) {
      super(keyType);
    }
  }
}
