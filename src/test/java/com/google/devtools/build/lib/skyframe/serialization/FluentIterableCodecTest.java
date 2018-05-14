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

import com.google.common.collect.FluentIterable;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link FluentIterableCodec}. */
@RunWith(JUnit4.class)
public final class FluentIterableCodecTest {

  @Test
  public void testCodec() throws Exception {
    new SerializationTester(
            FluentIterable.of("x"),
            FluentIterable.of("abc", "def"),
            Iterables.concat(ImmutableList.of("first", "second"), ImmutableList.of("third")))
        .setVerificationFunction(FluentIterableCodecTest::verifyEquals)
        .runTests();
  }

  private static void verifyEquals(FluentIterable<Object> first, FluentIterable<Object> second) {
    assertThat(first.toList()).isEqualTo(second.toList());
  }
}
