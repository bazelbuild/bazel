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

import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester.VerificationFunction;
import java.util.LinkedHashSet;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link LinkedHashSetCodec}. */
@RunWith(JUnit4.class)
public class LinkedHashSetCodecTest {

  @Test
  public void smokeTest() throws Exception {
    LinkedHashSet<String> subjectTwo = new LinkedHashSet<>();
    subjectTwo.add("one");
    subjectTwo.add("two");
    subjectTwo.add("three");
    LinkedHashSet<String> nullSet = new LinkedHashSet<>();
    nullSet.add(null);
    LinkedHashSet<String> mixedSet = new LinkedHashSet<>();
    mixedSet.add(null);
    mixedSet.add("memberOne");
    new SerializationTester(new LinkedHashSet<String>(), subjectTwo, nullSet, mixedSet)
        .setVerificationFunction(
            (VerificationFunction<LinkedHashSet<String>>)
                (deserialized, subject) -> {
                  assertThat(deserialized).isEqualTo(subject);
                  assertThat(deserialized.size()).isEqualTo(subject.size());
                })
        .runTests();
  }
}
