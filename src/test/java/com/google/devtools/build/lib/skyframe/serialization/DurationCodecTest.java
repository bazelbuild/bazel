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

import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import java.time.Duration;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** A test for {@link DurationCodec} */
@RunWith(JUnit4.class)
public class DurationCodecTest {

  private static final int MAX_NANO_ADJUSTMENT = 999999999;
  private static final int MIN_NANO_ADJUSTMENT = 0;

  @Test
  public void testDurationCodec() throws Exception {
    new SerializationTester(
            Duration.ofMinutes(123),
            Duration.ofSeconds(231),
            Duration.ofDays(321),
            Duration.ofSeconds(Long.MAX_VALUE),
            Duration.ZERO,
            Duration.ofSeconds(Long.MIN_VALUE),
            Duration.ofSeconds(Long.MAX_VALUE, MAX_NANO_ADJUSTMENT),
            Duration.ofSeconds(Long.MAX_VALUE, MIN_NANO_ADJUSTMENT),
            Duration.ofSeconds(Long.MIN_VALUE, 123),
            Duration.ofSeconds(Long.MAX_VALUE, 321))
        .runTests();
  }
}
