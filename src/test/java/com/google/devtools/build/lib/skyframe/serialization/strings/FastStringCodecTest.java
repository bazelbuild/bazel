// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.serialization.strings;

import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.skyframe.serialization.testutils.ObjectCodecTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link FastStringCodec}. */
@RunWith(JUnit4.class)
public class FastStringCodecTest {

  @Test
  public void testCodec() throws Exception {
    if (!FastStringCodec.isAvailable()) {
      // Not available on this platform, skip test.
      return;
    }

    ObjectCodecTester.newBuilder(new FastStringCodec())
        .verificationFunction(
            (original, deserialized) -> {
              // hashCode is stored in String. Because we're using Unsafe to bypass standard String
              // constructors, make sure it still works.
              new EqualsTester().addEqualityGroup(original, deserialized).testEquals();
            })
        .addSubjects(
            "ow now brown cow. ow now brown cow",
            "（╯°□°）╯︵┻━┻ string with utf8/ascii",
            "string with ascii/utf8 （╯°□°）╯︵┻━┻",
            "last character utf8 ╯",
            "last char only non-ascii ƒ",
            "ƒ",
            "╯",
            "",
            Character.toString((char) 0xc3))
        .buildAndRunTests();
  }
}
