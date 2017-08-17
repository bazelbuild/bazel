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

package com.google.devtools.build.lib.skyframe.serialization;

import com.google.common.testing.EqualsTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link FastStringCodec}. */
@RunWith(JUnit4.class)
public class FastStringCodecTest extends AbstractObjectCodecTest<String> {

  public FastStringCodecTest() {
    super(
        new FastStringCodec(),
        "ow now brown cow. ow now brown cow",
        "（╯°□°）╯︵┻━┻ string with utf8/ascii",
        "string with ascii/utf8 （╯°□°）╯︵┻━┻",
        "last character utf8 ╯",
        "last char only non-ascii ƒ",
        "ƒ",
        "╯",
        "",
        Character.toString((char) 0xc3));;
  }

  // hashCode is stored in String. Because we're using Unsafe to bypass standard String
  // constructors, make sure it still works.
  @Test
  public void testEqualsAndHashCodePreserved() throws Exception {
    String original1 = "hello world";
    String original2 = "dlrow olleh";

    // Equals tester tests equals and hash code.
    new EqualsTester()
        .addEqualityGroup(original1, fromBytes(toBytes(original1)))
        .addEqualityGroup(original2, fromBytes(toBytes(original2)))
        .testEquals();
  }
}
