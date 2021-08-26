// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.util.StringUtilities.joinLines;
import static com.google.devtools.build.lib.util.StringUtilities.prettyPrintBytes;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** A test for {@link StringUtilities}. */
@RunWith(JUnit4.class)
public class StringUtilitiesTest {

  @Test
  public void emptyLinesYieldsEmptyString() {
    assertThat(joinLines()).isEmpty();
  }

  @Test
  public void twoLinesGetjoinedNicely() {
    assertThat(joinLines("line 1", "line 2")).isEqualTo("line 1\nline 2");
  }

  @Test
  public void aTrailingNewlineIsAvailableWhenYouNeedIt() {
    assertThat(joinLines("two lines", "with trailing newline", ""))
        .isEqualTo("two lines\nwith trailing newline\n");
  }

  @Test
  public void replaceAllLiteral() throws Exception {
    assertThat(StringUtilities.replaceAllLiteral("bababa", "ba", "ab")).isEqualTo("ababab");
    assertThat(StringUtilities.replaceAllLiteral("bababa", "ba", "")).isEmpty();
    assertThat(StringUtilities.replaceAllLiteral("bababa", "", "ab")).isEqualTo("bababa");
  }

  @Test
  public void testPrettyPrintBytes() {
    String[] expected = {
      "2B",
      "23B",
      "234B",
      "2345B",
      "23KB",
      "234KB",
      "2345KB",
      "23MB",
      "234MB",
      "2345MB",
      "23456MB",
      "234GB",
      "2345GB",
      "23456GB",
    };
    double x = 2.3456;
    for (int ii = 0; ii < expected.length; ++ii) {
      assertThat(prettyPrintBytes((long) x)).isEqualTo(expected[ii]);
      x = x * 10.0;
    }
  }

  @Test
  public void sanitizeControlChars() {
    assertThat(StringUtilities.sanitizeControlChars("\000")).isEqualTo("<?>");
    assertThat(StringUtilities.sanitizeControlChars("\001")).isEqualTo("<?>");
    assertThat(StringUtilities.sanitizeControlChars("\r")).isEqualTo("\\r");
    assertThat(StringUtilities.sanitizeControlChars(" abc123")).isEqualTo(" abc123");
  }
}
