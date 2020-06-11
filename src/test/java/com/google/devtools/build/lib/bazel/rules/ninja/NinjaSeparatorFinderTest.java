// Copyright 2019 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.rules.ninja;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.bazel.rules.ninja.file.FileFragment;
import com.google.devtools.build.lib.bazel.rules.ninja.file.IncorrectSeparatorException;
import com.google.devtools.build.lib.bazel.rules.ninja.file.NinjaSeparatorFinder;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link NinjaSeparatorFinder}. */
@RunWith(JUnit4.class)
public class NinjaSeparatorFinderTest {
  @Test
  public void testIsSeparator() throws IncorrectSeparatorException {
    doTestIsSeparator(" \na", 1);
    doTestIsSeparator("b\na", 1);
    doTestIsSeparator(" \na", 1);
    doTestIsSeparator("b\n$", 1);
    doTestIsSeparator(" \n\n", 1);
    doTestIsSeparator("a\n\n", 1);
    // We are pointing to the last symbol of separator.
    doTestIsSeparator("a\r\n\n", 2);
    doTestIsSeparator(" \r\n\n", 2);
    doTestIsSeparator("a\r\na", 2);
    doTestIsSeparator("\r\na", 1);

    doTestIsSeparator(" \n ", -1);
    doTestIsSeparator(" \r\n ", -1);
    doTestIsSeparator("$\n ", -1);
    doTestIsSeparator("$\r\n ", -1);
    doTestIsSeparator("$\r\n\n ", -1);
    doTestIsSeparator("$\r\n\r\n ", -1);
    doTestIsSeparator("$\n\n", -1);
    doTestIsSeparator("$\na", -1);
    doTestIsSeparator("$\r\na", -1);
    doTestIsSeparator("a\n ", -1);
    doTestIsSeparator("a\n\t", -1);
    // Not enough information.
    doTestIsSeparator("\r\n", -1);
    doTestIsSeparator("\n", -1);
    // Test for incorrect separators.
    byte[] bytes = "a\rb".getBytes(StandardCharsets.ISO_8859_1);
    ByteBuffer buffer = ByteBuffer.wrap(bytes);
    FileFragment fragment = new FileFragment(buffer, 0, 0, buffer.limit());
    assertThrows(
        IncorrectSeparatorException.class,
        () -> NinjaSeparatorFinder.findNextSeparator(fragment, 0, -1));
  }

  private static void doTestIsSeparator(String s, int expected) throws IncorrectSeparatorException {
    byte[] bytes = s.getBytes(StandardCharsets.ISO_8859_1);
    ByteBuffer buffer = ByteBuffer.wrap(bytes);
    FileFragment fragment = new FileFragment(buffer, 0, 0, buffer.limit());
    int result = NinjaSeparatorFinder.findNextSeparator(fragment, 0, -1);
    assertThat(result).isEqualTo(expected);
  }
}
