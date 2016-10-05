// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Strings;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link LineBufferedOutputStream} .
 */
@TestSpec(size = Suite.SMALL_TESTS)
@RunWith(JUnit4.class)
public class LineBufferedOutputStreamTest {
  private static class MockOutputStream extends OutputStream {
    private final List<String> writes = new ArrayList<>();

    @Override
    public void write(int byteAsInt) throws IOException {
      byte b = (byte) byteAsInt; // make sure we work with bytes in comparisons
      write(new byte[] {b}, 0, 1);
    }

    @Override
    public synchronized void write(byte[] b, int off, int inlen) throws IOException {
      writes.add(new String(b, off, inlen, StandardCharsets.UTF_8));
    }
  }

  private List<String> lineBuffer(String... inputs) throws Exception {
    MockOutputStream mockOutputStream = new MockOutputStream();
    try (LineBufferedOutputStream cut = new LineBufferedOutputStream(mockOutputStream, 6)) {
      for (String input : inputs) {
        cut.write(input.getBytes(StandardCharsets.UTF_8));
      }
    }

    return mockOutputStream.writes;
  }

  @Test
  public void testLineBuffering() throws Exception {
    String large = Strings.repeat("a", 100);

    assertThat(lineBuffer("foo\nbar")).containsExactly("foo\n", "bar");
    assertThat(lineBuffer("foobarfoobar")).containsExactly("foobar", "foobar");
    assertThat(lineBuffer("fivey\none\n")).containsExactly("fivey\n", "one\n");
    assertThat(lineBuffer("sixish\none\n")).containsExactly("sixish", "\n", "one\n");
    assertThat(lineBuffer("s")).containsExactly("s");
    assertThat(lineBuffer("\n\n\n\n")).containsExactly("\n", "\n", "\n", "\n");
    assertThat(lineBuffer("foo\n\nbar\n")).containsExactly("foo\n", "\n", "bar\n");

    assertThat(lineBuffer("a", "a", large, large, "a")).containsExactly(
        "aa", large, large, "a");

  }
}
