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
package com.google.devtools.build.lib.util.io;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link LinePrefixingOutputStream}.
 */
@RunWith(JUnit4.class)
public class LinePrefixingOutputStreamTest {

  private byte[] bytes(String string) {
    return string.getBytes(UTF_8);
  }

  private String string(byte[] bytes) {
    return new String(bytes, UTF_8);
  }

  private ByteArrayOutputStream out = new ByteArrayOutputStream();
  private LinePrefixingOutputStream prefixOut =
      new LinePrefixingOutputStream("Prefix: ", out);

  @Test
  public void testNoOutputUntilNewline() throws IOException {
    prefixOut.write(bytes("We won't be seeing any output."));
    assertThat(string(out.toByteArray())).isEmpty();
  }

  @Test
  public void testOutputIfFlushed() throws IOException {
    prefixOut.write(bytes("We'll flush after this line."));
    prefixOut.flush();
    assertThat(string(out.toByteArray())).isEqualTo("Prefix: We'll flush after this line.\n");
  }

  @Test
  public void testAutoflushUponNewline() throws IOException {
    prefixOut.write(bytes("Hello, newline.\n"));
    assertThat(string(out.toByteArray())).isEqualTo("Prefix: Hello, newline.\n");
  }

  @Test
  public void testAutoflushUponEmbeddedNewLine() throws IOException {
    prefixOut.write(bytes("Hello line1.\nHello line2.\nHello line3.\n"));
    assertThat(string(out.toByteArray()))
        .isEqualTo("Prefix: Hello line1.\nPrefix: Hello line2.\nPrefix: Hello line3.\n");
  }

  @Test
  public void testBufferMaxLengthFlush() throws IOException {
    String junk = "lots of characters of non-newline junk. ";
    while (junk.length() < LineFlushingOutputStream.BUFFER_LENGTH) {
      junk = junk + junk;
    }
    junk = junk.substring(0, LineFlushingOutputStream.BUFFER_LENGTH);

    // Also test bug where write on a full buffer blows up
    prefixOut.write(bytes(junk + junk));
    prefixOut.write(bytes(junk + junk));
    prefixOut.write(bytes("x"));
    assertThat(string(out.toByteArray())).isEqualTo("Prefix: " + junk + "\nPrefix: " + junk
        + "\nPrefix: " + junk + "\nPrefix: " + junk + "\n");
  }
}
