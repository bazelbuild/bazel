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

import com.google.common.io.ByteStreams;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UnsupportedEncodingException;
import java.util.Arrays;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test for {@link StreamMultiplexer}.
 */
@RunWith(JUnit4.class)
public class StreamMultiplexerTest {

  private ByteArrayOutputStream multiplexed;
  private OutputStream out;
  private OutputStream err;
  private OutputStream ctl;

  @Before
  public final void createOutputStreams() throws Exception  {
    multiplexed = new ByteArrayOutputStream();
    StreamMultiplexer multiplexer = new StreamMultiplexer(multiplexed);
    out = multiplexer.createStdout();
    err = multiplexer.createStderr();
    ctl = multiplexer.createControl();
  }

  @Test
  public void testEmptyWire() throws IOException {
    out.flush();
    err.flush();
    ctl.flush();
    assertThat(multiplexed.toByteArray()).isEmpty();
  }

  private static byte[] getLatin(String string)
      throws UnsupportedEncodingException {
    return string.getBytes("ISO-8859-1");
  }

  @Test
  public void testHelloWorldOnStdOut() throws Exception {
    out.write(getLatin("Hello, world."));
    out.flush();
    assertMessage(multiplexed.toByteArray(), 0, "Hello, world.");
  }

  @Test
  public void testInterleavedStdoutStderrControl() throws Exception {
    int start = 0;
    out.write(getLatin("Hello, stdout."));
    out.flush();
    assertMessage(multiplexed.toByteArray(), start, "Hello, stdout.");
    start = multiplexed.toByteArray().length;

    err.write(getLatin("Hello, stderr."));
    err.flush();
    assertMessage(multiplexed.toByteArray(), start, "Hello, stderr.");
    start = multiplexed.toByteArray().length;

    ctl.write(getLatin("Hello, control."));
    ctl.flush();
    assertMessage(multiplexed.toByteArray(), start, "Hello, control.");
    start = multiplexed.toByteArray().length;

    out.write(getLatin("... and back!"));
    out.flush();
    assertMessage(multiplexed.toByteArray(), start, "... and back!");
  }

  @Test
  public void testWillNotCommitToUnderlyingStreamUnlessFlushOrNewline()
      throws Exception {
    out.write(getLatin("There are no newline characters in here, so it won't" +
        " get written just yet."));
    assertThat(new byte[0]).isEqualTo(multiplexed.toByteArray());
  }

  @Test
  public void testNewlineTriggersFlush() throws Exception {
    out.write(getLatin("No newline just yet, so no flushing. "));
    assertThat(new byte[0]).isEqualTo(multiplexed.toByteArray());
    out.write(getLatin("OK, here we go:\nAnd more to come."));
    assertMessage(
        multiplexed.toByteArray(), 0, "No newline just yet, so no flushing. OK, here we go:\n");
    int firstMessageLength = multiplexed.toByteArray().length;
    out.write((byte) '\n');
    assertMessage(multiplexed.toByteArray(), firstMessageLength, "And more to come.\n");
  }

  @Test
  public void testFlush() throws Exception {
    out.write(getLatin("Don't forget to flush!"));
    assertThat(multiplexed.toByteArray()).isEqualTo(new byte[0]);
    out.flush(); // now the output will appear in multiplexed.
    assertStartsWith(multiplexed.toByteArray(), 1, 0, 0, 0);
    assertMessage(multiplexed.toByteArray(), 0, "Don't forget to flush!");
  }

  @Test
  public void testByteEncoding() throws IOException {
    OutputStream devNull = ByteStreams.nullOutputStream();
    StreamDemultiplexer demux = new StreamDemultiplexer((byte) 1, devNull);
    StreamMultiplexer mux = new StreamMultiplexer(demux);
    OutputStream out = mux.createStdout();

    // When we cast 266 to a byte, we get 10. So basically, we ended up
    // comparing 266 with 10 as an integer (because out.write takes an int),
    // and then later cast it to 10. This way we'd end up with a control
    // character \n in the middle of the payload which would then screw things
    // up when the real control character arrived. The fixed version of the
    // StreamMultiplexer avoids this problem by always casting to a byte before
    // carrying out any comparisons.

    out.write(266);
    out.write(10);
  }

  private static void assertStartsWith(byte[] actual, int... expectedPrefix){
    for (int i = 0; i < expectedPrefix.length; i++) {
      assertThat(actual[i]).isEqualTo(expectedPrefix[i]);
    }
  }

  private static void assertMessage(byte[] actual, int start, String expected) throws Exception {
    assertThat(Arrays.copyOfRange(actual, start + 5, actual.length)).isEqualTo(getLatin(expected));
  }
}
