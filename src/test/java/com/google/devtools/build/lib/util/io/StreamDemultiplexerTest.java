// Copyright 2014 Google Inc. All rights reserved.
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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import com.google.devtools.build.lib.util.StringUtilities;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.ByteArrayOutputStream;
import java.io.OutputStream;
import java.io.UnsupportedEncodingException;
import java.util.Random;

/**
 * Tests {@link StreamDemultiplexer}.
 */
@RunWith(JUnit4.class)
public class StreamDemultiplexerTest {

  private ByteArrayOutputStream out = new ByteArrayOutputStream();
  private ByteArrayOutputStream err = new ByteArrayOutputStream();
  private ByteArrayOutputStream ctl = new ByteArrayOutputStream();

  private byte[] lines(String... lines) {
    try {
      return StringUtilities.joinLines(lines).getBytes("ISO-8859-1");
    } catch (UnsupportedEncodingException e) {
      throw new AssertionError(e);
    }
  }

  private String toAnsi(ByteArrayOutputStream stream) {
    try {
      return new String(stream.toByteArray(), "ISO-8859-1");
    } catch (UnsupportedEncodingException e) {
      throw new AssertionError(e);
    }
  }

  private byte[] inAnsi(String string) {
    try {
      return string.getBytes("ISO-8859-1");
    } catch (UnsupportedEncodingException e) {
      throw new AssertionError(e);
    }
  }

  @Test
  public void testHelloWorldOnStandardOut() throws Exception {
    byte[] multiplexed = lines("@1@", "Hello, world.");
    try (final StreamDemultiplexer demux = new StreamDemultiplexer((byte) '1', out)) {
      demux.write(multiplexed);
    }
    assertEquals("Hello, world.", out.toString("ISO-8859-1"));
  }

  @Test
  public void testOutErrCtl() throws Exception {
    byte[] multiplexed = lines("@1@", "out", "@2@", "err", "@3@", "ctl", "");
    try (final StreamDemultiplexer demux = new StreamDemultiplexer((byte) '1', out, err, ctl)) {
      demux.write(multiplexed);
    }
    assertEquals("out", toAnsi(out));
    assertEquals("err", toAnsi(err));
    assertEquals("ctl", toAnsi(ctl));
  }

  @Test
  public void testWithoutLineBreaks() throws Exception {
    byte[] multiplexed = lines("@1@", "just ", "@1@", "one ", "@1@", "line", "");
    try (final StreamDemultiplexer demux = new StreamDemultiplexer((byte) '1', out)) {
      demux.write(multiplexed);
    }
    assertEquals("just one line", out.toString("ISO-8859-1"));
  }

  @Test
  public void testLineBreaks() throws Exception {
    byte[] multiplexed = lines("@1", "two", "@1", "lines", "");
    try (StreamDemultiplexer demux = new StreamDemultiplexer((byte) '1', out)) {
      demux.write(multiplexed);
      demux.flush();
      assertEquals("two\nlines\n", out.toString("ISO-8859-1"));
    }
  }

  @Test
  public void testMultiplexAndBackWithHelloWorld() throws Exception {
    StreamDemultiplexer demux = new StreamDemultiplexer((byte) '1', out);
    StreamMultiplexer mux = new StreamMultiplexer(demux);
    OutputStream out = mux.createStdout();
    out.write(inAnsi("Hello, world."));
    out.flush();
    assertEquals("Hello, world.", toAnsi(this.out));
  }

  @Test
  public void testMultiplexDemultiplexBinaryStress() throws Exception {
    StreamDemultiplexer demux = new StreamDemultiplexer((byte) '1', out, err, ctl);
    StreamMultiplexer mux = new StreamMultiplexer(demux);
    OutputStream[] muxOuts = {mux.createStdout(), mux.createStderr(), mux.createControl()};
    ByteArrayOutputStream[] expectedOuts =
        {new ByteArrayOutputStream(), new ByteArrayOutputStream(), new ByteArrayOutputStream()};

    Random random = new Random(0xdeadbeef);
    for (int round = 0; round < 100; round++) {
      byte[] buffer = new byte[random.nextInt(100)];
      random.nextBytes(buffer);
      int streamId = random.nextInt(3);
      expectedOuts[streamId].write(buffer);
      expectedOuts[streamId].flush();
      muxOuts[streamId].write(buffer);
      muxOuts[streamId].flush();
    }
    assertArrayEquals(expectedOuts[0].toByteArray(), out.toByteArray());
    assertArrayEquals(expectedOuts[1].toByteArray(), err.toByteArray());
    assertArrayEquals(expectedOuts[2].toByteArray(), ctl.toByteArray());
  }
}
