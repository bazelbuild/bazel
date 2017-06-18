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

import java.io.ByteArrayOutputStream;
import java.io.OutputStream;
import java.io.UnsupportedEncodingException;
import java.nio.charset.Charset;
import java.util.Random;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link StreamDemultiplexer}.
 */
@RunWith(JUnit4.class)
public class StreamDemultiplexerTest {

  private ByteArrayOutputStream out = new ByteArrayOutputStream();
  private ByteArrayOutputStream err = new ByteArrayOutputStream();
  private ByteArrayOutputStream ctl = new ByteArrayOutputStream();

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
    byte[] multiplexed = chunk(1, "Hello, world.");
    try (final StreamDemultiplexer demux = new StreamDemultiplexer((byte) 1, out)) {
      demux.write(multiplexed);
    }
    assertThat(out.toString("ISO-8859-1")).isEqualTo("Hello, world.");
  }

  @Test
  public void testOutErrCtl() throws Exception {
    byte[] multiplexed = concat(chunk(1, "out"), chunk(2, "err"), chunk(3, "ctl"));
    try (final StreamDemultiplexer demux = new StreamDemultiplexer((byte) 1, out, err, ctl)) {
      demux.write(multiplexed);
    }
    assertThat(toAnsi(out)).isEqualTo("out");
    assertThat(toAnsi(err)).isEqualTo("err");
    assertThat(toAnsi(ctl)).isEqualTo("ctl");
  }

  @Test
  public void testWithoutLineBreaks() throws Exception {
    byte[] multiplexed = concat(chunk(1, "just "), chunk(1, "one "), chunk(1, "line"));
    try (final StreamDemultiplexer demux = new StreamDemultiplexer((byte) 1, out)) {
      demux.write(multiplexed);
    }
    assertThat(out.toString("ISO-8859-1")).isEqualTo("just one line");
  }

  @Test
  public void testMultiplexAndBackWithHelloWorld() throws Exception {
    StreamDemultiplexer demux = new StreamDemultiplexer((byte) 1, out);
    StreamMultiplexer mux = new StreamMultiplexer(demux);
    OutputStream out = mux.createStdout();
    out.write(inAnsi("Hello, world."));
    out.flush();
    assertThat(toAnsi(this.out)).isEqualTo("Hello, world.");
  }

  @Test
  public void testMultiplexDemultiplexBinaryStress() throws Exception {
    StreamDemultiplexer demux = new StreamDemultiplexer((byte) 1, out, err, ctl);
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
    assertThat(out.toByteArray()).isEqualTo(expectedOuts[0].toByteArray());
    assertThat(err.toByteArray()).isEqualTo(expectedOuts[1].toByteArray());
    assertThat(ctl.toByteArray()).isEqualTo(expectedOuts[2].toByteArray());
  }

  private static byte[] chunk(int stream, String payload) {
    byte[] payloadBytes = payload.getBytes(Charset.defaultCharset());
    byte[] result = new byte[payloadBytes.length + 5];

    System.arraycopy(payloadBytes, 0, result, 5, payloadBytes.length);
    result[0] = (byte) stream;
    result[1] = (byte) (payloadBytes.length >> 24);
    result[2] = (byte) ((payloadBytes.length >> 16) & 0xff);
    result[3] = (byte) ((payloadBytes.length >> 8) & 0xff);
    result[4] = (byte) (payloadBytes.length & 0xff);
    return result;
  }

  private static byte[] concat(byte[]... chunks) {
    int length = 0;
    for (byte[] chunk : chunks) {
      length += chunk.length;
    }

    byte[] result = new byte[length];
    int previousChunks = 0;
    for (byte[] chunk : chunks) {
      System.arraycopy(chunk, 0, result, previousChunks, chunk.length);
      previousChunks += chunk.length;
    }
    return result;
  }
}
