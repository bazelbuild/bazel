/* Copyright 2017 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

package org.brotli.wrapper.dec;

import java.io.IOException;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.channels.ReadableByteChannel;
import java.util.ArrayList;

/**
 * Base class for InputStream / Channel implementations.
 */
public class Decoder {
  private static final ByteBuffer EMPTY_BUFER = ByteBuffer.allocate(0);
  private final ReadableByteChannel source;
  private final DecoderJNI.Wrapper decoder;
  ByteBuffer buffer;
  boolean closed;
  boolean eager;

  /**
   * Creates a Decoder wrapper.
   *
   * @param source underlying source
   * @param inputBufferSize read buffer size
   */
  public Decoder(ReadableByteChannel source, int inputBufferSize)
      throws IOException {
    if (inputBufferSize <= 0) {
      throw new IllegalArgumentException("buffer size must be positive");
    }
    if (source == null) {
      throw new NullPointerException("source can not be null");
    }
    this.source = source;
    this.decoder = new DecoderJNI.Wrapper(inputBufferSize);
  }

  private void fail(String message) throws IOException {
    try {
      close();
    } catch (IOException ex) {
      /* Ignore */
    }
    throw new IOException(message);
  }

  public void setEager(boolean eager) {
    this.eager = eager;
  }

  /**
   * Continue decoding.
   *
   * @return -1 if stream is finished, or number of bytes available in read buffer (> 0)
   */
  int decode() throws IOException {
    while (true) {
      if (buffer != null) {
        if (!buffer.hasRemaining()) {
          buffer = null;
        } else {
          return buffer.remaining();
        }
      }

      switch (decoder.getStatus()) {
        case DONE:
          return -1;

        case OK:
          decoder.push(0);
          break;

        case NEEDS_MORE_INPUT:
          // In "eager" more pulling preempts pushing.
          if (eager && decoder.hasOutput()) {
            buffer = decoder.pull();
            break;
          }
          ByteBuffer inputBuffer = decoder.getInputBuffer();
          ((Buffer) inputBuffer).clear();
          int bytesRead = source.read(inputBuffer);
          if (bytesRead == -1) {
            fail("unexpected end of input");
          }
          if (bytesRead == 0) {
            // No input data is currently available.
            buffer = EMPTY_BUFER;
            return 0;
          }
          decoder.push(bytesRead);
          break;

        case NEEDS_MORE_OUTPUT:
          buffer = decoder.pull();
          break;

        default:
          fail("corrupted input");
      }
    }
  }

  void discard(int length) {
    ((Buffer) buffer).position(buffer.position() + length);
    if (!buffer.hasRemaining()) {
      buffer = null;
    }
  }

  int consume(ByteBuffer dst) {
    ByteBuffer slice = buffer.slice();
    int limit = Math.min(slice.remaining(), dst.remaining());
    ((Buffer) slice).limit(limit);
    dst.put(slice);
    discard(limit);
    return limit;
  }

  void close() throws IOException {
    if (closed) {
      return;
    }
    closed = true;
    decoder.destroy();
    source.close();
  }

  /**
   * Decodes the given data buffer.
   */
  public static byte[] decompress(byte[] data) throws IOException {
    DecoderJNI.Wrapper decoder = new DecoderJNI.Wrapper(data.length);
    ArrayList<byte[]> output = new ArrayList<byte[]>();
    int totalOutputSize = 0;
    try {
      decoder.getInputBuffer().put(data);
      decoder.push(data.length);
      while (decoder.getStatus() != DecoderJNI.Status.DONE) {
        switch (decoder.getStatus()) {
          case OK:
            decoder.push(0);
            break;

          case NEEDS_MORE_OUTPUT:
            ByteBuffer buffer = decoder.pull();
            byte[] chunk = new byte[buffer.remaining()];
            buffer.get(chunk);
            output.add(chunk);
            totalOutputSize += chunk.length;
            break;

          default:
            throw new IOException("corrupted input");
        }
      }
    } finally {
      decoder.destroy();
    }
    if (output.size() == 1) {
      return output.get(0);
    }
    byte[] result = new byte[totalOutputSize];
    int offset = 0;
    for (byte[] chunk : output) {
      System.arraycopy(chunk, 0, result, offset, chunk.length);
      offset += chunk.length;
    }
    return result;
  }
}
