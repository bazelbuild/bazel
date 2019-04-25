/* Copyright 2017 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

package org.brotli.wrapper.enc;

import java.io.IOException;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.channels.WritableByteChannel;
import java.util.ArrayList;

/**
 * Base class for OutputStream / Channel implementations.
 */
public class Encoder {
  private final WritableByteChannel destination;
  private final EncoderJNI.Wrapper encoder;
  private ByteBuffer buffer;
  final ByteBuffer inputBuffer;
  boolean closed;

  /**
   * Brotli encoder settings.
   */
  public static final class Parameters {
    private int quality = -1;
    private int lgwin = -1;

    public Parameters() { }

    private Parameters(Parameters other) {
      this.quality = other.quality;
      this.lgwin = other.lgwin;
    }

    /**
     * @param quality compression quality, or -1 for default
     */
    public Parameters setQuality(int quality) {
      if (quality < -1 || quality > 11) {
        throw new IllegalArgumentException("quality should be in range [0, 11], or -1");
      }
      this.quality = quality;
      return this;
    }

    /**
     * @param lgwin log2(LZ window size), or -1 for default
     */
    public Parameters setWindow(int lgwin) {
      if ((lgwin != -1) && ((lgwin < 10) || (lgwin > 24))) {
        throw new IllegalArgumentException("lgwin should be in range [10, 24], or -1");
      }
      this.lgwin = lgwin;
      return this;
    }
  }

  /**
   * Creates a Encoder wrapper.
   *
   * @param destination underlying destination
   * @param params encoding parameters
   * @param inputBufferSize read buffer size
   */
  Encoder(WritableByteChannel destination, Parameters params, int inputBufferSize)
      throws IOException {
    if (inputBufferSize <= 0) {
      throw new IllegalArgumentException("buffer size must be positive");
    }
    if (destination == null) {
      throw new NullPointerException("destination can not be null");
    }
    this.destination = destination;
    this.encoder = new EncoderJNI.Wrapper(inputBufferSize, params.quality, params.lgwin);
    this.inputBuffer = this.encoder.getInputBuffer();
  }

  private void fail(String message) throws IOException {
    try {
      close();
    } catch (IOException ex) {
      /* Ignore */
    }
    throw new IOException(message);
  }

  /**
   * @param force repeat pushing until all output is consumed
   * @return true if all encoder output is consumed
   */
  boolean pushOutput(boolean force) throws IOException {
    while (buffer != null) {
      if (buffer.hasRemaining()) {
        destination.write(buffer);
      }
      if (!buffer.hasRemaining()) {
        buffer = null;
      } else if (!force) {
        return false;
      }
    }
    return true;
  }

  /**
   * @return true if there is space in inputBuffer.
   */
  boolean encode(EncoderJNI.Operation op) throws IOException {
    boolean force = (op != EncoderJNI.Operation.PROCESS);
    if (force) {
      ((Buffer) inputBuffer).limit(inputBuffer.position());
    } else if (inputBuffer.hasRemaining()) {
      return true;
    }
    boolean hasInput = true;
    while (true) {
      if (!encoder.isSuccess()) {
        fail("encoding failed");
      } else if (!pushOutput(force)) {
        return false;
      } else if (encoder.hasMoreOutput()) {
        buffer = encoder.pull();
      } else if (encoder.hasRemainingInput()) {
        encoder.push(op, 0);
      } else if (hasInput) {
        encoder.push(op, inputBuffer.limit());
        hasInput = false;
      } else {
        ((Buffer) inputBuffer).clear();
        return true;
      }
    }
  }

  void flush() throws IOException {
    encode(EncoderJNI.Operation.FLUSH);
  }

  void close() throws IOException {
    if (closed) {
      return;
    }
    closed = true;
    try {
      encode(EncoderJNI.Operation.FINISH);
    } finally {
      encoder.destroy();
      destination.close();
    }
  }

  /**
   * Encodes the given data buffer.
   */
  public static byte[] compress(byte[] data, Parameters params) throws IOException {
    if (data.length == 0) {
      byte[] empty = new byte[1];
      empty[0] = 6;
      return empty;
    }
    /* data.length > 0 */
    EncoderJNI.Wrapper encoder = new EncoderJNI.Wrapper(data.length, params.quality, params.lgwin);
    ArrayList<byte[]> output = new ArrayList<byte[]>();
    int totalOutputSize = 0;
    try {
      encoder.getInputBuffer().put(data);
      encoder.push(EncoderJNI.Operation.FINISH, data.length);
      while (true) {
        if (!encoder.isSuccess()) {
          throw new IOException("encoding failed");
        } else if (encoder.hasMoreOutput()) {
          ByteBuffer buffer = encoder.pull();
          byte[] chunk = new byte[buffer.remaining()];
          buffer.get(chunk);
          output.add(chunk);
          totalOutputSize += chunk.length;
        } else if (!encoder.isFinished()) {
          encoder.push(EncoderJNI.Operation.FINISH, 0);
        } else {
          break;
        }
      }
    } finally {
      encoder.destroy();
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

  public static byte[] compress(byte[] data) throws IOException {
    return compress(data, new Parameters());
  }
}
