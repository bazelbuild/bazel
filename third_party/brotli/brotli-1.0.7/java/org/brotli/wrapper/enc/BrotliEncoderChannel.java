/* Copyright 2017 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

package org.brotli.wrapper.enc;

import java.io.IOException;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.channels.ClosedChannelException;
import java.nio.channels.WritableByteChannel;

/**
 * WritableByteChannel that wraps native brotli encoder.
 */
public class BrotliEncoderChannel extends Encoder implements WritableByteChannel {
  /** The default internal buffer size used by the decoder. */
  private static final int DEFAULT_BUFFER_SIZE = 16384;

  private final Object mutex = new Object();

  /**
   * Creates a BrotliEncoderChannel.
   *
   * @param destination underlying destination
   * @param params encoding settings
   * @param bufferSize intermediate buffer size
   */
  public BrotliEncoderChannel(WritableByteChannel destination, Encoder.Parameters params,
      int bufferSize) throws IOException {
    super(destination, params, bufferSize);
  }

  public BrotliEncoderChannel(WritableByteChannel destination, Encoder.Parameters params)
      throws IOException {
    this(destination, params, DEFAULT_BUFFER_SIZE);
  }

  public BrotliEncoderChannel(WritableByteChannel destination) throws IOException {
    this(destination, new Encoder.Parameters());
  }

  @Override
  public boolean isOpen() {
    synchronized (mutex) {
      return !closed;
    }
  }

  @Override
  public void close() throws IOException {
    synchronized (mutex) {
      super.close();
    }
  }

  @Override
  public int write(ByteBuffer src) throws IOException {
    synchronized (mutex) {
      if (closed) {
        throw new ClosedChannelException();
      }
      int result = 0;
      while (src.hasRemaining() && encode(EncoderJNI.Operation.PROCESS)) {
        int limit = Math.min(src.remaining(), inputBuffer.remaining());
        ByteBuffer slice = src.slice();
        ((Buffer) slice).limit(limit);
        inputBuffer.put(slice);
        result += limit;
        ((Buffer) src).position(src.position() + limit);
      }
      return result;
    }
  }
}
