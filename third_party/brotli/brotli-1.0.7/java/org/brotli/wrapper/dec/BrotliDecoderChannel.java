/* Copyright 2017 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

package org.brotli.wrapper.dec;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.ClosedChannelException;
import java.nio.channels.ReadableByteChannel;

/**
 * ReadableByteChannel that wraps native brotli decoder.
 */
public class BrotliDecoderChannel extends Decoder implements ReadableByteChannel {
  /** The default internal buffer size used by the decoder. */
  private static final int DEFAULT_BUFFER_SIZE = 16384;

  private final Object mutex = new Object();

  /**
   * Creates a BrotliDecoderChannel.
   *
   * @param source underlying source
   * @param bufferSize intermediate buffer size
   * @param customDictionary initial LZ77 dictionary
   */
  public BrotliDecoderChannel(ReadableByteChannel source, int bufferSize) throws IOException {
    super(source, bufferSize);
  }

  public BrotliDecoderChannel(ReadableByteChannel source) throws IOException {
    this(source, DEFAULT_BUFFER_SIZE);
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
  public int read(ByteBuffer dst) throws IOException {
    synchronized (mutex) {
      if (closed) {
        throw new ClosedChannelException();
      }
      int result = 0;
      while (dst.hasRemaining()) {
        int outputSize = decode();
        if (outputSize <= 0) {
          return result == 0 ? outputSize : result;
        }
        result += consume(dst);
      }
      return result;
    }
  }
}
