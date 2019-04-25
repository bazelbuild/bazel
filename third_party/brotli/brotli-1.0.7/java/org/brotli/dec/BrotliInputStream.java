/* Copyright 2015 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

package org.brotli.dec;

import java.io.IOException;
import java.io.InputStream;

/**
 * {@link InputStream} decorator that decompresses brotli data.
 *
 * <p> Not thread-safe.
 */
public class BrotliInputStream extends InputStream {

  public static final int DEFAULT_INTERNAL_BUFFER_SIZE = 256;

  /**
   * Internal buffer used for efficient byte-by-byte reading.
   */
  private byte[] buffer;

  /**
   * Number of decoded but still unused bytes in internal buffer.
   */
  private int remainingBufferBytes;

  /**
   * Next unused byte offset.
   */
  private int bufferOffset;

  /**
   * Decoder state.
   */
  private final State state = new State();

  /**
   * Creates a {@link InputStream} wrapper that decompresses brotli data.
   *
   * <p> For byte-by-byte reading ({@link #read()}) internal buffer with
   * {@link #DEFAULT_INTERNAL_BUFFER_SIZE} size is allocated and used.
   *
   * <p> Will block the thread until first {@link BitReader#CAPACITY} bytes of data of source
   * are available.
   *
   * @param source underlying data source
   * @throws IOException in case of corrupted data or source stream problems
   */
  public BrotliInputStream(InputStream source) throws IOException {
    this(source, DEFAULT_INTERNAL_BUFFER_SIZE);
  }

  /**
   * Creates a {@link InputStream} wrapper that decompresses brotli data.
   *
   * <p> For byte-by-byte reading ({@link #read()}) internal buffer of specified size is
   * allocated and used.
   *
   * <p> Will block the thread until first {@link BitReader#CAPACITY} bytes of data of source
   * are available.
   *
   * @param source compressed data source
   * @param byteReadBufferSize size of internal buffer used in case of
   *        byte-by-byte reading
   * @throws IOException in case of corrupted data or source stream problems
   */
  public BrotliInputStream(InputStream source, int byteReadBufferSize) throws IOException {
    if (byteReadBufferSize <= 0) {
      throw new IllegalArgumentException("Bad buffer size:" + byteReadBufferSize);
    } else if (source == null) {
      throw new IllegalArgumentException("source is null");
    }
    this.buffer = new byte[byteReadBufferSize];
    this.remainingBufferBytes = 0;
    this.bufferOffset = 0;
    try {
      Decode.initState(state, source);
    } catch (BrotliRuntimeException ex) {
      throw new IOException("Brotli decoder initialization failed", ex);
    }
  }

  public void setEager(boolean eager) {
    state.isEager = eager ? 1 : 0;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public void close() throws IOException {
    Decode.close(state);
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public int read() throws IOException {
    if (bufferOffset >= remainingBufferBytes) {
      remainingBufferBytes = read(buffer, 0, buffer.length);
      bufferOffset = 0;
      if (remainingBufferBytes == -1) {
        return -1;
      }
    }
    return buffer[bufferOffset++] & 0xFF;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public int read(byte[] destBuffer, int destOffset, int destLen) throws IOException {
    if (destOffset < 0) {
      throw new IllegalArgumentException("Bad offset: " + destOffset);
    } else if (destLen < 0) {
      throw new IllegalArgumentException("Bad length: " + destLen);
    } else if (destOffset + destLen > destBuffer.length) {
      throw new IllegalArgumentException(
          "Buffer overflow: " + (destOffset + destLen) + " > " + destBuffer.length);
    } else if (destLen == 0) {
      return 0;
    }
    int copyLen = Math.max(remainingBufferBytes - bufferOffset, 0);
    if (copyLen != 0) {
      copyLen = Math.min(copyLen, destLen);
      System.arraycopy(buffer, bufferOffset, destBuffer, destOffset, copyLen);
      bufferOffset += copyLen;
      destOffset += copyLen;
      destLen -= copyLen;
      if (destLen == 0) {
        return copyLen;
      }
    }
    try {
      state.output = destBuffer;
      state.outputOffset = destOffset;
      state.outputLength = destLen;
      state.outputUsed = 0;
      Decode.decompress(state);
      if (state.outputUsed == 0) {
        return -1;
      }
      return state.outputUsed + copyLen;
    } catch (BrotliRuntimeException ex) {
      throw new IOException("Brotli stream decoding failed", ex);
    }

    // <{[INJECTED CODE]}>
  }
}
