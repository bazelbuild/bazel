package com.google.devtools.build.lib.hash;

import com.google.common.hash.HashCode;
import com.google.common.hash.Hasher;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class Blake3Hasher extends AbstractHasher {
  // These constants match the native definitions in:
  // https://github.com/BLAKE3-team/BLAKE3/blob/master/c/blake3.h
  public static final int KEY_LEN = 32;
  public static final int OUT_LEN = 32;

  // To reduce the number of calls made via JNI, buffer up to this many bytes.
  // If a call to "hash()" is made and less than this much data has been
  // written, a single JNI call will be made that initializes, hashes, and
  // cleans up the hasher, rather than making separate calls for each operation.
  public static final int ONESHOT_THRESHOLD = 8 * 1024;

  private ByteBuffer buffer;
  private long hasher = -1;

  private boolean isAllocated() {
    return (hasher != -1);
  }

  private void initOnce() {
    if (!isAllocated()) {
      hasher = Blake3JNI.allocate_and_initialize_hasher();
    }
  }

  private void resetBuffer(int minLength) {
    int length = Math.max(ONESHOT_THRESHOLD, minLength);
    if (buffer == null || buffer.capacity() < length) {
      buffer = ByteBuffer.allocateDirect(length);
      buffer.order(ByteOrder.nativeOrder());
    }
    buffer.clear();
  }

  public void update(byte[] data, int offset, int length) {
    if (buffer == null) {
      resetBuffer(length);
    }

    if (buffer.remaining() < length) {
      initOnce();
      Blake3JNI.blake3_hasher_update(hasher, buffer, 0, buffer.position());
      resetBuffer(length);
    }
    buffer.put(data, offset, length);
  }

  public void update(byte[] data) {
    update(data, 0, data.length);
  }

  public byte[] getOutput(int outputLength) throws IllegalArgumentException {
    byte[] retByteArray = new byte[outputLength];

    if (!isAllocated() && buffer != null) {
      Blake3JNI.blake3_hasher_oneshot(
          hasher, buffer, buffer.position(), retByteArray, outputLength);
    } else {
      initOnce();
      if (buffer != null && buffer.position() > 0) {
        Blake3JNI.blake3_hasher_update(hasher, buffer, 0, buffer.position());
      }
      Blake3JNI.blake3_hasher_finalize_and_close(hasher, retByteArray, outputLength);
      hasher = -1;
    }

    return retByteArray;
  }

  // The following overrides allow us to implement Hasher.
  @Override
  public Hasher putBytes(ByteBuffer b) {
    buffer = b;
    return this;
  }

  @Override
  public Hasher putBytes(byte[] bytes, int off, int len) {
    update(bytes, off, len);
    return this;
  }

  @Override
  public Hasher putBytes(byte[] bytes) {
    update(bytes, 0, bytes.length);
    return this;
  }

  @Override
  public Hasher putByte(byte b) {
    update(new byte[] {b});
    return this;
  }

  @Override
  public HashCode hash() {
    return HashCode.fromBytes(getOutput(OUT_LEN));
  }
}
