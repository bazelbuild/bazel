package com.google.devtools.build.lib.hash;

import com.google.common.hash.HashCode;
import com.google.common.hash.Hasher;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.locks.ReentrantLock;

public class Blake3Hasher extends AbstractHasher {
  // These constants match the native definitions in:
  // https://github.com/BLAKE3-team/BLAKE3/blob/master/c/blake3.h
  public static final int KEY_LEN = 32;
  public static final int OUT_LEN = 32;

  private final ReentrantLock rl = new ReentrantLock();
  private static ThreadLocal<ByteBuffer> nativeByteBuffer = new ThreadLocal<ByteBuffer>();
  private long hasher = -1;

  public boolean isValid() {
    return (hasher != -1);
  }

  private void checkValid() {
    if (!isValid()) {
      throw new IllegalStateException("Native hasher not initialized");
    }
  }

  public Blake3Hasher() throws IllegalStateException {
    hasher = Blake3JNI.allocate_hasher();
    checkValid();
    Blake3JNI.blake3_hasher_init(hasher);
  }

  public void close() {
    if (isValid()) {
      rl.lock();
      try {
        Blake3JNI.delete_hasher(hasher);
      } finally {
        hasher = -1;
        rl.unlock();
      }
    }
  }

  public void initDefault() {
    rl.lock();
    try {
      Blake3JNI.blake3_hasher_init(hasher);
    } finally {
      rl.unlock();
    }
  }

  public void initKeyed(byte[] key) {
    if (key.length != KEY_LEN) {
      throw new IllegalArgumentException("Invalid hasher key length");
    }

    rl.lock();
    try {
      Blake3JNI.blake3_hasher_init_keyed(hasher, key);
    } finally {
      rl.unlock();
    }
  }

  public void initDeriveKey(String context) {
    rl.lock();
    try {
      Blake3JNI.blake3_hasher_init_derive_key(hasher, context);
    } finally {
      rl.unlock();
    }
  }

   public void update(byte[] data) {
    update(data, 0, data.length);
  }

  public void update(byte[] data, int offset, int length) {
    ByteBuffer inputBuf = nativeByteBuffer.get();

    if (inputBuf == null || inputBuf.capacity() < data.length) {
      inputBuf = ByteBuffer.allocateDirect(data.length);
      inputBuf.order(ByteOrder.nativeOrder());
      nativeByteBuffer.set(inputBuf);
    }
    inputBuf.rewind();
    inputBuf.put(data);

    rl.lock();
    checkValid();

    try {
      Blake3JNI.blake3_hasher_update(hasher, inputBuf, offset, data.length);
    } finally {
      rl.unlock();
    }
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
    return HashCode.fromBytes(getOutput());
  }

  public byte[] getOutput() throws IllegalArgumentException {
    return getOutput(OUT_LEN);
  }

  public byte[] getOutput(int outputLength) throws IllegalArgumentException {
    byte[] retByteArray = new byte[outputLength];

    rl.lock();
    checkValid();
    try {
      Blake3JNI.blake3_hasher_finalize(hasher, retByteArray, outputLength);
    } finally {
      rl.unlock();
    }

    return retByteArray;
  }
}
