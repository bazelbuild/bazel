package com.google.devtools.build.lib.hash;

import java.util.concurrent.locks.ReentrantLock;

public class Blake3Hasher {
  // These constants match the native definitions in:
  // https://github.com/BLAKE3-team/BLAKE3/blob/master/c/blake3.h
  public static final int KEY_LEN = 32;
  public static final int OUT_LEN = 32;

  private final ReentrantLock rl = new ReentrantLock();
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
    rl.lock();
    checkValid();

    try {
      Blake3JNI.blake3_hasher_update(hasher, data, data.length);
    } finally {
      rl.unlock();
    }
  }

  public void update(byte[] data, int length) {
    rl.lock();
    checkValid();

    try {
      Blake3JNI.blake3_hasher_update(hasher, data, length);
    } finally {
      rl.unlock();
    }
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
