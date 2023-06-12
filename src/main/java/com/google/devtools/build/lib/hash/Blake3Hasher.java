package com.google.devtools.build.lib.hash;

import static com.google.common.base.Preconditions.checkState;

import com.google.common.hash.Funnel;
import com.google.common.hash.HashCode;
import com.google.common.hash.Hasher;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.Charset;

public class Blake3Hasher implements Hasher {
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
  private boolean isDone;

  public Blake3Hasher() {
    isDone = false;
  }

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

    checkState(!isDone);
    isDone = true;

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

  /* The following methods implement the {Hasher} interface. */

  @CanIgnoreReturnValue
  public Hasher putBytes(ByteBuffer b) {
    buffer = b;
    return this;
  }

  @CanIgnoreReturnValue
  public Hasher putBytes(byte[] bytes, int off, int len) {
    update(bytes, off, len);
    return this;
  }

  @CanIgnoreReturnValue
  public Hasher putBytes(byte[] bytes) {
    update(bytes, 0, bytes.length);
    return this;
  }

  @CanIgnoreReturnValue
  public Hasher putByte(byte b) {
    update(new byte[] {b});
    return this;
  }

  public HashCode hash() {
    return HashCode.fromBytes(getOutput(OUT_LEN));
  }

  @CanIgnoreReturnValue
  public final Hasher putBoolean(boolean b) {
    return putByte(b ? (byte) 1 : (byte) 0);
  }

  @CanIgnoreReturnValue
  public final Hasher putDouble(double d) {
    return putLong(Double.doubleToRawLongBits(d));
  }

  @CanIgnoreReturnValue
  public final Hasher putFloat(float f) {
    return putInt(Float.floatToRawIntBits(f));
  }

  @CanIgnoreReturnValue
  public Hasher putUnencodedChars(CharSequence charSequence) {
    for (int i = 0, len = charSequence.length(); i < len; i++) {
      putChar(charSequence.charAt(i));
    }
    return this;
  }

  @CanIgnoreReturnValue
  public Hasher putString(CharSequence charSequence, Charset charset) {
    return putBytes(charSequence.toString().getBytes(charset));
  }

  @CanIgnoreReturnValue
  public Hasher putShort(short s) {
    putByte((byte) s);
    putByte((byte) (s >>> 8));
    return this;
  }

  @CanIgnoreReturnValue
  public Hasher putInt(int i) {
    putByte((byte) i);
    putByte((byte) (i >>> 8));
    putByte((byte) (i >>> 16));
    putByte((byte) (i >>> 24));
    return this;
  }

  @CanIgnoreReturnValue
  public Hasher putLong(long l) {
    for (int i = 0; i < 64; i += 8) {
      putByte((byte) (l >>> i));
    }
    return this;
  }

  @CanIgnoreReturnValue
  public Hasher putChar(char c) {
    putByte((byte) c);
    putByte((byte) (c >>> 8));
    return this;
  }

  @CanIgnoreReturnValue
  public <T extends Object> Hasher putObject(T instance, Funnel<? super T> funnel) {
    funnel.funnel(instance, this);
    return this;
  }
}
