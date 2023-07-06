package com.google.devtools.build.lib.vfs.bazel;

import static com.google.common.base.Preconditions.checkState;

import com.google.common.hash.Funnel;
import com.google.common.hash.HashCode;
import com.google.common.hash.Hasher;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;

public final class Blake3Hasher implements Hasher {
  private Blake3MessageDigest messageDigest;
  private boolean isDone = false;

  public Blake3Hasher(Blake3MessageDigest blake3MessageDigest) {
    messageDigest = blake3MessageDigest;
  }

  /* The following methods implement the {Hasher} interface. */

  @CanIgnoreReturnValue
  public Hasher putBytes(ByteBuffer b) {
    messageDigest.engineUpdate(b);
    return this;
  }

  @CanIgnoreReturnValue
  public Hasher putBytes(byte[] bytes, int off, int len) {
    messageDigest.engineUpdate(bytes, off, len);
    return this;
  }

  @CanIgnoreReturnValue
  public Hasher putBytes(byte[] bytes) {
    messageDigest.engineUpdate(bytes, 0, bytes.length);
    return this;
  }

  @CanIgnoreReturnValue
  public Hasher putByte(byte b) {
    messageDigest.engineUpdate(new byte[] {b});
    return this;
  }

  public HashCode hash() throws IllegalStateException {
    checkState(!isDone);
    isDone = true;

    return HashCode.fromBytes(messageDigest.engineDigest());
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
