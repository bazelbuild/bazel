package com.google.devtools.build.lib.vfs.bazel;

import static com.google.common.base.Preconditions.checkState;

import com.google.common.hash.Funnel;
import com.google.common.hash.HashCode;
import com.google.common.hash.Hasher;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.security.DigestException;
import java.security.MessageDigest;

public final class Blake3MessageDigest extends MessageDigest implements Hasher {
  // These constants match the native definitions in:
  // https://github.com/BLAKE3-team/BLAKE3/blob/master/c/blake3.h
  public static final int KEY_LEN = 32;
  public static final int OUT_LEN = 32;

  // To reduce the number of calls made via JNI, buffer up to this many bytes.
  // If a call to "hash()" is made and less than this much data has been
  // written, a single JNI call will be made that initializes, hashes, and
  // cleans up the hasher, rather than making separate calls for each operation.
  public static final int ONESHOT_THRESHOLD = 8 * 1024;
  private ByteBuffer buffer = ByteBuffer.allocate(ONESHOT_THRESHOLD);

  private long hasher = -1;
  private boolean isDone;

  public Blake3MessageDigest() {
    super("BLAKE3");
    isDone = false;
  }

  private void flush() {
    if (hasher == -1) {
      hasher = Blake3JNI.allocate_and_initialize_hasher();
    }

    if (buffer.position() > 0) {
      Blake3JNI.blake3_hasher_update(hasher, buffer.array(), buffer.position());
      buffer.clear();
    }
  }

  public void engineUpdate(byte[] data, int offset, int length) {
    while (length > 0) {
      int numToCopy = Math.min(length, buffer.remaining());
      buffer.put(data, offset, numToCopy);
      length -= numToCopy;
      offset += numToCopy;

      if (buffer.remaining() == 0) {
        flush();
      }
    }
  }

  public void engineUpdate(byte[] data) {
    engineUpdate(data, 0, data.length);
  }

  public void engineUpdate(byte b) {
    engineUpdate(new byte[] {b});
  }

  private byte[] getOutput(int outputLength) throws IllegalArgumentException {
    byte[] retByteArray = new byte[outputLength];

    checkState(!isDone);
    isDone = true;

    if (hasher == -1) {
      // If no flush has happened yet; oneshot this.
      Blake3JNI.oneshot(buffer.array(), buffer.position(), retByteArray, outputLength);
      buffer.clear();
    } else {
      flush();
      Blake3JNI.blake3_hasher_finalize_and_close(hasher, retByteArray, outputLength);
      hasher = -1;
    }
    return retByteArray;
  }

  public Object clone() throws CloneNotSupportedException {
    throw new CloneNotSupportedException();
  }

  public void engineReset() {
    if (hasher != -1) {
      Blake3JNI.blake3_hasher_close(hasher);
      hasher = -1;
    }
    buffer.clear();
    isDone = false;
  }

  public void engineUpdate(ByteBuffer input) {
    if (input.hasArray()) {
      engineUpdate(input.array());
    } else {
      byte[] bufCopy = new byte[input.position()];
      input.get(bufCopy);
      engineUpdate(bufCopy);
    }
  }

  public int engineGetDigestLength() {
    return OUT_LEN;
  }

  public byte[] engineDigest() {
    byte[] digestBytes = getOutput(OUT_LEN);
    return digestBytes;
  }

  public int engineDigest(byte[] buf, int off, int len) throws DigestException {
    if (len < OUT_LEN) {
      throw new DigestException("partial digests not returned");
    }
    if (buf.length - off < OUT_LEN) {
      throw new DigestException("insufficient space in the output buffer to store the digest");
    }

    byte[] digestBytes = getOutput(OUT_LEN);
    System.arraycopy(digestBytes, 0, buf, off, digestBytes.length);
    return digestBytes.length;
  }

  @Override
  protected void finalize() throws Throwable {
    engineReset();
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
