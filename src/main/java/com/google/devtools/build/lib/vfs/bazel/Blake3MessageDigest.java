package com.google.devtools.build.lib.vfs.bazel;

import java.nio.ByteBuffer;
import java.security.DigestException;
import java.security.MessageDigest;

public final class Blake3MessageDigest extends MessageDigest {
  // These constants match the native definitions in:
  // https://github.com/BLAKE3-team/BLAKE3/blob/master/c/blake3.h
  public static final int KEY_LEN = 32;
  public static final int OUT_LEN = 32;

  private static int STATE_SIZE = Blake3JNI.hasher_size();
  private static byte[] INITIAL_STATE = new byte[STATE_SIZE];

  static {
    Blake3JNI.initialize_hasher(INITIAL_STATE);
  }

  // To reduce the number of calls made via JNI, buffer up to this many bytes
  // before updating the hasher.
  public static final int ONESHOT_THRESHOLD = 8 * 1024;

  private ByteBuffer buffer = ByteBuffer.allocate(ONESHOT_THRESHOLD);
  private byte[] hasher = new byte[STATE_SIZE];

  public Blake3MessageDigest() {
    super("BLAKE3");
    System.arraycopy(INITIAL_STATE, 0, hasher, 0, STATE_SIZE);
  }

  private void flush() {
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
    if (buffer.remaining() == 0) {
      flush();
    }
    buffer.put(b);
  }

  private byte[] getOutput(int outputLength) {
    flush();

    byte[] retByteArray = new byte[outputLength];
    Blake3JNI.blake3_hasher_finalize(hasher, retByteArray, outputLength);

    engineReset();
    return retByteArray;
  }

  public Object clone() throws CloneNotSupportedException {
    throw new CloneNotSupportedException();
  }

  public void engineReset() {
    buffer.clear();
    System.arraycopy(INITIAL_STATE, 0, hasher, 0, STATE_SIZE);
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
}
