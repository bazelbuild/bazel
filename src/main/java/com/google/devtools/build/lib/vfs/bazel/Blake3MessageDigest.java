package com.google.devtools.build.lib.vfs.bazel;

import java.nio.ByteBuffer;
import java.security.DigestException;
import java.security.MessageDigest;

public final class Blake3MessageDigest extends MessageDigest {
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

  public Blake3MessageDigest() {
    super("BLAKE3");
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
    if (buffer.remaining() == 0) {
      flush();
    }
    buffer.put(b);
  }

  private byte[] getOutput(int outputLength) {
    byte[] retByteArray = new byte[outputLength];

    if (hasher == -1) {
      // If no flush has happened yet; oneshot this.
      Blake3JNI.oneshot(buffer.array(), buffer.position(), retByteArray, outputLength);
    } else {
      flush();
      Blake3JNI.blake3_hasher_finalize_and_reset(hasher, retByteArray, outputLength);
    }

    buffer.clear();
    return retByteArray;
  }

  public Object clone() throws CloneNotSupportedException {
    throw new CloneNotSupportedException();
  }

  public void engineReset() {
    if (hasher != -1) {
      Blake3JNI.blake3_hasher_reset(hasher);
    }
    buffer.clear();
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
    if (hasher != -1) {
      Blake3JNI.blake3_hasher_close(hasher);
      hasher = -1;
    }
  }
}
