// Copyright 2023 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.vfs.bazel;

import com.google.devtools.build.lib.jni.JniLoader;
import java.nio.ByteBuffer;
import java.security.DigestException;
import java.security.MessageDigest;

/** A {@link MessageDigest} for BLAKE3. */
public final class Blake3MessageDigest extends MessageDigest {
  // These constants match the native definitions in:
  // https://github.com/BLAKE3-team/BLAKE3/blob/master/c/blake3.h
  public static final int KEY_LEN = 32;
  public static final int OUT_LEN = 32;

  static {
    JniLoader.loadJni();
  }

  private static final int STATE_SIZE = hasher_size();
  private static final byte[] INITIAL_STATE = new byte[STATE_SIZE];

  static {
    initialize_hasher(INITIAL_STATE);
  }

  private final byte[] hasher = new byte[STATE_SIZE];
  private final byte[] oneByteArray = new byte[1];

  public Blake3MessageDigest() {
    super("BLAKE3");
    System.arraycopy(INITIAL_STATE, 0, hasher, 0, STATE_SIZE);
  }

  @Override
  public void engineUpdate(byte[] data, int offset, int length) {
    blake3_hasher_update(hasher, data, offset, length);
  }

  @Override
  public void engineUpdate(byte b) {
    oneByteArray[0] = b;
    engineUpdate(oneByteArray, 0, 1);
  }

  @Override
  public void engineUpdate(ByteBuffer input) {
    super.engineUpdate(input);
  }

  private byte[] getOutput(int outputLength) {
    byte[] retByteArray = new byte[outputLength];
    blake3_hasher_finalize(hasher, retByteArray, outputLength);

    engineReset();
    return retByteArray;
  }

  @Override
  public Object clone() throws CloneNotSupportedException {
    throw new CloneNotSupportedException();
  }

  @Override
  public void engineReset() {
    System.arraycopy(INITIAL_STATE, 0, hasher, 0, STATE_SIZE);
  }

  @Override
  public int engineGetDigestLength() {
    return OUT_LEN;
  }

  @Override
  public byte[] engineDigest() {
    return getOutput(OUT_LEN);
  }

  @Override
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

  public static final native int hasher_size();

  public static final native void initialize_hasher(byte[] hasher);

  public static final native void blake3_hasher_update(
      byte[] hasher, byte[] input, int offset, int inputLen);

  public static final native void blake3_hasher_finalize(byte[] hasher, byte[] out, int outLen);
}
