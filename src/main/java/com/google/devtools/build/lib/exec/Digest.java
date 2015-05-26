// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.exec;

import static java.nio.charset.StandardCharsets.US_ASCII;

import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.util.Pair;
import com.google.protobuf.ByteString;
import com.google.protobuf.MessageLite;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/**
 * A utility class for obtaining MD5 digests.
 * Digests are represented as 32 characters in lowercase ASCII.
 */
public class Digest {

  public static final ByteString EMPTY_DIGEST = fromContent(new byte[]{});

  private Digest() {
  }

  /**
   * Get the digest from the given byte array.
   * @param bytes the byte array.
   * @return a digest.
   */
  public static ByteString fromContent(byte[] bytes) {
    MessageDigest md = newBuilder();
    md.update(bytes, 0, bytes.length);
    return toByteString(BaseEncoding.base16().lowerCase().encode(md.digest()));
  }

  /**
   * Get the digest from the given ByteBuffer.
   * @param buffer the ByteBuffer.
   * @return a digest.
   */
  public static ByteString fromBuffer(ByteBuffer buffer) {
    MessageDigest md = newBuilder();
    md.update(buffer);
    return toByteString(BaseEncoding.base16().lowerCase().encode(md.digest()));
  }

  /**
   * Gets the digest of the given proto.
   *
   * @param proto a protocol buffer.
   * @return the digest.
   */
  public static ByteString fromProto(MessageLite proto) {
    MD5OutputStream md5Stream = new MD5OutputStream();
    try {
      proto.writeTo(md5Stream);
    } catch (IOException e) {
      throw new IllegalStateException("Unexpected IOException: ", e);
    }
    return toByteString(md5Stream.getDigest());
  }

  /**
   * Gets the digest and size of a given VirtualActionInput.
   *
   * @param input the VirtualActionInput.
   * @return the digest and size.
   */
  public static Pair<ByteString, Long> fromVirtualActionInput(VirtualActionInput input)
      throws IOException {
    CountingMD5OutputStream md5Stream = new CountingMD5OutputStream();
    input.writeTo(md5Stream);
    ByteString digest = toByteString(md5Stream.getDigest());
    return Pair.of(digest, md5Stream.getSize());
  }

  /**
   * A Sink that does an online MD5 calculation, which avoids forcing us to keep the entire
   * proto message in memory.
   */
  private static class MD5OutputStream extends OutputStream {
    private final MessageDigest md = newBuilder();

    @Override
    public void write(int b) {
      md.update((byte) b);
    }

    @Override
    public void write(byte[] b, int off, int len) {
      md.update(b, off, len);
    }

    public String getDigest() {
      return BaseEncoding.base16().lowerCase().encode(md.digest());
    }
  }

  private static final class CountingMD5OutputStream extends MD5OutputStream {
    private long size;

    @Override
    public void write(int b) {
      super.write(b);
      size++;
    }

    @Override
    public void write(byte[] b, int off, int len) {
      super.write(b, off, len);
      size += len;
    }

    public long getSize() {
      return size;
    }
  }

  /**
   * @param digest the digest to check.
   * @return true iff digest is a syntactically legal digest. It must be 32
   *         characters of hex with lowercase letters.
   */
  public static boolean isDigest(ByteString digest) {
    if (digest == null || digest.size() != 32) {
      return false;
    }

    for (byte b : digest) {
      char c = (char) b;
      if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
        continue;
      }
      return false;
    }
    return true;
  }

  /**
   * @param digest the digest.
   * @return true iff the digest is that of an empty file.
   */
  public static boolean isEmpty(ByteString digest) {
    return digest.equals(EMPTY_DIGEST);
  }

  /**
   * @return a new MD5 digest builder.
   */
  public static MessageDigest newBuilder() {
    try {
      return MessageDigest.getInstance("md5");
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException("MD5 not available", e);
    }
  }

  /**
   * Convert a String digest into a ByteString using ascii.
   * @param digest the digest in ascii.
   * @return the digest as a ByteString.
   */
  public static ByteString toByteString(String digest) {
    return ByteString.copyFrom(digest.getBytes(US_ASCII));
  }
}
