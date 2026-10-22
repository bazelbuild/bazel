// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.sandbox.backend;

import build.bazel.remote.execution.v2.Digest;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.protobuf.ByteString;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;

/**
 * The convention by which a sandbox backend exchanges file locations instead of file contents.
 *
 * <p>Bazel never copies input files to the backend. Instead, the bytes it streams to {@code
 * ByteStream.Write} for the digest of an input file are the absolute, percent-encoded {@code
 * file:///} URI of the file, and the backend binds the digest to that location. In the other
 * direction, the bytes the backend streams to {@code ByteStream.Read} for an output file may be the
 * URI of a staged copy of the file.
 *
 * <p>Since the digest that names the blob still describes the file contents, a receiver tells a
 * location apart from a real blob as follows: the data is a location if and only if it starts with
 * {@code file://} and is not the blob described by the digest, i.e., its length is not the digest's
 * size or its hash is not the digest's hash.
 */
public final class LocationBlobs {
  private static final ByteString SCHEME = ByteString.copyFromUtf8("file://");

  private LocationBlobs() {}

  /**
   * Encodes the absolute path {@code path} as a location.
   *
   * <p>The path is a Bazel internal string, i.e., every character stands for one byte of the path
   * as seen by the operating system.
   */
  public static ByteString encode(String path) {
    if (!path.startsWith("/")) {
      // A relative path would be parsed as having a host.
      throw new IllegalArgumentException("Not an absolute path: " + path);
    }
    byte[] bytes = path.getBytes(StandardCharsets.ISO_8859_1);
    ByteString.Output out = ByteString.newOutput(SCHEME.size() + bytes.length);
    out.write(SCHEME.toByteArray(), 0, SCHEME.size());
    for (byte b : bytes) {
      if (isUnreserved(b) || b == '/') {
        out.write(b);
      } else {
        out.write('%');
        out.write(Character.toUpperCase(Character.forDigit((b >> 4) & 0xF, 16)));
        out.write(Character.toUpperCase(Character.forDigit(b & 0xF, 16)));
      }
    }
    return out.toByteString();
  }

  /**
   * Returns whether {@code data}, as exchanged for {@code digest}, is a location rather than the
   * blob described by {@code digest}.
   */
  public static boolean isLocation(ByteString data, Digest digest, DigestUtil digestUtil) {
    if (!data.startsWith(SCHEME)) {
      return false;
    }
    return data.size() != digest.getSizeBytes()
        || !digestUtil.compute(data).getHash().equals(digest.getHash());
  }

  /**
   * Returns the path a location refers to, as a Bazel internal string.
   *
   * @throws IllegalArgumentException if {@code data} is not a well-formed location
   */
  public static String decode(ByteString data) {
    if (!data.startsWith(SCHEME)) {
      throw new IllegalArgumentException("Not a file URI");
    }
    byte[] bytes = data.substring(SCHEME.size()).toByteArray();
    if (bytes.length == 0 || bytes[0] != '/') {
      throw new IllegalArgumentException("File URI with a host or without a path");
    }
    ByteArrayOutputStream out = new ByteArrayOutputStream(bytes.length);
    for (int i = 0; i < bytes.length; i++) {
      if (bytes[i] != '%') {
        out.write(bytes[i]);
        continue;
      }
      if (i + 2 >= bytes.length) {
        throw new IllegalArgumentException("Truncated percent-encoding");
      }
      int hi = Character.digit(bytes[i + 1], 16);
      int lo = Character.digit(bytes[i + 2], 16);
      if (hi < 0 || lo < 0) {
        throw new IllegalArgumentException("Invalid percent-encoding");
      }
      out.write((hi << 4) | lo);
      i += 2;
    }
    return new String(out.toByteArray(), StandardCharsets.ISO_8859_1);
  }

  private static boolean isUnreserved(byte b) {
    return (b >= 'a' && b <= 'z')
        || (b >= 'A' && b <= 'Z')
        || (b >= '0' && b <= '9')
        || b == '-'
        || b == '.'
        || b == '_'
        || b == '~';
  }
}
