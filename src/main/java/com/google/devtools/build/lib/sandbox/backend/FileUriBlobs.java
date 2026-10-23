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
 * The convention by which a sandbox backend exchanges file URIs instead of file contents.
 *
 * <p>Bazel never copies input files to the backend. Instead, the bytes it streams to {@code
 * ByteStream.Write} for the digest of an input file are the absolute, percent-encoded {@code
 * file:///} URI of the file, and the backend binds the digest to that file. In the other
 * direction, the bytes the backend streams to {@code ByteStream.Read} for an output file may be the
 * URI of a staged copy of the file.
 *
 * <p>Since the digest that names the blob still describes the file contents, a receiver tells a
 * file URI apart from a real blob as follows: the data is a file URI if and only if it starts with
 * {@code file://} and is not the blob described by the digest, i.e., its length is not the digest's
 * size or its hash is not the digest's hash.
 *
 * <p>An absolute Windows path such as {@code C:/dir/file}, as Bazel writes them, is the URI {@code
 * file:///C:/dir/file}, with a literal colon after the drive letter. A Unix path that happens to
 * start with {@code /C:/} stays unambiguous, as its colon is percent-encoded, and every Unix path
 * is encoded as it always was.
 */
public final class FileUriBlobs {
  private static final ByteString SCHEME = ByteString.copyFromUtf8("file://");

  private FileUriBlobs() {}

  /**
   * Encodes the absolute path {@code path} as a file URI.
   *
   * <p>The path is a Bazel internal string, i.e., every character stands for one byte of the path
   * as seen by the operating system.
   */
  public static ByteString encode(String path) {
    boolean drive = isDrivePath(path);
    if (!path.startsWith("/") && !drive) {
      // A relative path would be parsed as having a host.
      throw new IllegalArgumentException("Not an absolute path: " + path);
    }
    byte[] bytes = path.getBytes(StandardCharsets.ISO_8859_1);
    ByteString.Output out = ByteString.newOutput(SCHEME.size() + bytes.length + 1);
    out.write(SCHEME.toByteArray(), 0, SCHEME.size());
    if (drive) {
      out.write('/');
    }
    for (int i = 0; i < bytes.length; i++) {
      byte b = bytes[i];
      if (isUnreserved(b) || b == '/' || (drive && i == 1)) {
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
   * Returns whether {@code data}, as exchanged for {@code digest}, is a file URI rather than the
   * blob described by {@code digest}.
   */
  public static boolean isFileUri(ByteString data, Digest digest, DigestUtil digestUtil) {
    if (!data.startsWith(SCHEME)) {
      return false;
    }
    return data.size() != digest.getSizeBytes()
        || !digestUtil.compute(data).getHash().equals(digest.getHash());
  }

  /**
   * Returns the path a file URI refers to, as a Bazel internal string.
   *
   * @throws IllegalArgumentException if {@code data} is not a well-formed file URI
   */
  public static String decode(ByteString data) {
    if (!data.startsWith(SCHEME)) {
      throw new IllegalArgumentException("Not a file URI");
    }
    byte[] bytes = data.substring(SCHEME.size()).toByteArray();
    if (bytes.length == 0 || bytes[0] != '/') {
      throw new IllegalArgumentException("File URI with a host or without a path");
    }
    // A literal colon after a drive letter, which encode never writes for a Unix path.
    int start =
        bytes.length >= 4
                && isLetter(bytes[1])
                && bytes[2] == ':'
                && bytes[3] == '/'
            ? 1
            : 0;
    ByteArrayOutputStream out = new ByteArrayOutputStream(bytes.length);
    for (int i = start; i < bytes.length; i++) {
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

  private static boolean isDrivePath(String path) {
    return path.length() >= 3
        && isLetter((byte) path.charAt(0))
        && path.charAt(1) == ':'
        && path.charAt(2) == '/';
  }

  private static boolean isLetter(byte b) {
    return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z');
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
