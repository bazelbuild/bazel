// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.util;

import static java.nio.charset.StandardCharsets.UTF_8;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.DigestFunction;
import com.google.common.hash.HashCode;
import com.google.common.hash.HashingOutputStream;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.cache.DigestUtils;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.Message;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;

/** Utility methods to work with {@link Digest}. */
public class DigestUtil {

  /**
   * A special type of Digest that is used only as a remote action cache key. This is a separate
   * type in order to prevent accidentally using other Digests as action keys.
   */
  public static final class ActionKey {
    private final Digest digest;

    public Digest getDigest() {
      return digest;
    }

    public ActionKey(Digest digest) {
      this.digest = digest;
    }
  }

  private final DigestHashFunction hashFn;

  public DigestUtil(DigestHashFunction hashFn) {
    this.hashFn = hashFn;
  }

  /** Returns the currently used digest function. */
  public DigestFunction getDigestFunction() {
    for (String name : hashFn.getNames()) {
      try {
        return DigestFunction.valueOf(name);
      } catch (IllegalArgumentException e) {
        // continue.
      }
    }
    return DigestFunction.UNKNOWN;
  }

  public Digest compute(byte[] blob) {
    return buildDigest(hashFn.getHashFunction().hashBytes(blob).toString(), blob.length);
  }

  public Digest compute(Path file) throws IOException {
    return compute(file, file.getFileSize());
  }

  public Digest compute(Path file, long fileSize) throws IOException {
    return buildDigest(DigestUtils.getDigestOrFail(file, fileSize), fileSize);
  }

  public Digest compute(VirtualActionInput input) throws IOException {
    ByteArrayOutputStream buffer = new ByteArrayOutputStream();
    input.writeTo(buffer);
    return compute(buffer.toByteArray());
  }

  /**
   * Computes a digest of the given proto message. Currently, we simply rely on message output as
   * bytes, but this implementation relies on the stability of the proto encoding, in particular
   * between different platforms and languages. TODO(olaola): upgrade to a better implementation!
   */
  public Digest compute(Message message) {
    return compute(message.toByteArray());
  }

  public Digest computeAsUtf8(String str) {
    return compute(str.getBytes(UTF_8));
  }

  public DigestUtil.ActionKey computeActionKey(Action action) {
    return new DigestUtil.ActionKey(compute(action));
  }

  /**
   * Assumes that the given Digest is a valid digest of an Action, and creates an ActionKey wrapper.
   * This should not be called on the client side!
   */
  public DigestUtil.ActionKey asActionKey(Digest digest) {
    return new DigestUtil.ActionKey(digest);
  }

  /** Returns the hash of {@code data} in binary. */
  public byte[] hash(byte[] data) {
    return hashFn.getHashFunction().hashBytes(data).asBytes();
  }

  public static Digest buildDigest(byte[] hash, long size) {
    return buildDigest(HashCode.fromBytes(hash).toString(), size);
  }

  public static Digest buildDigest(String hexHash, long size) {
    return Digest.newBuilder().setHash(hexHash).setSizeBytes(size).build();
  }

  public static String hashCodeToString(HashCode hash) {
    return BaseEncoding.base16().lowerCase().encode(hash.asBytes());
  }

  public HashingOutputStream newHashingOutputStream(OutputStream out) {
    return new HashingOutputStream(hashFn.getHashFunction(), out);
  }

  public String toString(Digest digest) {
    return digest.getHash() + "/" + digest.getSizeBytes();
  }

}
