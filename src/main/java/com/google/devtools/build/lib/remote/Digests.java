// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.cache.DigestUtils;
import com.google.devtools.build.lib.actions.cache.Metadata;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.remoteexecution.v1test.Action;
import com.google.devtools.remoteexecution.v1test.Digest;
import com.google.protobuf.Message;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

/** Helper methods relating to computing Digest messages for remote execution. */
@ThreadSafe
public final class Digests {
  private Digests() {}

  public static Digest computeDigest(byte[] blob) {
    return buildDigest(
        FileSystem.getDigestFunction().getHash().hashBytes(blob).toString(), blob.length);
  }

  public static Digest computeDigest(Path file) throws IOException {
    long fileSize = file.getFileSize();
    byte[] digest = DigestUtils.getDigestOrFail(file, fileSize);
    return buildDigest(digest, fileSize);
  }

  public static Digest computeDigest(VirtualActionInput input) throws IOException {
    ByteArrayOutputStream buffer = new ByteArrayOutputStream();
    input.writeTo(buffer);
    return computeDigest(buffer.toByteArray());
  }

  /**
   * Computes a digest of the given proto message. Currently, we simply rely on message output as
   * bytes, but this implementation relies on the stability of the proto encoding, in particular
   * between different platforms and languages. TODO(olaola): upgrade to a better implementation!
   */
  public static Digest computeDigest(Message message) {
    return computeDigest(message.toByteArray());
  }

  public static Digest computeDigestUtf8(String str) {
    return computeDigest(str.getBytes(UTF_8));
  }

  /**
   * A special type of Digest that is used only as a remote action cache key. This is a
   * separate type in order to prevent accidentally using other Digests as action keys.
   */
  public static final class ActionKey {
    private final Digest digest;

    public Digest getDigest() {
      return digest;
    }

    private ActionKey(Digest digest) {
      this.digest = digest;
    }
  }

  public static ActionKey computeActionKey(Action action) {
    return new ActionKey(computeDigest(action));
  }

  /**
   * Assumes that the given Digest is a valid digest of an Action, and creates an ActionKey
   * wrapper. This should not be called on the client side!
   */
  public static ActionKey unsafeActionKeyFromDigest(Digest digest) {
    return new ActionKey(digest);
  }

  public static Digest buildDigest(byte[] hash, long size) {
    return buildDigest(HashCode.fromBytes(hash).toString(), size);
  }

  public static Digest buildDigest(String hexHash, long size) {
    return Digest.newBuilder().setHash(hexHash).setSizeBytes(size).build();
  }

  public static Digest getDigestFromInputCache(ActionInput input, MetadataProvider cache)
      throws IOException {
    Metadata metadata = cache.getMetadata(input);
    return buildDigest(metadata.getDigest(), metadata.getSize());
  }
}
