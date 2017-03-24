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

import com.google.common.hash.HashCode;
import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.remote.RemoteProtocol.Action;
import com.google.devtools.build.lib.remote.RemoteProtocol.ContentDigest;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import com.google.protobuf.Message;
import java.io.IOException;

/** Helper methods relating to computing ContentDigest messages for remote execution. */
@ThreadSafe
public final class ContentDigests {
  private ContentDigests() {}

  public static ContentDigest computeDigest(byte[] blob) {
    return buildDigest(Hashing.sha1().hashBytes(blob).asBytes(), blob.length);
  }

  // TODO(olaola): cache these in ActionInputFileCache!
  public static ContentDigest computeDigest(Path file) throws IOException {
    return buildDigest(file.getSHA1Digest(), file.getFileSize());
  }

  /**
   * Computes a digest of the given proto message. Currently, we simply rely on message output as
   * bytes, but this implementation relies on the stability of the proto encoding, in particular
   * between different platforms and languages. TODO(olaola): upgrade to a better implementation!
   */
  public static ContentDigest computeDigest(Message message) {
    return computeDigest(message.toByteArray());
  }

  /**
   * A special type of ContentDigest that is used only as a remote action cache key. This is a
   * separate type in order to prevent accidentally using other ContentDigests as action keys.
   */
  public static final class ActionKey {
    private final ContentDigest digest;

    public ContentDigest getDigest() {
      return digest;
    }

    private ActionKey(ContentDigest digest) {
      this.digest = digest;
    }
  }

  public static ActionKey computeActionKey(Action action) {
    return new ActionKey(computeDigest(action));
  }

  /**
   * Assumes that the given ContentDigest is a valid digest of an Action, and creates an ActionKey
   * wrapper. This should not be called on the client side!
   */
  public static ActionKey unsafeActionKeyFromDigest(ContentDigest digest) {
    return new ActionKey(digest);
  }

  public static ContentDigest buildDigest(byte[] digest, long size) {
    ContentDigest.Builder b = ContentDigest.newBuilder();
    b.setDigest(ByteString.copyFrom(digest)).setSizeBytes(size);
    return b.build();
  }

  public static ContentDigest getDigestFromInputCache(ActionInput input, ActionInputFileCache cache)
      throws IOException {
    return buildDigest(cache.getDigest(input), cache.getSizeInBytes(input));
  }

  public static String toHexString(ContentDigest digest) {
    return HashCode.fromBytes(digest.getDigest().toByteArray()).toString();
  }

  public static String toString(ContentDigest digest) {
    return "<digest: " + toHexString(digest) + ", size: " + digest.getSizeBytes() + " bytes>";
  }
}
