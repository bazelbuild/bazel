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

package com.google.devtools.build.lib.remote.common;

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Digest;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.OutputStream;

/**
 * An interface for storing BLOBs each one indexed by a string (hash in hexadecimal).
 *
 * <p>Implementations must be thread-safe.
 */
public interface SimpleBlobStore {

  /**
   * A special type of Digest that is used only as a remote action cache key. This is a separate
   * type in order to prevent accidentally using other Digests as action keys.
   */
  final class ActionKey {
    private final Digest digest;

    public Digest getDigest() {
      return digest;
    }

    public ActionKey(Digest digest) {
      this.digest = digest;
    }
  }

  /**
   * Fetches the BLOB associated with the {@code key} from the CAS and writes it to {@code out}.
   *
   * <p>The caller is responsible to close {@code out}.
   *
   * @return {@code true} if the {@code key} was found. {@code false} otherwise.
   */
  ListenableFuture<Boolean> get(String key, OutputStream out);

  /**
   * Fetches the BLOB associated with the {@code key} from the Action Cache and writes it to {@code
   * out}.
   *
   * <p>The caller is responsible to close {@code out}.
   *
   * @return {@code true} if the {@code key} was found. {@code false} otherwise.
   */
  ListenableFuture<Boolean> getActionResult(String actionKey, OutputStream out);

  /** Uploads an {@link ActionResult} keyed by the action hash to the action cache. */
  void putActionResult(ActionKey actionDigest, ActionResult actionResult)
      throws IOException, InterruptedException;

  /** Close resources associated with the blob store. */
  void close();

  /**
   * Uploads a file.
   *
   * @param digest the digest of the file.
   * @param file the file to upload.
   */
  ListenableFuture<Void> uploadFile(Digest digest, Path file);

  /**
   * Uploads a BLOB.
   *
   * @param digest the digest of the blob.
   * @param data the blob to upload.
   */
  ListenableFuture<Void> uploadBlob(Digest digest, ByteString data);
}
