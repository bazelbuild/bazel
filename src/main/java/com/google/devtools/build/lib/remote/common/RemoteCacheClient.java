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

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.CacheCapabilities;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.DigestFunction;
import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.OutputStream;
import javax.annotation.Nullable;

/**
 * An interface for a remote caching protocol.
 *
 * <p>Implementations must be thread-safe.
 */
public interface RemoteCacheClient extends MissingDigestsFinder {
  CacheCapabilities getCacheCapabilities() throws IOException;

  ListenableFuture<String> getAuthority();

  /**
   * A key in the remote action cache. The type wraps around a {@link Digest} of an {@link Action}.
   * Action keys are special in that they aren't content-addressable but refer to action results.
   *
   * <p>Terminology note: "action" is used here in the remote execution protocol sense, which is
   * equivalent to a Bazel "spawn" (a Bazel "action" being a higher-level concept).
   */
  final class ActionKey {

    private final Digest digest;

    public Digest getDigest() {
      return digest;
    }

    public ActionKey(Digest digest) {
      this.digest = Preconditions.checkNotNull(digest, "digest");
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof ActionKey otherKey)) {
        return false;
      }

      return digest.equals(otherKey.digest);
    }

    @Override
    public int hashCode() {
      return digest.hashCode();
    }
  }

  /**
   * Class to keep track of which cache (disk or remote) a given [cached] ActionResult comes from.
   */
  @AutoValue
  abstract class CachedActionResult {
    @Nullable
    public static CachedActionResult remote(ActionResult actionResult) {
      if (actionResult == null) {
        return null;
      }
      return new AutoValue_RemoteCacheClient_CachedActionResult(actionResult, "remote");
    }

    @Nullable
    public static CachedActionResult disk(ActionResult actionResult) {
      if (actionResult == null) {
        return null;
      }
      return new AutoValue_RemoteCacheClient_CachedActionResult(actionResult, "disk");
    }

    /** A actionResult can have a cache name ascribed to it. */
    public abstract ActionResult actionResult();

    /** Indicates which cache the {@link #actionResult} came from (disk/remote) */
    public abstract String cacheName();
  }

  /**
   * Downloads an action result for the {@code actionKey}.
   *
   * @param context the context for the action.
   * @param actionKey The digest of the {@link Action} that generated the action result.
   * @param inlineOutErr A hint to the server to inline the stdout and stderr in the {@code
   *     ActionResult} message.
   * @return A Future representing pending download of an action result. If an action result for
   *     {@code actionKey} cannot be found the result of the Future is {@code null}.
   */
  ListenableFuture<CachedActionResult> downloadActionResult(
      RemoteActionExecutionContext context, ActionKey actionKey, boolean inlineOutErr);

  /**
   * Uploads an action result for the {@code actionKey}.
   *
   * @param context the context for the action.
   * @param actionKey The digest of the {@link Action} that generated the action result.
   * @param actionResult The action result to associate with the {@code actionKey}.
   * @return A Future representing pending completion of the upload.
   */
  ListenableFuture<Void> uploadActionResult(
      RemoteActionExecutionContext context, ActionKey actionKey, ActionResult actionResult);

  /**
   * Downloads a BLOB for the given {@code digest} and writes it to {@code out}.
   *
   * <p>It's the callers responsibility to close {@code out}.
   *
   * @param context the context for the action.
   * @return A Future representing pending completion of the download. If a BLOB for {@code digest}
   *     does not exist in the cache the Future fails with a {@link CacheNotFoundException}.
   */
  ListenableFuture<Void> downloadBlob(
      RemoteActionExecutionContext context,
      Digest digest,
      @Nullable DigestFunction.Value digestFunction,
      OutputStream out);

  /**
   * Uploads a {@code file} to the CAS.
   *
   * @param context the context for the action.
   * @param digest The digest of the file.
   * @param file The file to upload.
   * @return A future representing pending completion of the upload.
   */
  ListenableFuture<Void> uploadFile(RemoteActionExecutionContext context, Digest digest, Path file);

  /**
   * Uploads a BLOB to the CAS.
   *
   * @param context the context for the action.
   * @param digest The digest of the blob.
   * @param data The BLOB to upload.
   * @return A future representing pending completion of the upload.
   */
  ListenableFuture<Void> uploadBlob(
      RemoteActionExecutionContext context, Digest digest, ByteString data);

  /** Close resources associated with the remote cache. */
  void close();
}
