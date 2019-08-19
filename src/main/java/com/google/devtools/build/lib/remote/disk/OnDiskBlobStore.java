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
package com.google.devtools.build.lib.remote.disk;

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Digest;
import com.google.common.io.ByteStreams;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.SimpleBlobStore;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.UUID;

/** A on-disk store for the remote action cache. */
public class OnDiskBlobStore implements SimpleBlobStore {
  private final Path root;
  private static final String ACTION_KEY_PREFIX = "ac_";

  public OnDiskBlobStore(Path root) {
    this.root = root;
  }

  /** Returns {@code true} if the provided {@code key} is stored in the CAS. */
  public boolean contains(Digest digest) {
    return toPath(digest.getHash(), /* actionResult= */ false).exists();
  }

  /** Returns {@code true} if the provided {@code key} is stored in the Action Cache. */
  public boolean containsActionResult(ActionKey actionKey) {
    return toPath(actionKey.getDigest().getHash(), /* isActionCache= */ true).exists();
  }

  public void captureFile(Path src, Digest digest, boolean isActionCache) throws IOException {
    Path target = toPath(digest.getHash(), isActionCache);
    src.renameTo(target);
  }

  private ListenableFuture<Void> download(Digest digest, OutputStream out, boolean isActionCache) {
    SettableFuture<Void> f = SettableFuture.create();
    Path p = toPath(digest.getHash(), isActionCache);
    if (!p.exists()) {
      return Futures.immediateFailedFuture(new CacheNotFoundException(digest));
    } else {
      try (InputStream in = p.getInputStream()) {
        ByteStreams.copy(in, out);
        return Futures.immediateFuture(null);
      } catch (IOException e) {
        return Futures.immediateFailedFuture(e);
      }
    }
  }

  @Override
  public ListenableFuture<Void> downloadBlob(Digest digest, OutputStream out) {
    return download(digest, out, /* isActionCache= */ false);
  }

  @Override
  public ListenableFuture<ActionResult> downloadActionResult(ActionKey actionKey) {
    return Utils.downloadAsActionResult(actionKey,
        (digest, out) -> download(digest, out, /* isActionCache= */ true));
  }

  @Override
  public void uploadActionResult(ActionKey actionKey, ActionResult actionResult) throws IOException {
    try (InputStream data = actionResult.toByteString().newInput()) {
      saveFile(getDiskKey(actionKey.getDigest().getHash(), /* actionResult= */ true), data);
    }
  }

  @Override
  public void close() {}

  @Override
  public ListenableFuture<Void> uploadFile(Digest digest, Path file) {
    try (InputStream in = file.getInputStream()) {
      saveFile(digest.getHash(), in);
    } catch (IOException e) {
      return Futures.immediateFailedFuture(e);
    }
    return Futures.immediateFuture(null);
  }

  @Override
  public ListenableFuture<Void> uploadBlob(Digest digest, ByteString data) {
    try (InputStream in = data.newInput()) {
      saveFile(digest.getHash(), in);
    } catch (IOException e) {
      return Futures.immediateFailedFuture(e);
    }
    return Futures.immediateFuture(null);
  }

  protected Path toPath(String key, boolean actionResult) {
    return root.getChild(getDiskKey(key, actionResult));
  }

  private static String getDiskKey(String key, boolean actionResult) {
    return actionResult ? ACTION_KEY_PREFIX + key : key;
  }

  private void saveFile(String key, InputStream in) throws IOException {
    Path target = toPath(key, /* actionResult= */ false);
    if (target.exists()) {
      return;
    }

    // Write a temporary file first, and then rename, to avoid data corruption in case of a crash.
    Path temp = toPath(UUID.randomUUID().toString(), /* actionResult= */ false);
    try (OutputStream out = temp.getOutputStream()) {
      ByteStreams.copy(in, out);
    }
    // TODO(ulfjack): Fsync temp here before we rename it to avoid data loss in the case of machine
    // crashes (the OS may reorder the writes and the rename).
    temp.renameTo(target);
  }
}
