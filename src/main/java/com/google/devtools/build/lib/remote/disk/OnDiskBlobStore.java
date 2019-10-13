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
import com.google.devtools.build.lib.remote.common.SimpleBlobStore;
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
  public boolean contains(String key) {
    return toPath(key, /* actionResult= */ false).exists();
  }

  /** Returns {@code true} if the provided {@code key} is stored in the Action Cache. */
  public boolean containsActionResult(String key) {
    return toPath(key, /* actionResult= */ true).exists();
  }

  @Override
  public ListenableFuture<Boolean> get(String key, OutputStream out) {
    SettableFuture<Boolean> f = SettableFuture.create();
    Path p = toPath(key, /* actionResult= */ false);
    if (!p.exists()) {
      f.set(false);
    } else {
      try (InputStream in = p.getInputStream()) {
        ByteStreams.copy(in, out);
        f.set(true);
      } catch (IOException e) {
        f.setException(e);
      }
    }
    return f;
  }

  @Override
  public ListenableFuture<Boolean> getActionResult(String key, OutputStream out) {
    return get(getDiskKey(key, /* actionResult= */ true), out);
  }

  @Override
  public void putActionResult(ActionKey actionKey, ActionResult actionResult) throws IOException {
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
