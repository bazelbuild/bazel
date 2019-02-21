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
package com.google.devtools.build.lib.remote.blobstore;

import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;

import com.google.common.io.ByteStreams;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.vfs.Path;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.UUID;

/** A on-disk store for the remote action cache. */
public class OnDiskBlobStore implements SimpleBlobStore {
  private final Path root;
  static final String ACTION_KEY_PREFIX = "ac_";

  public OnDiskBlobStore(Path root) {
    this.root = root;
  }

  @Override
  public boolean containsKey(String key) {
    return toPath(key).exists();
  }

  @Override
  public ListenableFuture<Boolean> get(String key, OutputStream out) {
    SettableFuture<Boolean> f = SettableFuture.create();
    Path p = toPath(key);
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
  public boolean getActionResult(String key, OutputStream out)
      throws IOException, InterruptedException {
    return getFromFuture(get(ACTION_KEY_PREFIX + key, out));
  }

  @Override
  public void put(String key, long length, InputStream in) throws IOException, InterruptedException {
    Path target = toPath(key);
    if (target.exists()) {
      return;
    }

    // Write a temporary file first, and then rename, to avoid data corruption in case of a crash.
    Path temp = toPath(UUID.randomUUID().toString());
    try (OutputStream out = temp.getOutputStream()) {
      ByteStreams.copy(in, out);
    }
    // TODO(ulfjack): Fsync temp here before we rename it to avoid data loss in the case of machine
    // crashes (the OS may reorder the writes and the rename).
    temp.renameTo(target);
  }

  @Override
  public void putActionResult(String key, byte[] in) throws IOException, InterruptedException {
    put(ACTION_KEY_PREFIX + key, in.length, new ByteArrayInputStream(in));
  }

  @Override
  public void close() {}

  protected Path toPath(String key) {
    return root.getChild(key);
  }
}
