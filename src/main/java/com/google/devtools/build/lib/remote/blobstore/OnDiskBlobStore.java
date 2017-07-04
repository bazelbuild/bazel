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

import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.io.OutputStream;
import java.util.UUID;

/** A on-disk store for the remote action cache. */
public final class OnDiskBlobStore implements SimpleBlobStore {
  private final Path root;

  public OnDiskBlobStore(Path root) {
    this.root = root;
  }

  @Override
  public boolean containsKey(String key) {
    return toPath(key).exists();
  }

  @Override
  public byte[] get(String key) throws IOException {
    Path f = toPath(key);
    return f.exists() ? FileSystemUtils.readContent(f) : null;
  }

  @Override
  public void put(String key, byte[] value) throws IOException {
    Path temp = toPath(UUID.randomUUID().toString());
    try (OutputStream out = temp.getOutputStream()) {
      out.write(value);
    }
    Path f = toPath(key);
    temp.renameTo(f);
  }

  @Override
  public void close() {}

  private Path toPath(String key) {
    return root.getChild(key);
  }
}