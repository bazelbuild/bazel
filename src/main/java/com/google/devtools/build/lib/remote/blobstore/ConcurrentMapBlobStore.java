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

import com.google.common.base.Preconditions;
import com.google.common.io.ByteStreams;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.concurrent.ConcurrentMap;

/** A {@link SimpleBlobStore} implementation using a {@link ConcurrentMap}. */
public final class ConcurrentMapBlobStore implements SimpleBlobStore {
  private final ConcurrentMap<String, byte[]> map;
  static final String ACTION_KEY_PREFIX = "ac_";

  public ConcurrentMapBlobStore(ConcurrentMap<String, byte[]> map) {
    this.map = map;
  }

  @Override
  public boolean contains(String key) {
    return map.containsKey(key);
  }

  @Override
  public boolean containsActionResult(String key) {
    return map.containsKey(ACTION_KEY_PREFIX + key);
  }

  @Override
  public ListenableFuture<Boolean> get(String key, OutputStream out) {
    byte[] data = map.get(key);
    SettableFuture<Boolean> f = SettableFuture.create();
    if (data == null) {
      f.set(false);
    } else {
      try {
        out.write(data);
        f.set(true);
      } catch (IOException e) {
        f.setException(e);
      }
    }
    return f;
  }

  @Override
  public ListenableFuture<Boolean> getActionResult(String key, OutputStream out) {
    return get(ACTION_KEY_PREFIX + key, out);
  }

  @Override
  public void put(String key, long length, InputStream in) throws IOException {
    byte[] value = ByteStreams.toByteArray(in);
    Preconditions.checkState(value.length == length);
    map.put(key, value);
  }

  @Override
  public void putActionResult(String key, byte[] in) {
    map.put(ACTION_KEY_PREFIX + key, in);
  }

  @Override
  public void close() {}

}
