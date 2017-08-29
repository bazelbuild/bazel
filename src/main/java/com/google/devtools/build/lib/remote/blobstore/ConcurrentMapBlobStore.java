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

import com.google.common.io.ByteStreams;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.concurrent.ConcurrentMap;

/** A {@link SimpleBlobStore} implementation using a {@link ConcurrentMap}. */
public final class ConcurrentMapBlobStore implements SimpleBlobStore {
  private final ConcurrentMap<String, byte[]> map;

  public ConcurrentMapBlobStore(ConcurrentMap<String, byte[]> map) {
    this.map = map;
  }

  @Override
  public boolean containsKey(String key) {
    return map.containsKey(key);
  }

  @Override
  public boolean get(String key, OutputStream out) throws IOException {
    byte[] data = map.get(key);
    if (data == null) {
      return false;
    }
    out.write(data);
    return true;
  }

  @Override
  public boolean getActionResult(String key, OutputStream out)
      throws IOException, InterruptedException {
    return get(key, out);
  }

  @Override
  public void put(String key, InputStream in) throws IOException {
    byte[] value = ByteStreams.toByteArray(in);
    map.put(key, value);
  }

  @Override
  public void putActionResult(String key, InputStream in)
      throws IOException, InterruptedException {
    put(key, in);
  }

  @Override
  public void close() {}
}