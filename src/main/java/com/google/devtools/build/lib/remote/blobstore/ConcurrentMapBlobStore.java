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
  public byte[] get(String key) {
    return map.get(key);
  }

  @Override
  public void put(String key, byte[] value) {
    map.put(key, value);
  }

  @Override
  public void close() {}
}