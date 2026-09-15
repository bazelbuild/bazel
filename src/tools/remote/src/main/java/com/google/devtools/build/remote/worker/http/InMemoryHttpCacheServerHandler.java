// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.remote.worker.http;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import io.netty.channel.ChannelHandler.Sharable;
import java.util.concurrent.ConcurrentMap;
import javax.annotation.Nullable;

/** A simple HTTP REST in-memory cache used during testing the LRE. */
@Sharable
public class InMemoryHttpCacheServerHandler extends AbstractHttpCacheServerHandler {

  private final ConcurrentMap<String, byte[]> cache;

  @VisibleForTesting
  public InMemoryHttpCacheServerHandler(ConcurrentMap<String, byte[]> cache) {
    this.cache = Preconditions.checkNotNull(cache);
  }

  @Nullable
  @Override
  protected byte[] readFromCache(String uri) {
    return cache.get(uri);
  }

  @Override
  protected void writeToCache(String uri, byte[] content) {
    cache.putIfAbsent(uri, content);
  }
}
