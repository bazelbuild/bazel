// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.remote.worker;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.remote.Store;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.remote.worker.http.AbstractHttpCacheServerHandler;
import io.netty.channel.ChannelHandler.Sharable;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import javax.annotation.Nullable;

/** A simple HTTP REST disk cache used during test. */
@Sharable
public class OnDiskHttpCacheServerHandler extends AbstractHttpCacheServerHandler {
  private final OnDiskBlobStoreCache cache;

  public OnDiskHttpCacheServerHandler(OnDiskBlobStoreCache cache) {
    this.cache = cache;
  }

  @Nullable
  @Override
  protected byte[] readFromCache(String uri) throws IOException {
    var diskCache = cache.getDiskCacheClient();
    Path path;
    if (uri.startsWith("/ac/")) {
      path = diskCache.toPath(uri.substring("/ac/".length()), Store.AC);
    } else if (uri.startsWith("/cas/")) {
      path = diskCache.toPath(uri.substring("/cas/".length()), Store.CAS);
    } else {
      throw new IOException("Invalid uri: " + uri);
    }

    try (var out = new ByteArrayOutputStream();
        var in = path.getInputStream()) {
      ByteStreams.copy(in, out);
      return out.toByteArray();
    }
  }

  @Override
  protected void writeToCache(String uri, byte[] content) throws IOException {
    var diskCache = cache.getDiskCacheClient();
    Digest digest;
    Store store;
    if (uri.startsWith("/ac/")) {
      digest = DigestUtil.buildDigest(uri.substring(4), content.length);
      store = Store.AC;
    } else if (uri.startsWith("/cas/")) {
      digest = DigestUtil.buildDigest(uri.substring(5), content.length);
      store = Store.CAS;
    } else {
      throw new IOException("Invalid uri: " + uri);
    }

    diskCache.saveFile(digest, store, new ByteArrayInputStream(content));
  }
}
