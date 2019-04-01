// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote;

import com.google.auth.Credentials;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.remote.blobstore.CombinedDiskHttpBlobStore;
import com.google.devtools.build.lib.remote.blobstore.ConcurrentMapBlobStore;
import com.google.devtools.build.lib.remote.blobstore.OnDiskBlobStore;
import com.google.devtools.build.lib.remote.blobstore.SimpleBlobStore;
import com.google.devtools.build.lib.remote.blobstore.http.HttpBlobStore;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import io.netty.channel.unix.DomainSocketAddress;
import java.io.IOException;
import java.net.URI;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;

/**
 * A factory class for providing a {@link SimpleBlobStore} to be used with {@link
 * SimpleBlobStoreActionCache}. Currently implemented with HTTP or local.
 */
public final class SimpleBlobStoreFactory {

  private SimpleBlobStoreFactory() {}

  public static SimpleBlobStore create(RemoteOptions remoteOptions, @Nullable Path casPath) {
    if (isHttpUrlOptions(remoteOptions)) {
      return createHttp(remoteOptions, /* creds= */ null);
    } else if (casPath != null) {
      return new OnDiskBlobStore(casPath);
    } else {
      return new ConcurrentMapBlobStore(new ConcurrentHashMap<>());
    }
  }

  public static SimpleBlobStore create(
      RemoteOptions options, @Nullable Credentials creds, Path workingDirectory)
      throws IOException {

    Preconditions.checkNotNull(workingDirectory, "workingDirectory");
    if (isHttpUrlOptions(options) && isDiskCache(options)) {
      return createCombinedCache(workingDirectory, options.diskCache, options, creds);
    }
    if (isHttpUrlOptions(options)) {
      return createHttp(options, creds);
    }
    if (isDiskCache(options)) {
      return createDiskCache(workingDirectory, options.diskCache);
    }
    throw new IllegalArgumentException(
        "Unrecognized RemoteOptions configuration: remote Http cache URL and/or local disk cache"
            + " options expected.");
  }

  public static boolean isRemoteCacheOptions(RemoteOptions options) {
    return isHttpUrlOptions(options) || isDiskCache(options);
  }

  private static SimpleBlobStore createHttp(RemoteOptions options, Credentials creds) {
    Preconditions.checkNotNull(options.remoteCache, "remoteCache");
    Preconditions.checkArgument(options.remoteCache.toLowerCase().startsWith("http"), "remoteCache should start with http");
    try {
      URI uri = URI.create(options.remoteCache);

      if (options.remoteCacheProxy != null) {
        if (options.remoteCacheProxy.startsWith("unix:")) {
          return HttpBlobStore.create(
              new DomainSocketAddress(options.remoteCacheProxy.replaceFirst("^unix:", "")),
              uri,
              options.remoteTimeout,
              options.remoteMaxConnections,
              creds);
        } else {
          throw new Exception("Remote cache proxy unsupported: " + options.remoteCacheProxy);
        }
      } else {
        return HttpBlobStore.create(
            uri, options.remoteTimeout, options.remoteMaxConnections, creds);
      }
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  private static SimpleBlobStore createDiskCache(Path workingDirectory, PathFragment diskCachePath)
      throws IOException {
    Path cacheDir =
        workingDirectory.getRelative(Preconditions.checkNotNull(diskCachePath, "diskCachePath"));
    if (!cacheDir.exists()) {
      cacheDir.createDirectoryAndParents();
    }
    return new OnDiskBlobStore(cacheDir);
  }

  private static SimpleBlobStore createCombinedCache(
      Path workingDirectory, PathFragment diskCachePath, RemoteOptions options, Credentials cred)
      throws IOException {
    Path cacheDir =
        workingDirectory.getRelative(Preconditions.checkNotNull(diskCachePath, "diskCachePath"));
    if (!cacheDir.exists()) {
      cacheDir.createDirectoryAndParents();
    }
    return new CombinedDiskHttpBlobStore(cacheDir, createHttp(options, cred));
  }

  private static boolean isDiskCache(RemoteOptions options) {
    return options.diskCache != null && !options.diskCache.isEmpty();
  }

  private static boolean isHttpUrlOptions(RemoteOptions options) {
    return options.remoteCache != null
        && options.remoteCache.toLowerCase().startsWith("http");
  }
}
