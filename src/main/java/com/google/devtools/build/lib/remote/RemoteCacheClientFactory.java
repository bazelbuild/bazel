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
import com.google.common.base.Ascii;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.disk.CombinedDiskHttpCacheClient;
import com.google.devtools.build.lib.remote.disk.DiskCacheClient;
import com.google.devtools.build.lib.remote.http.HttpCacheClient;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import io.netty.channel.unix.DomainSocketAddress;
import java.io.IOException;
import java.net.URI;
import javax.annotation.Nullable;

/**
 * A factory class for providing a {@link RemoteCacheClient}. Currently implemented for HTTP and
 * disk caching.
 */
public final class RemoteCacheClientFactory {

  private RemoteCacheClientFactory() {}

  public static RemoteCacheClient create(RemoteOptions options, @Nullable Credentials creds,
      Path workingDirectory, DigestUtil digestUtil) throws IOException {
    Preconditions.checkNotNull(workingDirectory, "workingDirectory");
    if (isHttpUrlOptions(options) && isDiskCache(options)) {
      return createCombinedCache(workingDirectory, options.diskCache, options, creds, digestUtil);
    }
    if (isHttpUrlOptions(options)) {
      return createHttp(options, creds, digestUtil);
    }
    if (isDiskCache(options)) {
      return createDiskCache(workingDirectory, options.diskCache, options.remoteVerifyDownloads,
          digestUtil);
    }
    throw new IllegalArgumentException(
        "Unrecognized RemoteOptions configuration: remote Http cache URL and/or local disk cache"
            + " options expected.");
  }

  public static boolean isRemoteCacheOptions(RemoteOptions options) {
    return isHttpUrlOptions(options) || isDiskCache(options);
  }

  private static RemoteCacheClient createHttp(RemoteOptions options, Credentials creds,
      DigestUtil digestUtil) {
    Preconditions.checkNotNull(options.remoteCache, "remoteCache");

    try {
      URI uri = URI.create(options.remoteCache);
      Preconditions.checkArgument(
          Ascii.toLowerCase(uri.getScheme()).startsWith("http"),
          "remoteCache should start with http");

      if (options.remoteProxy != null) {
        if (options.remoteProxy.startsWith("unix:")) {
          return HttpCacheClient.create(
              new DomainSocketAddress(options.remoteProxy.replaceFirst("^unix:", "")),
              uri,
              options.remoteTimeout,
              options.remoteMaxConnections,
              options.remoteVerifyDownloads,
              ImmutableList.copyOf(options.remoteHeaders),
              digestUtil,
              creds);
        } else {
          throw new Exception("Remote cache proxy unsupported: " + options.remoteProxy);
        }
      } else {
        return HttpCacheClient.create(
            uri,
            options.remoteTimeout,
            options.remoteMaxConnections,
            options.remoteVerifyDownloads,
            ImmutableList.copyOf(options.remoteHeaders),
            digestUtil,
            creds);
      }
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  private static RemoteCacheClient createDiskCache(Path workingDirectory, PathFragment diskCachePath,
      boolean verifyDownloads, DigestUtil digestUtil)
      throws IOException {
    Path cacheDir =
        workingDirectory.getRelative(Preconditions.checkNotNull(diskCachePath, "diskCachePath"));
    if (!cacheDir.exists()) {
      cacheDir.createDirectoryAndParents();
    }
    return new DiskCacheClient(cacheDir, verifyDownloads, digestUtil);
  }

  private static RemoteCacheClient createCombinedCache(
      Path workingDirectory, PathFragment diskCachePath, RemoteOptions options, Credentials cred,
      DigestUtil digestUtil) throws IOException {
    Path cacheDir =
        workingDirectory.getRelative(Preconditions.checkNotNull(diskCachePath, "diskCachePath"));
    if (!cacheDir.exists()) {
      cacheDir.createDirectoryAndParents();
    }

    DiskCacheClient diskCache =
        new DiskCacheClient(cacheDir, options.remoteVerifyDownloads, digestUtil);
    RemoteCacheClient httpCache = createHttp(options, cred, digestUtil);

    return new CombinedDiskHttpCacheClient(diskCache, httpCache);
  }

  private static boolean isDiskCache(RemoteOptions options) {
    return options.diskCache != null && !options.diskCache.isEmpty();
  }

  private static boolean isHttpUrlOptions(RemoteOptions options) {
    return options.remoteCache != null
        && (Ascii.toLowerCase(options.remoteCache).startsWith("http://")
            || Ascii.toLowerCase(options.remoteCache).startsWith("https://"));
  }
}
