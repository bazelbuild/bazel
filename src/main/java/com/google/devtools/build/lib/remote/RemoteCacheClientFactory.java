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
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.disk.DiskAndRemoteCacheClient;
import com.google.devtools.build.lib.remote.disk.DiskCacheClient;
import com.google.devtools.build.lib.remote.http.HttpCacheClient;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import io.netty.channel.unix.DomainSocketAddress;
import java.io.IOException;
import java.net.URI;
import java.util.concurrent.ExecutorService;
import javax.annotation.Nullable;

/**
 * A factory class for providing a {@link RemoteCacheClient}. Currently implemented for HTTP and
 * disk caching.
 */
public final class RemoteCacheClientFactory {

  private RemoteCacheClientFactory() {}

  public static RemoteCacheClient createDiskAndRemoteClient(
      Path workingDirectory,
      PathFragment diskCachePath,
      DigestUtil digestUtil,
      ExecutorService executorService,
      boolean remoteVerifyDownloads,
      RemoteCacheClient remoteCacheClient)
      throws IOException {
    DiskCacheClient diskCacheClient =
        createDiskCache(
            workingDirectory, diskCachePath, digestUtil, executorService, remoteVerifyDownloads);
    return new DiskAndRemoteCacheClient(diskCacheClient, remoteCacheClient);
  }

  public static RemoteCacheClient create(
      RemoteOptions options,
      @Nullable Credentials creds,
      AuthAndTLSOptions authAndTlsOptions,
      Path workingDirectory,
      DigestUtil digestUtil,
      ExecutorService executorService,
      RemoteRetrier retrier)
      throws IOException {
    Preconditions.checkNotNull(workingDirectory, "workingDirectory");
    if (isHttpCache(options) && isDiskCache(options)) {
      return createDiskAndHttpCache(
          workingDirectory,
          options.diskCache,
          options,
          creds,
          authAndTlsOptions,
          digestUtil,
          executorService,
          retrier);
    }
    if (isHttpCache(options)) {
      return createHttp(options, creds, authAndTlsOptions, digestUtil, retrier);
    }
    if (isDiskCache(options)) {
      return createDiskCache(
          workingDirectory,
          options.diskCache,
          digestUtil,
          executorService,
          options.remoteVerifyDownloads);
    }
    throw new IllegalArgumentException(
        "Unrecognized RemoteOptions configuration: remote Http cache URL and/or local disk cache"
            + " options expected.");
  }

  public static boolean isRemoteCacheOptions(RemoteOptions options) {
    return isHttpCache(options) || isDiskCache(options);
  }

  private static RemoteCacheClient createHttp(
      RemoteOptions options,
      Credentials creds,
      AuthAndTLSOptions authAndTlsOptions,
      DigestUtil digestUtil,
      RemoteRetrier retrier) {
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
              Math.toIntExact(options.remoteTimeout.getSeconds()),
              options.remoteMaxConnections,
              options.remoteVerifyDownloads,
              ImmutableList.copyOf(options.remoteHeaders),
              digestUtil,
              retrier,
              creds,
              authAndTlsOptions);
        } else {
          throw new Exception("Remote cache proxy unsupported: " + options.remoteProxy);
        }
      } else {
        return HttpCacheClient.create(
            uri,
            Math.toIntExact(options.remoteTimeout.getSeconds()),
            options.remoteMaxConnections,
            options.remoteVerifyDownloads,
            ImmutableList.copyOf(options.remoteHeaders),
            digestUtil,
            retrier,
            creds,
            authAndTlsOptions);
      }
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  private static DiskCacheClient createDiskCache(
      Path workingDirectory,
      PathFragment diskCachePath,
      DigestUtil digestUtil,
      ExecutorService executorService,
      boolean verifyDownloads)
      throws IOException {
    Path cacheDir =
        workingDirectory.getRelative(Preconditions.checkNotNull(diskCachePath, "diskCachePath"));
    return new DiskCacheClient(cacheDir, digestUtil, executorService, verifyDownloads);
  }

  private static RemoteCacheClient createDiskAndHttpCache(
      Path workingDirectory,
      PathFragment diskCachePath,
      RemoteOptions options,
      Credentials cred,
      AuthAndTLSOptions authAndTlsOptions,
      DigestUtil digestUtil,
      ExecutorService executorService,
      RemoteRetrier retrier)
      throws IOException {
    RemoteCacheClient httpCache = createHttp(options, cred, authAndTlsOptions, digestUtil, retrier);
    return createDiskAndRemoteClient(
        workingDirectory,
        diskCachePath,
        digestUtil,
        executorService,
        options.remoteVerifyDownloads,
        httpCache);
  }

  public static boolean isDiskCache(RemoteOptions options) {
    return options.diskCache != null && !options.diskCache.isEmpty();
  }

  public static boolean isHttpCache(RemoteOptions options) {
    return options.remoteCache != null
        && (Ascii.toLowerCase(options.remoteCache).startsWith("http://")
            || Ascii.toLowerCase(options.remoteCache).startsWith("https://"));
  }
}
