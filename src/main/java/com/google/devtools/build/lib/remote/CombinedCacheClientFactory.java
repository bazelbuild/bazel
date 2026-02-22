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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.disk.DiskCacheClient;
import com.google.devtools.build.lib.remote.http.HttpCacheClient;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.RemoteProxyHelper;
import com.google.devtools.build.lib.remote.util.RemoteProxyHelper.ProxyInfo;
import com.google.devtools.build.lib.vfs.Path;
import io.netty.channel.unix.DomainSocketAddress;
import java.io.IOException;
import java.net.URI;
import java.util.Map;
import javax.annotation.Nullable;

/** A factory class for providing a {@link CombinedCacheClient}. */
public final class CombinedCacheClientFactory {

  private CombinedCacheClientFactory() {}

  /**
   * A record holding a {@link DiskCacheClient} and {@link RemoteCacheClient} pair. Either may be
   * absent.
   */
  public record CombinedCacheClient(
      @Nullable RemoteCacheClient remoteCacheClient, @Nullable DiskCacheClient diskCacheClient) {}

  public static CombinedCacheClient create(
      RemoteOptions options,
      @Nullable Credentials creds,
      AuthAndTLSOptions authAndTlsOptions,
      Path workingDirectory,
      DigestUtil digestUtil,
      RemoteRetrier retrier)
      throws IOException {
    return create(
        options, creds, authAndTlsOptions, workingDirectory, digestUtil, retrier, ImmutableMap.of());
  }

  public static CombinedCacheClient create(
      RemoteOptions options,
      @Nullable Credentials creds,
      AuthAndTLSOptions authAndTlsOptions,
      Path workingDirectory,
      DigestUtil digestUtil,
      RemoteRetrier retrier,
      Map<String, String> clientEnv)
      throws IOException {
    Preconditions.checkNotNull(workingDirectory, "workingDirectory");
    RemoteCacheClient httpCacheClient = null;
    DiskCacheClient diskCacheClient = null;
    if (isHttpCache(options)) {
      httpCacheClient =
          createHttp(options, creds, authAndTlsOptions, digestUtil, retrier, clientEnv);
    }
    if (isDiskCache(options)) {
      diskCacheClient =
          createDiskCache(workingDirectory, options, digestUtil, options.remoteVerifyDownloads);
    }
    if (httpCacheClient == null && diskCacheClient == null) {
      throw new IllegalArgumentException(
          "Unrecognized RemoteOptions configuration: remote Http cache URL and/or local disk cache"
              + " options expected.");
    }
    return new CombinedCacheClient(httpCacheClient, diskCacheClient);
  }

  public static boolean isRemoteCacheOptions(RemoteOptions options) {
    return isHttpCache(options) || isDiskCache(options);
  }

  private static RemoteCacheClient createHttp(
      RemoteOptions options,
      Credentials creds,
      AuthAndTLSOptions authAndTlsOptions,
      DigestUtil digestUtil,
      RemoteRetrier retrier,
      Map<String, String> clientEnv) {
    Preconditions.checkNotNull(options.remoteCache, "remoteCache");

    try {
      URI uri = URI.create(options.remoteCache);
      Preconditions.checkArgument(
          Ascii.toLowerCase(uri.getScheme()).startsWith("http"),
          "remoteCache should start with http");

      if (options.remoteProxy != null) {
        if (options.remoteProxy.startsWith("unix:")) {
          // Unix domain socket proxy
          return HttpCacheClient.create(
              new DomainSocketAddress(options.remoteProxy.replaceFirst("^unix:", "")),
              uri,
              Math.toIntExact(options.remoteTimeout.toSeconds()),
              options.remoteMaxConnections,
              options.remoteVerifyDownloads,
              ImmutableList.copyOf(options.remoteHeaders),
              digestUtil,
              retrier,
              creds,
              authAndTlsOptions);
        } else {
          // HTTP proxy from flag (e.g., http://proxy:8080)
          ProxyInfo proxyInfo = RemoteProxyHelper.parseProxyAddress(options.remoteProxy);
          if (proxyInfo.hasProxy()) {
            return HttpCacheClient.create(
                uri,
                Math.toIntExact(options.remoteTimeout.toSeconds()),
                options.remoteMaxConnections,
                options.remoteVerifyDownloads,
                ImmutableList.copyOf(options.remoteHeaders),
                digestUtil,
                retrier,
                creds,
                authAndTlsOptions,
                proxyInfo.address(),
                proxyInfo.username(),
                proxyInfo.password());
          } else {
            throw new Exception("Invalid remote cache proxy: " + options.remoteProxy);
          }
        }
      } else {
        // No explicit proxy flag - check environment variables (HTTPS_PROXY, HTTP_PROXY)
        RemoteProxyHelper proxyHelper = new RemoteProxyHelper(clientEnv);
        ProxyInfo proxyInfo = proxyHelper.createProxyIfNeeded(uri);
        if (proxyInfo.hasProxy()) {
          return HttpCacheClient.create(
              uri,
              Math.toIntExact(options.remoteTimeout.toSeconds()),
              options.remoteMaxConnections,
              options.remoteVerifyDownloads,
              ImmutableList.copyOf(options.remoteHeaders),
              digestUtil,
              retrier,
              creds,
              authAndTlsOptions,
              proxyInfo.address(),
              proxyInfo.username(),
              proxyInfo.password());
        } else {
          // Direct connection
          return HttpCacheClient.create(
              uri,
              Math.toIntExact(options.remoteTimeout.toSeconds()),
              options.remoteMaxConnections,
              options.remoteVerifyDownloads,
              ImmutableList.copyOf(options.remoteHeaders),
              digestUtil,
              retrier,
              creds,
              authAndTlsOptions);
        }
      }
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  public static DiskCacheClient createDiskCache(
      Path workingDirectory, RemoteOptions options, DigestUtil digestUtil, boolean verifyDownloads)
      throws IOException {
    Path cacheDir = workingDirectory.getRelative(Preconditions.checkNotNull(options.diskCache));
    return new DiskCacheClient(cacheDir, digestUtil, verifyDownloads);
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
