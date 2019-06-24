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

import static com.google.common.base.Strings.isNullOrEmpty;

import com.google.auth.Credentials;
import com.google.common.base.Ascii;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.remote.blobstore.CombinedDiskHttpBlobStore;
import com.google.devtools.build.lib.remote.blobstore.ConcurrentMapBlobStore;
import com.google.devtools.build.lib.remote.blobstore.OnDiskBlobStore;
import com.google.devtools.build.lib.remote.blobstore.SimpleBlobStore;
import com.google.devtools.build.lib.remote.blobstore.grpc.GrpcBlobStore;
import com.google.devtools.build.lib.remote.blobstore.http.HttpBlobStore;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.shared.ByteStreamUploader;
import com.google.devtools.build.lib.remote.shared.ReferenceCountedChannel;
import com.google.devtools.build.lib.remote.shared.RemoteRetrier;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import io.grpc.CallCredentials;
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
    if (isHttpCache(remoteOptions)) {
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
    if (isHttpCache(options) && isDiskCache(options)) {
      return createCombinedCache(workingDirectory, options.diskCache, options, creds);
    }
    if (isHttpCache(options)) {
      return createHttp(options, creds);
    }
    if (isDiskCache(options)) {
      return createDiskCache(workingDirectory, options.diskCache);
    }
    throw new IllegalArgumentException(
        "Unrecognized RemoteOptions configuration: remote Http cache URL and/or local disk cache"
            + " options expected.");
  }

  // TODO: This function can be expanded to combine disk cache and gRPC cache
  public static SimpleBlobStore create(
      RemoteOptions options,
      ReferenceCountedChannel channel,
      CallCredentials credentials,
      RemoteRetrier retrier,
      ByteStreamUploader uploader,
      DigestUtil digestUtil)
      throws IOException {
    return createGrpcCache(channel, credentials, options, retrier, digestUtil, uploader);
  }

  private static SimpleBlobStore createHttp(RemoteOptions options, Credentials creds) {
    Preconditions.checkNotNull(options.remoteCache, "remoteCache");

    try {
      URI uri = URI.create(options.remoteCache);
      Preconditions.checkArgument(
          Ascii.toLowerCase(uri.getScheme()).startsWith("http"),
          "remoteCache should start with http");

      if (options.remoteProxy != null) {
        if (options.remoteProxy.startsWith("unix:")) {
          return HttpBlobStore.create(
              new DomainSocketAddress(options.remoteProxy.replaceFirst("^unix:", "")),
              uri,
              options.remoteTimeout,
              options.remoteMaxConnections,
              creds);
        } else {
          throw new Exception("Remote cache proxy unsupported: " + options.remoteProxy);
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

  private static SimpleBlobStore createGrpcCache(
      ReferenceCountedChannel channel,
      CallCredentials credentials,
      RemoteOptions options,
      RemoteRetrier retrier,
      DigestUtil digestUtil,
      ByteStreamUploader uploader) {
    return new GrpcBlobStore(channel, credentials, options, retrier, digestUtil, uploader);
  }

  private static SimpleBlobStore createCombinedCache(
      Path workingDirectory, PathFragment diskCachePath, RemoteOptions options, Credentials cred)
      throws IOException {

    Path cacheDir =
        workingDirectory.getRelative(Preconditions.checkNotNull(diskCachePath, "diskCachePath"));
    if (!cacheDir.exists()) {
      cacheDir.createDirectoryAndParents();
    }

    OnDiskBlobStore diskCache = new OnDiskBlobStore(cacheDir);
    SimpleBlobStore httpCache = createHttp(options, cred);
    return new CombinedDiskHttpBlobStore(diskCache, httpCache);
  }

  public static boolean isDiskCache(RemoteOptions options) {
    return options.diskCache != null && !options.diskCache.isEmpty();
  }

  public static boolean isHttpCache(RemoteOptions options) {
    return options.remoteCache != null
        && (Ascii.toLowerCase(options.remoteCache).startsWith("http://")
            || Ascii.toLowerCase(options.remoteCache).startsWith("https://"));
  }

  /** Returns true if 'options.remoteCache' uses 'grpc' or an empty scheme */
  public static boolean isGrpcCache(RemoteOptions options) {
    if (isNullOrEmpty(options.remoteCache)) {
      return false;
    }
    // TODO(ishikhman): add proper URI validation/parsing for remote options
    return !(Ascii.toLowerCase(options.remoteCache).startsWith("http://")
        || Ascii.toLowerCase(options.remoteCache).startsWith("https://"));
  }
}
