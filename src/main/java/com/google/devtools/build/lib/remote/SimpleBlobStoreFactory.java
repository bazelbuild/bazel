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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.auth.Credentials;
import com.google.devtools.build.lib.remote.blobstore.OnDiskBlobStore;
import com.google.devtools.build.lib.remote.blobstore.RestBlobStore;
import com.google.devtools.build.lib.remote.blobstore.SimpleBlobStore;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.concurrent.TimeUnit;
import javax.annotation.Nullable;

/**
 * A factory class for providing a {@link SimpleBlobStore} to be used with {@link
 * SimpleBlobStoreActionCache}. Currently implemented with REST or local.
 */
public final class SimpleBlobStoreFactory {

  private SimpleBlobStoreFactory() {}

  public static SimpleBlobStore createRest(RemoteOptions options, Credentials creds)
      throws IOException {
    return new RestBlobStore(
        options.remoteRestCache, (int) TimeUnit.SECONDS.toMillis(options.remoteTimeout), creds);
  }

  public static SimpleBlobStore createLocalDisk(RemoteOptions options, Path workingDirectory)
      throws IOException {
    return new OnDiskBlobStore(
        workingDirectory.getRelative(checkNotNull(options.experimentalLocalDiskCachePath)));
  }

  public static SimpleBlobStore create(
      RemoteOptions options, @Nullable Credentials creds, @Nullable Path workingDirectory)
      throws IOException {
    if (isRestUrlOptions(options)) {
      return createRest(options, creds);
    }
    if (workingDirectory != null && isLocalDiskCache(options)) {
      return createLocalDisk(options, workingDirectory);
    }
    throw new IllegalArgumentException(
        "Unrecognized concurrent map RemoteOptions: must specify "
            + "either Rest URL, or local cache options.");
  }

  public static boolean isRemoteCacheOptions(RemoteOptions options) {
    return isRestUrlOptions(options) || isLocalDiskCache(options);
  }

  public static boolean isLocalDiskCache(RemoteOptions options) {
    return options.experimentalLocalDiskCache;
  }

  private static boolean isRestUrlOptions(RemoteOptions options) {
    return options.remoteRestCache != null;
  }
}
