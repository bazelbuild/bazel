// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository.downloader;

import com.google.common.base.Optional;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.net.URI;
import java.net.URL;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A {@link Downloader} that delegates to another Downloader. Primarily useful for mutable
 * dependency injection.
 */
public class DelegatingDownloader implements Downloader {
  private final Downloader defaultDelegate;
  @Nullable private Downloader delegate;

  public DelegatingDownloader(Downloader defaultDelegate) {
    this.defaultDelegate = defaultDelegate;
  }

  /**
   * Sets the {@link Downloader} to delegate to. If setDelegate(null) is called, the default
   * delegate passed to the constructor will be used.
   */
  public void setDelegate(@Nullable Downloader delegate) {
    this.delegate = delegate;
  }

  @Override
  public void download(
      List<URL> urls,
      Map<URI, Map<String, String>> authHeaders,
      Optional<Checksum> checksum,
      String canonicalId,
      Path destination,
      ExtendedEventHandler eventHandler,
      Map<String, String> clientEnv,
      Optional<String> type)
      throws IOException, InterruptedException {
    Downloader downloader = defaultDelegate;
    if (delegate != null) {
      downloader = delegate;
    }
    downloader.download(
        urls, authHeaders, checksum, canonicalId, destination, eventHandler, clientEnv, type);
  }
}
