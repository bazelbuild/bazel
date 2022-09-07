// Copyright 2021 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Map;
import java.util.function.Supplier;

/** Prod implementation of {@link RegistryFactory}. */
public class RegistryFactoryImpl implements RegistryFactory {
  private final DownloadManager downloadManager;
  private final Supplier<Map<String, String>> clientEnvironmentSupplier;

  public RegistryFactoryImpl(
      DownloadManager downloadManager, Supplier<Map<String, String>> clientEnvironmentSupplier) {
    this.downloadManager = downloadManager;
    this.clientEnvironmentSupplier = clientEnvironmentSupplier;
  }

  @Override
  public Registry getRegistryWithUrl(String url) throws URISyntaxException {
    URI uri = new URI(url);
    if (uri.getScheme() == null) {
      throw new URISyntaxException(
          uri.toString(), "Registry URL has no scheme -- did you mean to use file://?");
    }
    switch (uri.getScheme()) {
      case "http":
      case "https":
      case "file":
        return new IndexRegistry(uri, downloadManager, clientEnvironmentSupplier.get());
      default:
        throw new URISyntaxException(uri.toString(), "Unrecognized registry URL protocol");
    }
  }
}
