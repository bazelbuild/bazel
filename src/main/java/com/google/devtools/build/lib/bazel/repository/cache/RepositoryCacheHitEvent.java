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

package com.google.devtools.build.lib.bazel.repository.cache;

import com.google.devtools.build.lib.events.ExtendedEventHandler.ProgressLike;
import java.net.URL;

/** Event reporting about cache hits for download requests. */
public class RepositoryCacheHitEvent implements ProgressLike {
  private final String repo;
  private final String hash;
  private final URL url;

  public RepositoryCacheHitEvent(String repo, String hash, URL url) {
    this.repo = repo;
    this.hash = hash;
    this.url = url;
  }

  public String getRepo() {
    return repo;
  }

  public URL getUrl() {
    return url;
  }

  public String getFileHash() {
    return hash;
  }
}
