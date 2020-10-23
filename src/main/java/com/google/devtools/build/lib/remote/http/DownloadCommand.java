// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.http;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.base.Preconditions;
import java.io.OutputStream;
import java.net.URI;

/** Object sent through the channel pipeline to start a download. */
final class DownloadCommand {

  private final URI uri;
  private final boolean casDownload;
  private final Digest digest;
  private final OutputStream out;

  DownloadCommand(URI uri, boolean casDownload, Digest digest, OutputStream out) {
    this.uri = Preconditions.checkNotNull(uri);
    this.casDownload = casDownload;
    this.digest = Preconditions.checkNotNull(digest);
    this.out = Preconditions.checkNotNull(out);
  }

  public URI uri() {
    return uri;
  }

  public boolean casDownload() {
    return casDownload;
  }

  public Digest digest() {
    return digest;
  }

  public OutputStream out() {
    return out;
  }
}
