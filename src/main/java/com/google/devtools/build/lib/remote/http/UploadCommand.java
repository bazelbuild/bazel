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

import com.google.common.base.Preconditions;
import java.io.InputStream;
import java.net.URI;

/** Object sent through the channel pipeline to start an upload. */
final class UploadCommand {

  private final URI uri;
  private final boolean casUpload;
  private final String hash;
  private final InputStream data;
  private final long contentLength;

  UploadCommand(URI uri, boolean casUpload, String hash, InputStream data, long contentLength) {
    this.uri = Preconditions.checkNotNull(uri);
    this.casUpload = casUpload;
    this.hash = Preconditions.checkNotNull(hash);
    this.data = Preconditions.checkNotNull(data);
    this.contentLength = contentLength;
  }

  public URI uri() {
    return uri;
  }

  public boolean casUpload() {
    return casUpload;
  }

  public String hash() {
    return hash;
  }

  public InputStream data() {
    return data;
  }

  public long contentLength() {
    return contentLength;
  }
}
