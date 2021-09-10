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
package com.google.devtools.build.lib.bazel.repository.downloader;

import java.io.IOException;

/** An exception indicates that the size of downloaded content doesn't match the Content-Length. */
public class ContentLengthMismatchException extends IOException {
  private final long actualSize;
  private final long expectedSize;

  public ContentLengthMismatchException(long actualSize, long expectedSize) {
    super(String.format("Bytes read %s but wanted %s", actualSize, expectedSize));
    this.actualSize = actualSize;
    this.expectedSize = expectedSize;
  }

  public long getActualSize() {
    return actualSize;
  }

  public long getExpectedSize() {
    return expectedSize;
  }
}
