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

import java.io.IOException;

/** Exception thrown when a HTTP download times out. */
public class DownloadTimeoutException extends IOException {

  public DownloadTimeoutException(String url, long bytesReceived, long contentLength) {
    super(buildMessage(url, bytesReceived, contentLength));
  }

  private static String buildMessage(String url, long bytesReceived, long contentLength) {
    if (contentLength < 0) {
      return String.format("Download of '%s' timed out. Received %d bytes.", url, bytesReceived);
    } else {
      return String.format(
          "Download of '%s' timed out. Received %d of %d bytes.",
          url, bytesReceived, contentLength);
    }
  }
}
