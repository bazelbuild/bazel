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

package com.google.devtools.build.lib.bazel.repository.downloader;

import com.google.devtools.build.lib.events.ExtendedEventHandler;
import java.net.URL;
import java.text.NumberFormat;
import java.util.Locale;

/**
 * Postable event reporting on progress made downloading an URL. It can be used to report the URL
 * being downloaded and the number of bytes read so far.
 */
public class DownloadProgressEvent implements ExtendedEventHandler.FetchProgress {
  private final URL originalUrl;
  private final URL actualUrl;
  private final long bytesRead;
  private final boolean downloadFinished;

  public DownloadProgressEvent(URL originalUrl, URL actualUrl, long bytesRead, boolean finished) {
    this.originalUrl = originalUrl;
    this.actualUrl = actualUrl;
    this.bytesRead = bytesRead;
    this.downloadFinished = finished;
  }

  public DownloadProgressEvent(URL originalUrl, long bytesRead, boolean finished) {
    this(originalUrl, null, bytesRead, finished);
  }

  public DownloadProgressEvent(URL url, long bytesRead) {
    this(url, bytesRead, false);
  }

  public DownloadProgressEvent(URL url) {
    this(url, 0);
  }

  public URL getOriginalUrl() {
    return originalUrl;
  }

  @Override
  public String getResourceIdentifier() {
    return originalUrl.toString();
  }

  public URL getActualUrl() {
    return actualUrl;
  }

  @Override
  public boolean isFinished() {
    return downloadFinished;
  }

  public long getBytesRead() {
    return bytesRead;
  }

  @Override
  public String getProgress() {
    if (bytesRead > 0) {
      NumberFormat formatter = NumberFormat.getIntegerInstance(Locale.ENGLISH);
      formatter.setGroupingUsed(true);
      return formatter.format(bytesRead) + "B";
    } else {
      return "";
    }
  }
}
