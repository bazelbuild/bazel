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
import com.google.devtools.build.lib.remote.util.Utils;
import java.net.URL;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Locale;
import java.util.OptionalLong;

/**
 * Postable event reporting on progress made downloading an URL. It can be used to report the URL
 * being downloaded and the number of bytes read so far.
 */
public class DownloadProgressEvent implements ExtendedEventHandler.FetchProgress {
  private final URL originalUrl;
  private final URL actualUrl;
  private final long bytesRead;
  private final OptionalLong totalBytes;
  private final boolean downloadFinished;

  public DownloadProgressEvent(
      URL originalUrl, URL actualUrl, long bytesRead, OptionalLong totalBytes, boolean finished) {
    this.originalUrl = originalUrl;
    this.actualUrl = actualUrl;
    this.bytesRead = bytesRead;
    this.totalBytes = totalBytes;
    this.downloadFinished = finished;
  }

  public DownloadProgressEvent(URL originalUrl, long bytesRead, boolean finished) {
    this(originalUrl, null, bytesRead, OptionalLong.empty(), finished);
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

  private static final DecimalFormat PERCENTAGE_FORMAT =
      new DecimalFormat("0.0%", new DecimalFormatSymbols(Locale.US));

  @Override
  public String getProgress() {
    if (bytesRead > 0) {
      if (totalBytes.isPresent()) {
        double totalBytesDouble = this.totalBytes.getAsLong();
        double ratio = totalBytesDouble != 0 ? bytesRead / totalBytesDouble : 1;
        // 10.1 MiB (20.2%)
        return String.format(
            "%s (%s)", Utils.bytesCountToDisplayString(bytesRead), PERCENTAGE_FORMAT.format(ratio));
      } else {
        // 10.1 MiB (10,590,000B)
        return String.format("%s (%,dB)", Utils.bytesCountToDisplayString(bytesRead), bytesRead);
      }
    } else {
      return "";
    }
  }
}
