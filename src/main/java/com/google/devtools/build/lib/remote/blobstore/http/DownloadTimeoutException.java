package com.google.devtools.build.lib.remote.blobstore.http;

import java.io.IOException;

class DownloadTimeoutException extends IOException {

  public DownloadTimeoutException(String url, long bytesReceived, long contentLength) {
    super(buildMessage(url, bytesReceived, contentLength));
  }

  private static String buildMessage(String url, long bytesReceived, long contentLength) {
    if (contentLength < 0) {
      return String.format("Download of '%s' timed out. Received %d bytes.", url, bytesReceived);
    } else {
      return String.format("Download of '%s' timed out. Received %d/%d bytes.", url, bytesReceived,
          contentLength);
    }
  }
}
