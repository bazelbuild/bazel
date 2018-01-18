package com.google.devtools.build.lib.remote.blobstore.http;

import io.netty.handler.codec.http.HttpResponseStatus;
import java.io.IOException;

class HttpException extends IOException {
  private HttpResponseStatus status;

  HttpException(HttpResponseStatus status, String message, Throwable cause) {
    super(message, cause);
    this.status = status;
  }

  public HttpResponseStatus status() {
    return status;
  }
}
