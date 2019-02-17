package com.google.devtools.build.lib.remote.http;

import com.google.common.collect.ImmutableMultimap;
import com.google.devtools.build.lib.runtime.AuthHeaderRequest;
import io.netty.handler.codec.http.HttpRequest;
import java.net.URI;
import java.util.Optional;

/**
 * Concrete realisation of AuthHeaderRequest allowing seperation of request details from
 * authentication providers
 */
class NettyAuthHeaderRequest implements AuthHeaderRequest {

  private final URI uri;
  private final HttpRequest request;

  NettyAuthHeaderRequest(final URI uri, final HttpRequest request) {
    this.uri = uri;
    this.request = request;
  }

  @Override
  public boolean isHttp() {
    return true;
  }

  @Override
  public URI uri() {
    return this.uri;
  }

  public Optional<String> method() {
    return Optional.of(this.request.method().name());
  }

  public Optional<ImmutableMultimap<String, String>> headers() {
    final ImmutableMultimap.Builder<String, String> headers = ImmutableMultimap.builder();
    this.request.headers().forEach(headers::put);
    return Optional.of(headers.build());
  }
}
