// Copyright 2025 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.authandtls.BasicHttpAuthenticationEncoder;
import java.net.Proxy;
import javax.annotation.Nullable;

/**
 * Container for proxy configuration including the proxy address and optional authentication
 * credentials.
 *
 * <p>This class holds both the {@link Proxy} object for connection routing and the credentials
 * needed for proxy authentication. The credentials are encoded as a Basic authentication header
 * value suitable for use in the Proxy-Authorization HTTP header.
 */
public class ProxyInfo {

  /** A ProxyInfo representing no proxy (direct connection). */
  public static final ProxyInfo NO_PROXY = new ProxyInfo(Proxy.NO_PROXY, null, null);

  private final Proxy proxy;
  @Nullable private final String username;
  @Nullable private final String password;

  ProxyInfo(Proxy proxy, @Nullable String username, @Nullable String password) {
    this.proxy = proxy;
    this.username = username;
    this.password = password;
  }

  /** Returns the proxy to use for connections, or {@link Proxy#NO_PROXY} for direct connections. */
  public Proxy proxy() {
    return proxy;
  }

  /** Returns true if this proxy requires authentication. */
  public boolean hasCredentials() {
    return username != null && password != null;
  }

  /**
   * Returns the value for the Proxy-Authorization header, or null if no authentication is needed.
   *
   * <p>The returned value is Base64-encoded in the format required for HTTP Basic authentication
   * (RFC 7617). Uses UTF-8 encoding to support international characters in credentials.
   */
  @Nullable
  public String getProxyAuthorizationHeader() {
    if (!hasCredentials()) {
      return null;
    }
    return BasicHttpAuthenticationEncoder.encode(username, password);
  }
}
