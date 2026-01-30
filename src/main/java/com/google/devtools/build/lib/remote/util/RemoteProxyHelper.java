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

package com.google.devtools.build.lib.remote.util;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Strings;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URLDecoder;
import java.util.Map;
import java.util.Objects;
import java.util.function.Supplier;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import javax.annotation.Nullable;

/**
 * Helper class for setting up proxy server for remote cache and remote execution.
 *
 * <p>This class detects proxy configuration from environment variables (HTTPS_PROXY, HTTP_PROXY)
 * and system properties, similar to how the repository downloader handles proxies.
 */
public final class RemoteProxyHelper {

  private final Map<String, String> env;

  /** Information about a proxy server, including address and optional credentials. */
  public record ProxyInfo(
      @Nullable InetSocketAddress address,
      @Nullable String username,
      @Nullable String password,
      boolean useTls) {

    public static final ProxyInfo NO_PROXY = new ProxyInfo(null, null, null, false);

    public boolean hasProxy() {
      return address != null;
    }

    public boolean hasCredentials() {
      return username != null && password != null;
    }
  }

  /**
   * Creates a new instance.
   *
   * @param env client environment to check for proxy settings
   */
  public RemoteProxyHelper(Map<String, String> env) {
    this.env = env;
  }

  /**
   * Creates proxy configuration for a remote cache/executor URL if HTTPS_PROXY/HTTP_PROXY
   * environment variables are set.
   *
   * @param targetUri the URI of the remote cache or executor
   * @return ProxyInfo containing the proxy address and optional credentials
   * @throws IOException if the proxy address is invalid
   */
  public ProxyInfo createProxyIfNeeded(URI targetUri) throws IOException {
    // Check no_proxy/NO_PROXY environment variables
    if (shouldBypassProxy(targetUri.getHost())) {
      return ProxyInfo.NO_PROXY;
    }

    String proxyAddress = null;
    String scheme = targetUri.getScheme().toLowerCase();

    // For HTTPS targets, check https_proxy/HTTPS_PROXY
    if (scheme.equals("https") || scheme.equals("grpcs")) {
      proxyAddress =
          Stream.of(
                  (Supplier<String>) () -> env.get("https_proxy"),
                  () -> env.get("HTTPS_PROXY"),
                  () -> {
                    String host = System.getProperty("https.proxyHost");
                    if (host == null) {
                      return null;
                    }
                    String port = System.getProperty("https.proxyPort");
                    return host + (port != null ? ":" + port : "");
                  })
              .map(Supplier::get)
              .filter(Objects::nonNull)
              .findFirst()
              .orElse(null);
    }

    // For HTTP/gRPC targets, check http_proxy/HTTP_PROXY
    if (proxyAddress == null
        && (scheme.equals("http") || scheme.equals("grpc") || scheme.startsWith("http"))) {
      proxyAddress =
          Stream.of(
                  (Supplier<String>) () -> env.get("http_proxy"),
                  () -> env.get("HTTP_PROXY"),
                  () -> {
                    String host = System.getProperty("http.proxyHost");
                    if (host == null) {
                      return null;
                    }
                    String port = System.getProperty("http.proxyPort");
                    return host + (port != null ? ":" + port : "");
                  })
              .map(Supplier::get)
              .filter(Objects::nonNull)
              .findFirst()
              .orElse(null);
    }

    return parseProxyAddress(proxyAddress);
  }

  private boolean shouldBypassProxy(String host) {
    // Check no_proxy/NO_PROXY environment variables
    String noProxyUrl = env.get("no_proxy");
    if (Strings.isNullOrEmpty(noProxyUrl)) {
      noProxyUrl = env.get("NO_PROXY");
    }
    if (!Strings.isNullOrEmpty(noProxyUrl)) {
      String[] noProxyUrlArray = noProxyUrl.split("\\s*,\\s*");
      for (String pattern : noProxyUrlArray) {
        if (pattern.isEmpty()) {
          continue;
        }
        if (pattern.startsWith(".")) {
          // This entry applies to sub-domains only.
          if (host.endsWith(pattern)) {
            return true;
          }
        } else {
          // This entry applies to the literal hostname and sub-domains.
          if (host.equals(pattern) || host.endsWith("." + pattern)) {
            return true;
          }
        }
      }
    }

    // Check http.nonProxyHosts system property (Java standard, uses | separator and * wildcards)
    String nonProxyHosts = System.getProperty("http.nonProxyHosts");
    if (!Strings.isNullOrEmpty(nonProxyHosts)) {
      for (String pattern : nonProxyHosts.split("\\|")) {
        pattern = pattern.trim();
        if (pattern.isEmpty()) {
          continue;
        }
        if (pattern.startsWith("*")) {
          // Wildcard at start: *.example.com matches foo.example.com
          if (host.endsWith(pattern.substring(1))) {
            return true;
          }
        } else if (pattern.endsWith("*")) {
          // Wildcard at end: example.* matches example.com
          if (host.startsWith(pattern.substring(0, pattern.length() - 1))) {
            return true;
          }
        } else {
          // Exact match
          if (host.equals(pattern)) {
            return true;
          }
        }
      }
    }

    return false;
  }

  /**
   * Parses a proxy address string into a ProxyInfo.
   *
   * @param proxyAddress The proxy address (e.g., "http://user:pass@proxy.example.com:8080")
   * @return ProxyInfo containing the proxy address and optional credentials
   * @throws IOException if the proxy address is invalid
   */
  public static ProxyInfo parseProxyAddress(@Nullable String proxyAddress) throws IOException {
    if (Strings.isNullOrEmpty(proxyAddress)) {
      return ProxyInfo.NO_PROXY;
    }

    // Pattern to parse proxy URLs
    Pattern urlPattern =
        Pattern.compile("^(https?://)?(([^:@]+?)(?::([^@]+?))?@)?([^:]+)(?::(\\d+))?/?$");
    Matcher matcher = urlPattern.matcher(proxyAddress);
    if (!matcher.matches()) {
      throw new IOException("Proxy address " + proxyAddress + " is not a valid URL");
    }

    final String protocol = matcher.group(1);
    final String idAndPassword = matcher.group(2);
    final String urlUsername = matcher.group(3);
    final String urlPassword = matcher.group(4);
    final String hostname = matcher.group(5);
    final String portRaw = matcher.group(6);

    String cleanProxyAddress = proxyAddress;
    if (idAndPassword != null) {
      cleanProxyAddress = proxyAddress.replace(idAndPassword, "");
    }

    boolean useTls;
    if (protocol == null) {
      useTls = false;
    } else {
      useTls =
          switch (protocol) {
            case "https://" -> true;
            case "http://" -> false;
            default -> throw new IOException("Invalid proxy protocol for " + cleanProxyAddress);
          };
    }

    int port = useTls ? 443 : 80; // Default port numbers
    if (portRaw != null) {
      try {
        port = Integer.parseInt(portRaw);
      } catch (NumberFormatException e) {
        throw new IOException("Error parsing proxy port: " + cleanProxyAddress, e);
      }
    }

    InetSocketAddress address = new InetSocketAddress(hostname, port);

    // Decode URL-encoded credentials
    String username = urlUsername;
    String password = urlPassword;
    if (username != null) {
      if (password == null) {
        throw new IOException("No password given for proxy " + cleanProxyAddress);
      }
      username = URLDecoder.decode(username, UTF_8);
      password = URLDecoder.decode(password, UTF_8);
    }

    return new ProxyInfo(address, username, password, useTls);
  }

  /**
   * Creates a ProxyInfo from explicit host and port, typically from a flag value.
   *
   * @param host the proxy hostname
   * @param port the proxy port
   * @return ProxyInfo with the specified address
   */
  public static ProxyInfo createProxy(String host, int port) {
    return new ProxyInfo(new InetSocketAddress(host, port), null, null, false);
  }

  /**
   * Creates a ProxyInfo from a proxy string that may be a Unix socket path or HTTP proxy URL.
   *
   * @param proxy the proxy string (e.g., "unix:/path/to/socket" or "http://proxy:8080")
   * @return ProxyInfo for HTTP proxies, or null if it's a Unix socket proxy
   * @throws IOException if the proxy address is invalid
   */
  @Nullable
  public static ProxyInfo createProxyFromFlag(@Nullable String proxy) throws IOException {
    if (Strings.isNullOrEmpty(proxy)) {
      return null;
    }
    // Unix socket proxies are handled separately
    if (proxy.startsWith("unix:")) {
      return null;
    }
    return parseProxyAddress(proxy);
  }
}
