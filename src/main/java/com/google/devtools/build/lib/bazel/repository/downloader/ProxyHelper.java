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

import static com.google.devtools.build.lib.util.StringEncoding.internalToUnicode;
import static com.google.devtools.build.lib.util.StringEncoding.platformToInternal;
import static com.google.devtools.build.lib.util.StringEncoding.unicodeToInternal;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Splitter;
import com.google.common.base.Strings;
import java.io.IOException;
import java.net.Authenticator;
import java.net.InetSocketAddress;
import java.net.PasswordAuthentication;
import java.net.Proxy;
import java.net.URL;
import java.net.URLDecoder;
import java.util.Map;
import java.util.Objects;
import java.util.function.Supplier;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import javax.annotation.Nullable;

/** Helper class for setting up a proxy server for network communication */
public class ProxyHelper {

  // Lock for thread-safe authenticator setup. The Authenticator is a JVM-wide singleton,
  // so we use double-checked locking to ensure it's only set once.
  private static final Object AUTHENTICATOR_LOCK = new Object();
  private static volatile boolean authenticatorSet = false;

  private final Map<String, String> env;

  /** Resets the static authenticator state. This is intended for testing only. */
  static void resetAuthenticatorForTesting() {
    synchronized (AUTHENTICATOR_LOCK) {
      Authenticator.setDefault(null);
      authenticatorSet = false;
    }
  }

  /**
   * Creates new instance.
   *
   * @param env client environment to check for proxy settings
   */
  public ProxyHelper(Map<String, String> env) {
    this.env = env;
  }

  /**
   * This method takes a String for the resource being requested and sets up a proxy to make the
   * request if HTTP_PROXY and/or HTTPS_PROXY environment variables are set, or if the standard
   * `https_proxy` and `http_proxy` system properties are set.
   *
   * @param requestedUrl remote resource that may need to be retrieved through a proxy
   * @return ProxyInfo containing the proxy and optional credentials
   */
  public ProxyInfo createProxyIfNeeded(URL requestedUrl) throws IOException {
    String proxyAddress = null;
    String proxyUserProperty = null;
    String proxyPasswordProperty = null;

    // Check no_proxy/NO_PROXY environment variables
    String noProxyUrl = env.get("no_proxy");
    if (Strings.isNullOrEmpty(noProxyUrl)) {
      noProxyUrl = env.get("NO_PROXY");
    }
    if (!Strings.isNullOrEmpty(noProxyUrl)) {
      String[] noProxyUrlArray = noProxyUrl.split("\\s*,\\s*");
      String requestedHost = requestedUrl.getHost();
      for (int i = 0; i < noProxyUrlArray.length; i++) {
        if (noProxyUrlArray[i].startsWith(".")) {
          // This entry applies to sub-domains only.
          if (requestedHost.endsWith(noProxyUrlArray[i])) {
            return ProxyInfo.NO_PROXY;
          }
        } else {
          // This entry applies to the literal hostname and sub-domains.
          if (requestedHost.equals(noProxyUrlArray[i])
              || requestedHost.endsWith("." + noProxyUrlArray[i])) {
            return ProxyInfo.NO_PROXY;
          }
        }
      }
    }

    // Check http.nonProxyHosts system property (Java standard, uses | separator and * wildcards)
    String nonProxyHosts = System.getProperty("http.nonProxyHosts");
    if (!Strings.isNullOrEmpty(nonProxyHosts)) {
      nonProxyHosts = platformToInternal(nonProxyHosts);
      String requestedHost = requestedUrl.getHost();
      for (String pattern : Splitter.on('|').split(nonProxyHosts)) {
        pattern = pattern.trim();
        if (pattern.isEmpty()) {
          continue;
        }
        if (pattern.startsWith("*")) {
          // Wildcard at start: *.example.com matches foo.example.com
          if (requestedHost.endsWith(pattern.substring(1))) {
            return ProxyInfo.NO_PROXY;
          }
        } else if (pattern.endsWith("*")) {
          // Wildcard at end: example.* matches example.com
          if (requestedHost.startsWith(pattern.substring(0, pattern.length() - 1))) {
            return ProxyInfo.NO_PROXY;
          }
        } else {
          // Exact match
          if (requestedHost.equals(pattern)) {
            return ProxyInfo.NO_PROXY;
          }
        }
      }
    }

    if (HttpUtils.isProtocol(requestedUrl, "https")) {
      proxyAddress =
          Stream.of(
                  (Supplier<String>) () -> env.get("https_proxy"),
                  () -> env.get("HTTPS_PROXY"),
                  () -> {
                    String host = System.getProperty("https.proxyHost");
                    if (host == null) {
                      return null;
                    }
                    host = platformToInternal(host);
                    String port = System.getProperty("https.proxyPort");
                    if (port != null) {
                      port = platformToInternal(port);
                    }

                    return String.format("%s%s", host, port == null ? "" : ":" + port);
                  })
              .map(Supplier::get)
              .filter(Objects::nonNull)
              .findFirst()
              .orElse(null);
      // Check for credentials in system properties
      proxyUserProperty = System.getProperty("https.proxyUser");
      if (proxyUserProperty != null) {
        proxyUserProperty = platformToInternal(proxyUserProperty);
      }
      proxyPasswordProperty = System.getProperty("https.proxyPassword");
      if (proxyPasswordProperty != null) {
        proxyPasswordProperty = platformToInternal(proxyPasswordProperty);
      }
    } else if (HttpUtils.isProtocol(requestedUrl, "http")) {
      proxyAddress =
          Stream.of(
                  (Supplier<String>) () -> env.get("http_proxy"),
                  () -> env.get("HTTP_PROXY"),
                  () -> {
                    String host = System.getProperty("http.proxyHost");
                    if (host == null) {
                      return null;
                    }
                    host = platformToInternal(host);
                    String port = System.getProperty("http.proxyPort");
                    if (port != null) {
                      port = platformToInternal(port);
                    }

                    return String.format("%s%s", host, port == null ? "" : ":" + port);
                  })
              .map(Supplier::get)
              .filter(Objects::nonNull)
              .findFirst()
              .orElse(null);
      // Check for credentials in system properties
      proxyUserProperty = System.getProperty("http.proxyUser");
      if (proxyUserProperty != null) {
        proxyUserProperty = platformToInternal(proxyUserProperty);
      }
      proxyPasswordProperty = System.getProperty("http.proxyPassword");
      if (proxyPasswordProperty != null) {
        proxyPasswordProperty = platformToInternal(proxyPasswordProperty);
      }
    }
    return createProxyInfo(proxyAddress, proxyUserProperty, proxyPasswordProperty);
  }

  /**
   * This method takes a proxyAddress as a String (ex. {@code
   * http://userId:password@proxyhost.domain.com:8000}) and returns a ProxyInfo containing the proxy
   * configuration and optional authentication credentials.
   *
   * @param proxyAddress The fully qualified address of the proxy server
   * @return ProxyInfo containing the proxy and optional credentials
   * @throws IOException if the proxy address is invalid
   */
  public static ProxyInfo createProxy(@Nullable String proxyAddress) throws IOException {
    return createProxyInfo(proxyAddress, null, null);
  }

  /**
   * This method creates a ProxyInfo from either a proxy address URL (which may contain embedded
   * credentials) or from separate credential parameters (typically from system properties).
   *
   * <p>Credentials in the proxy address URL take precedence over separately provided credentials.
   *
   * @param proxyAddress The proxy address, optionally containing embedded credentials
   * @param systemPropertyUser Username from system property (http.proxyUser/https.proxyUser)
   * @param systemPropertyPassword Password from system property
   *     (http.proxyPassword/https.proxyPassword)
   * @return ProxyInfo containing the proxy and optional credentials
   * @throws IOException if the proxy address is invalid
   */
  public static ProxyInfo createProxyInfo(
      @Nullable String proxyAddress,
      @Nullable String systemPropertyUser,
      @Nullable String systemPropertyPassword)
      throws IOException {
    if (Strings.isNullOrEmpty(proxyAddress)) {
      return ProxyInfo.NO_PROXY;
    }

    // Here there be dragons.
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
      cleanProxyAddress =
          proxyAddress.replace(idAndPassword, ""); // Used to remove id+pwd from logging
    }

    boolean https;
    if (protocol == null) {
      https = false;
    } else {
      https =
          switch (protocol) {
            case "https://" -> true;
            case "http://" -> false;
            default -> throw new IOException("Invalid proxy protocol for " + cleanProxyAddress);
          };
    }

    int port = https ? 443 : 80; // Default port numbers

    if (portRaw != null) {
      try {
        port = Integer.parseInt(portRaw);
      } catch (NumberFormatException e) {
        throw new IOException("Error parsing proxy port: " + cleanProxyAddress, e);
      }
    }

    Proxy proxy = new Proxy(Proxy.Type.HTTP, new InetSocketAddress(hostname, port));

    // Determine credentials: URL credentials take precedence over system properties
    String username = urlUsername;
    String password = urlPassword;

    if (username != null) {
      if (password == null) {
        throw new IOException("No password given for proxy " + cleanProxyAddress);
      }
      // We need to make sure the proxy credentials are not url encoded; some special characters in
      // proxy passwords require url encoding for shells and other tools to properly consume.
      username = unicodeToInternal(URLDecoder.decode(internalToUnicode(username), UTF_8));
      password = unicodeToInternal(URLDecoder.decode(internalToUnicode(password), UTF_8));
    } else if (systemPropertyUser != null && systemPropertyPassword != null) {
      // Fall back to system property credentials
      username = systemPropertyUser;
      password = systemPropertyPassword;
    }

    // If credentials are provided, also set up Java's Authenticator for HTTPS proxy support.
    // For HTTPS connections through HTTP proxies (CONNECT tunneling), Java's HttpURLConnection
    // handles the CONNECT request internally and won't use Proxy-Authorization header we set.
    // Instead, it uses the Authenticator mechanism. We also enable Basic auth tunneling by
    // clearing the disabled schemes (by default, Basic auth is disabled for HTTPS tunneling).
    if (username != null && password != null) {
      // Use double-checked locking to ensure thread-safe, one-time setup of the global
      // Authenticator. The first caller with credentials wins. This is safe because Bazel
      // typically uses a single proxy configuration for all downloads.
      if (!authenticatorSet) {
        synchronized (AUTHENTICATOR_LOCK) {
          if (!authenticatorSet) {
            final String finalUsername = username;
            final String finalPassword = password;
            // Capture the previous authenticator to delegate non-proxy auth requests to it.
            // This preserves existing behavior for server authentication (e.g., .netrc).
            final Authenticator previousAuthenticator = Authenticator.getDefault();
            Authenticator.setDefault(
                new Authenticator() {
                  @Nullable
                  @Override
                  public PasswordAuthentication getPasswordAuthentication() {
                    // Only provide credentials for proxy authentication.
                    if (getRequestorType() == RequestorType.PROXY) {
                      return new PasswordAuthentication(
                          internalToUnicode(finalUsername),
                          internalToUnicode(finalPassword).toCharArray());
                    }
                    // Delegate non-proxy auth to previous authenticator (if any).
                    // This preserves existing behavior for server authentication.
                    if (previousAuthenticator != null) {
                      return previousAuthenticator.requestPasswordAuthenticationInstance(
                          getRequestingHost(),
                          getRequestingSite(),
                          getRequestingPort(),
                          getRequestingProtocol(),
                          getRequestingPrompt(),
                          getRequestingScheme(),
                          getRequestingURL(),
                          getRequestorType());
                    }
                    return null;
                  }
                });
            // Enable Basic authentication for HTTPS tunneling through HTTP proxies.
            // By default, Java disables Basic auth for tunneling
            // (jdk.http.auth.tunneling.disabledSchemes defaults to "Basic").
            enableBasicAuthTunneling();
            authenticatorSet = true;
          }
        }
      }
    }

    return new ProxyInfo(proxy, username, password);
  }

  /**
   * Enables Basic authentication for HTTPS tunneling through HTTP proxies.
   *
   * <p>By default, Java disables Basic authentication for HTTPS proxy tunneling for security
   * reasons (the {@code jdk.http.auth.tunneling.disabledSchemes} system property defaults to
   * "Basic"). This method clears that restriction to allow authenticated proxies to work with HTTPS
   * URLs.
   *
   * <p>This is necessary because most enterprise proxies use Basic authentication, and without this
   * setting, HTTPS downloads through authenticated proxies will fail with 407 errors.
   *
   * <p>Note: This modifies a JVM-wide setting. If the user has explicitly set this property, their
   * setting will be preserved.
   */
  private static void enableBasicAuthTunneling() {
    // Use putIfAbsent for thread-safe modification. Only modify if not already set by the user.
    System.getProperties().putIfAbsent("jdk.http.auth.tunneling.disabledSchemes", "");
  }
}
