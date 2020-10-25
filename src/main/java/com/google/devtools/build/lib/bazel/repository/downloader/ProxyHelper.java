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

/**
 * Helper class for setting up a proxy server for network communication
 */
public class ProxyHelper {

  private final Map<String, String> env;

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
   */
  public Proxy createProxyIfNeeded(URL requestedUrl) throws IOException {
    String proxyAddress = null;
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
            return Proxy.NO_PROXY;
          }
        } else {
          // This entry applies to the literal hostname and sub-domains.
          if (requestedHost.equals(noProxyUrlArray[i])
              || requestedHost.endsWith("." + noProxyUrlArray[i])) {
            return Proxy.NO_PROXY;
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
                    String port = System.getProperty("https.proxyPort");

                    return String.format("%s%s", host, port == null ? "" : ":" + port);
                  })
              .map(Supplier::get)
              .filter(Objects::nonNull)
              .findFirst()
              .orElse(null);
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
                    String port = System.getProperty("http.proxyPort");

                    return String.format("%s%s", host, port == null ? "" : ":" + port);
                  })
              .map(Supplier::get)
              .filter(Objects::nonNull)
              .findFirst()
              .orElse(null);
    }
    return createProxy(proxyAddress);
  }

  /**
   * This method takes a proxyAddress as a String (ex.
   * http://userId:password@proxyhost.domain.com:8000) and sets JVM arguments for http and https
   * proxy as well as returns a java.net.Proxy object for optional use.
   *
   * @param proxyAddress The fully qualified address of the proxy server
   * @return Proxy
   * @throws IOException
   */
  public static Proxy createProxy(@Nullable String proxyAddress) throws IOException {
    if (Strings.isNullOrEmpty(proxyAddress)) {
      return Proxy.NO_PROXY;
    }

    // Here there be dragons.
    Pattern urlPattern =
        Pattern.compile("^(https?)://(([^:@]+?)(?::([^@]+?))?@)?([^:]+)(?::(\\d+))?/?$");
    Matcher matcher = urlPattern.matcher(proxyAddress);
    if (!matcher.matches()) {
      throw new IOException("Proxy address " + proxyAddress + " is not a valid URL");
    }

    final String protocol = matcher.group(1);
    final String idAndPassword = matcher.group(2);
    final String username = matcher.group(3);
    final String password = matcher.group(4);
    final String hostname = matcher.group(5);
    final String portRaw = matcher.group(6);

    String cleanProxyAddress = proxyAddress;
    if (idAndPassword != null) {
      cleanProxyAddress =
          proxyAddress.replace(idAndPassword, ""); // Used to remove id+pwd from logging
    }

    boolean https;
    switch (protocol) {
      case "https":
        https = true;
        break;
      case "http":
        https = false;
        break;
      default:
        throw new IOException("Invalid proxy protocol for " + cleanProxyAddress);
    }

    int port = https ? 443 : 80; // Default port numbers

    if (portRaw != null) {
      try {
        port = Integer.parseInt(portRaw);
      } catch (NumberFormatException e) {
        throw new IOException("Error parsing proxy port: " + cleanProxyAddress, e);
      }
    }

    if (username != null) {
      if (password == null) {
        throw new IOException("No password given for proxy " + cleanProxyAddress);
      }

      // We need to make sure the proxy password is not url encoded; some special characters in
      // proxy passwords require url encoding for shells and other tools to properly consume.
      final String decodedPassword = URLDecoder.decode(password, "UTF-8");
      Authenticator.setDefault(
          new Authenticator() {
            @Override
            public PasswordAuthentication getPasswordAuthentication() {
              return new PasswordAuthentication(username, decodedPassword.toCharArray());
            }
          });
    }

    return new Proxy(Proxy.Type.HTTP, new InetSocketAddress(hostname, port));
  }
}
