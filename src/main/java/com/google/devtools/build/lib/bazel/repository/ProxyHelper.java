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

package com.google.devtools.build.lib.bazel.repository;

import com.google.common.base.Strings;

import java.io.IOException;
import java.net.Authenticator;
import java.net.InetSocketAddress;
import java.net.PasswordAuthentication;
import java.net.Proxy;
import java.net.URLDecoder;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Helper class for setting up a proxy server for network communication
 *
 */
public class ProxyHelper {

  /**
   * This method takes a String for the resource being requested and sets up a proxy to make
   * the request if HTTP_PROXY and/or HTTPS_PROXY environment variables are set.
   * @param requestedUrl The url for the remote resource that may need to be retrieved through a
   *   proxy
   * @return Proxy
   * @throws IOException
   */
  public static Proxy createProxyIfNeeded(String requestedUrl) throws IOException {
    String lcUrl = requestedUrl.toLowerCase();
    if (lcUrl.startsWith("https")) {
      return createProxy(System.getenv("HTTPS_PROXY"));
    } else if (lcUrl.startsWith("http")) {
      return createProxy(System.getenv("HTTP_PROXY"));
    }
    return Proxy.NO_PROXY;
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
  public static Proxy createProxy(String proxyAddress) throws IOException {
    if (Strings.isNullOrEmpty(proxyAddress)) {
      return Proxy.NO_PROXY;
    }

    // Here there be dragons.
    Pattern urlPattern =
        Pattern.compile("^(https?)://(([^:@]+?)(?::([^@]+?))?@)?([^:]+)(?::(\\d+))?$");
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
        throw new IOException("Error parsing proxy port: " + cleanProxyAddress);
      }
    }

    // We need to set both of these because jgit uses whichever the resource dictates
    System.setProperty("https.proxyHost", hostname);
    System.setProperty("https.proxyPort", Integer.toString(port));
    System.setProperty("http.proxyHost", hostname);
    System.setProperty("http.proxyPort", Integer.toString(port));

    if (username != null) {
      if (password == null) {
        throw new IOException("No password given for proxy " + cleanProxyAddress);
      }

      // We need to make sure the proxy password is not url encoded; some special characters in
      // proxy passwords require url encoding for shells and other tools to properly consume.
      final String decodedPassword = URLDecoder.decode(password, "UTF-8");
      System.setProperty("http.proxyUser", username);
      System.setProperty("http.proxyPassword", decodedPassword);
      System.setProperty("https.proxyUser", username);
      System.setProperty("https.proxyPassword", decodedPassword);

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
