package com.google.devtools.build.lib.bazel.repository;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Strings;
import java.io.IOException;
import java.net.Authenticator;
import java.net.InetSocketAddress;
import java.net.PasswordAuthentication;
import java.net.Proxy;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


/**
 * Helper class for setting up a proxy server for network communication
 *
 */
public class ProxyHelper {

  /**
   * This method takes a proxyAddress as a String (ex. http://userId:password@proxyhost.domain.com:8000)
   * and sets JVM arguments for http and https proxy as well as returns a java.net.Proxy object for optional use.
   * 
   * @param proxyAddress
   * @return Proxy
   * @throws IOException
   */
  @VisibleForTesting
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
    final String cleanProxyAddress = proxyAddress.replace(matcher.group(2), "");
    final String username = matcher.group(3);
    final String password = matcher.group(4);
    final String hostname = matcher.group(5);
    final String port = matcher.group(6);
    
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
    
    // We need to set both of these because we don't know which will be needed by jgit; refactor candidate
    System.setProperty("https.proxyHost", hostname);
    System.setProperty("https.proxyPort", port);
    System.setProperty("http.proxyHost", hostname);
    System.setProperty("http.proxyPort", port);

    if (username != null) {
      if (password == null) {
        throw new IOException("No password given for proxy " + cleanProxyAddress);
      }
      System.setProperty(protocol + ".proxyUser", username);
      System.setProperty(protocol + ".proxyPassword", password);

      Authenticator.setDefault(
          new Authenticator() {
            public PasswordAuthentication getPasswordAuthentication() {
              return new PasswordAuthentication(username, password.toCharArray());
            }
          });
    }

    if (port == null) {
      return new Proxy(Proxy.Type.HTTP, new InetSocketAddress(hostname, https ? 443 : 80));
    }

    try {
      return new Proxy(
          Proxy.Type.HTTP,
          new InetSocketAddress(hostname, Integer.parseInt(port)));
    } catch (NumberFormatException e) {
      throw new IOException("Error parsing proxy port: " + cleanProxyAddress);
    }
  }
}
