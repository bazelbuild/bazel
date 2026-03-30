// Copyright 2015 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import java.io.IOException;
import java.net.Proxy;
import java.net.URI;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link ProxyHelper}.
 */
@RunWith(JUnit4.class)
public class ProxyHelperTest {

  @Before
  public void setUp() {
    ProxyHelper.resetAuthenticatorForTesting();
  }

  @Test
  public void testCreateIfNeededHttpLowerCase() throws Exception {
    ProxyHelper helper = new ProxyHelper(ImmutableMap.of("http_proxy", "http://my.example.com"));
    ProxyInfo proxyInfo = helper.createProxyIfNeeded(URI.create("http://www.something.com"));
    assertThat(proxyInfo.proxy().toString())
        .containsMatch("my\\.example\\.com(/<unresolved>)?:80$");
  }

  @Test
  public void testCreateIfNeededHttpUpperCase() throws Exception {
    ProxyHelper helper = new ProxyHelper(ImmutableMap.of("HTTP_PROXY", "http://my.example.com"));
    ProxyInfo proxyInfo = helper.createProxyIfNeeded(URI.create("http://www.something.com"));
    assertThat(proxyInfo.proxy().toString())
        .containsMatch("my\\.example\\.com(/<unresolved>)?:80$");
  }

  @Test
  public void testCreateIfNeededHttpsLowerCase() throws Exception {
    ProxyHelper helper = new ProxyHelper(ImmutableMap.of("https_proxy", "https://my.example.com"));
    ProxyInfo proxyInfo = helper.createProxyIfNeeded(URI.create("https://www.something.com"));
    assertThat(proxyInfo.proxy().toString())
        .containsMatch("my\\.example\\.com(/<unresolved>)?:443$");
  }

  @Test
  public void testCreateIfNeededHttpsUpperCase() throws Exception {
    ProxyHelper helper = new ProxyHelper(ImmutableMap.of("HTTPS_PROXY", "https://my.example.com"));
    ProxyInfo proxyInfo = helper.createProxyIfNeeded(URI.create("https://www.something.com"));
    assertThat(proxyInfo.proxy().toString())
        .containsMatch("my\\.example\\.com(/<unresolved>)?:443$");
  }

  @Test
  public void testCreateIfNeededNoProxyLowerCase() throws Exception {
    ProxyHelper helper =
        new ProxyHelper(
            ImmutableMap.of(
                "no_proxy",
                "something.com,example.com,localhost",
                "HTTPS_PROXY",
                "https://my.example.com"));
    ProxyInfo proxyInfo = helper.createProxyIfNeeded(URI.create("https://www.example.com"));
    assertThat(proxyInfo.proxy()).isEqualTo(Proxy.NO_PROXY);
  }

  @Test
  public void testCreateIfNeededNoProxyUpperCase() throws Exception {
    ProxyHelper helper =
        new ProxyHelper(
            ImmutableMap.of(
                "NO_PROXY",
                "something.com,example.com,localhost",
                "HTTPS_PROXY",
                "https://my.example.com"));
    ProxyInfo proxyInfo = helper.createProxyIfNeeded(URI.create("https://www.example.com"));
    assertThat(proxyInfo.proxy()).isEqualTo(Proxy.NO_PROXY);
  }

  @Test
  public void testCreateIfNeededMultipleNoProxyLowerCase() throws Exception {
    ProxyHelper helper =
        new ProxyHelper(
            ImmutableMap.of(
                "no_proxy",
                "something.com,example.com,localhost",
                "HTTPS_PROXY",
                "https://my.example.com"));
    ProxyInfo proxyInfo = helper.createProxyIfNeeded(URI.create("https://www.example.com"));
    assertThat(proxyInfo.proxy()).isEqualTo(Proxy.NO_PROXY);
  }

  @Test
  public void testCreateIfNeededMultipleNoProxyUpperCase() throws Exception {
    ProxyHelper helper =
        new ProxyHelper(
            ImmutableMap.of(
                "NO_PROXY",
                "something.com,example.com,localhost",
                "HTTPS_PROXY",
                "https://my.example.com"));
    ProxyInfo proxyInfo = helper.createProxyIfNeeded(URI.create("https://www.example.com"));
    assertThat(proxyInfo.proxy()).isEqualTo(Proxy.NO_PROXY);
  }

  @Test
  public void testCreateIfNeededMultipleNoProxySpaces() throws Exception {
    ProxyHelper helper =
        new ProxyHelper(
            ImmutableMap.of(
                "no_proxy",
                "something.com ,   example.com, localhost",
                "HTTPS_PROXY",
                "https://my.example.com"));
    ProxyInfo proxyInfo = helper.createProxyIfNeeded(URI.create("https://www.something.com"));
    assertThat(proxyInfo.proxy()).isEqualTo(Proxy.NO_PROXY);

    ProxyInfo proxyInfo2 = helper.createProxyIfNeeded(URI.create("https://www.example.com"));
    assertThat(proxyInfo2.proxy()).isEqualTo(Proxy.NO_PROXY);

    ProxyInfo proxyInfo3 = helper.createProxyIfNeeded(URI.create("https://localhost"));
    assertThat(proxyInfo3.proxy()).isEqualTo(Proxy.NO_PROXY);
  }

  @Test
  public void testCreateIfNeededNoProxyNoMatchSubstring() throws Exception {
    ProxyHelper helper =
        new ProxyHelper(
            ImmutableMap.of(
                "NO_PROXY",
                "something.com,example.com,localhost",
                "HTTPS_PROXY",
                "https://my.example.com"));
    ProxyInfo proxyInfo = helper.createProxyIfNeeded(URI.create("https://www.not-example.com"));
    assertThat(proxyInfo.proxy().toString())
        .containsMatch("my\\.example\\.com(/<unresolved>)?:443$");
  }

  @Test
  public void testCreateIfNeededNoProxyMatchSubdomainInNoProxy() throws Exception {
    ProxyHelper helper =
        new ProxyHelper(
            ImmutableMap.of(
                "NO_PROXY",
                ".something.com,example.com,localhost",
                "HTTPS_PROXY",
                "https://my.example.com"));
    ProxyInfo proxyInfo = helper.createProxyIfNeeded(URI.create("https://www.my.something.com"));
    assertThat(proxyInfo.proxy()).isEqualTo(Proxy.NO_PROXY);
  }

  @Test
  public void testCreateIfNeededNoProxyMatchSubdomainInURL() throws Exception {
    ProxyHelper helper =
        new ProxyHelper(
            ImmutableMap.of(
                "NO_PROXY",
                "something.com,example.com,localhost",
                "HTTPS_PROXY",
                "https://my.example.com"));
    ProxyInfo proxyInfo =
        helper.createProxyIfNeeded(URI.create("https://www.my.subdomain.something.com"));
    assertThat(proxyInfo.proxy()).isEqualTo(Proxy.NO_PROXY);
  }

  @Test
  public void testNoProxy() throws Exception {
    // Empty address.
    ProxyInfo proxyInfo = ProxyHelper.createProxy(null);
    assertThat(proxyInfo.proxy()).isEqualTo(Proxy.NO_PROXY);
    proxyInfo = ProxyHelper.createProxy("");
    assertThat(proxyInfo.proxy()).isEqualTo(Proxy.NO_PROXY);
    Map<String, String> env = ImmutableMap.of();
    ProxyHelper helper = new ProxyHelper(env);
    proxyInfo = helper.createProxyIfNeeded(URI.create("https://www.something.com"));
    assertThat(proxyInfo.proxy()).isEqualTo(Proxy.NO_PROXY);
  }

  @Test
  public void testProxyDefaultPort() throws Exception {
    ProxyInfo proxyInfo = ProxyHelper.createProxy("http://my.example.com");
    assertThat(proxyInfo.proxy().type()).isEqualTo(Proxy.Type.HTTP);
    assertThat(proxyInfo.proxy().toString()).endsWith(":80");

    proxyInfo = ProxyHelper.createProxy("https://my.example.com");
    assertThat(proxyInfo.proxy().toString()).endsWith(":443");
  }

  @Test
  public void testProxyExplicitPort() throws Exception {
    ProxyInfo proxyInfo = ProxyHelper.createProxy("http://my.example.com:12345");
    assertThat(proxyInfo.proxy().toString()).endsWith(":12345");

    proxyInfo = ProxyHelper.createProxy("https://my.example.com:12345");
    assertThat(proxyInfo.proxy().toString()).endsWith(":12345");
  }

  @Test
  public void testProxyNoProtocol() throws Exception {
    ProxyInfo proxyInfo = ProxyHelper.createProxy("my.example.com");
    assertThat(proxyInfo.proxy().toString()).endsWith(":80");
  }

  @Test
  public void testProxyNoProtocolWithPort() throws Exception {
    ProxyInfo proxyInfo = ProxyHelper.createProxy("my.example.com:12345");
    assertThat(proxyInfo.proxy().toString()).endsWith(":12345");
  }

  @Test
  public void testProxyPortParsingError() throws Exception {
    IOException e =
        assertThrows(IOException.class, () -> ProxyHelper.createProxy("http://my.example.com:foo"));
    assertThat(e)
        .hasMessageThat()
        .contains("Proxy address http://my.example.com:foo is not a valid URL");
  }

  @Test
  public void testProxyAuth() throws Exception {
    ProxyInfo proxyInfo = ProxyHelper.createProxy("http://foo:barbaz@my.example.com");
    assertThat(proxyInfo.proxy().type()).isEqualTo(Proxy.Type.HTTP);
    assertThat(proxyInfo.proxy().toString()).endsWith(":80");

    proxyInfo = ProxyHelper.createProxy("https://biz:bat@my.example.com");
    assertThat(proxyInfo.proxy().toString()).endsWith(":443");
  }

  @Test
  public void testEncodedProxyAuth() throws Exception {
    ProxyInfo proxyInfo = ProxyHelper.createProxy("http://foo:b%40rb%40z@my.example.com");
    assertThat(proxyInfo.proxy().type()).isEqualTo(Proxy.Type.HTTP);
    assertThat(proxyInfo.proxy().toString()).endsWith(":80");
  }

  @Test
  public void testInvalidAuth() throws Exception {
    IOException e =
        assertThrows(IOException.class, () -> ProxyHelper.createProxy("http://foo@my.example.com"));
    assertThat(e).hasMessageThat().contains("No password given for proxy");
  }

  @Test
  public void testNoProxyAuth() throws Exception {
    ProxyInfo proxyInfo = ProxyHelper.createProxy("http://localhost:3128/");
    assertThat(proxyInfo.proxy().type()).isEqualTo(Proxy.Type.HTTP);
    assertThat(proxyInfo.proxy().toString()).endsWith(":3128");
  }

  @Test
  public void testTrailingSlash() throws Exception {
    ProxyInfo proxyInfo = ProxyHelper.createProxy("http://foo:bar@example.com:8000/");
    assertThat(proxyInfo.proxy().type()).isEqualTo(Proxy.Type.HTTP);
    assertThat(proxyInfo.proxy().toString()).endsWith(":8000");
  }

  // Tests for ProxyInfo credentials

  @Test
  public void testProxyInfoWithCredentials() throws Exception {
    ProxyInfo proxyInfo = ProxyHelper.createProxy("http://myuser:mypass@proxy.example.com:8080");
    assertThat(proxyInfo.hasCredentials()).isTrue();
    assertThat(proxyInfo.getProxyAuthorizationHeader()).isNotNull();
    // Verify it's a valid Basic auth header
    assertThat(proxyInfo.getProxyAuthorizationHeader()).startsWith("Basic ");
    // Decode and verify the credentials
    String encoded = proxyInfo.getProxyAuthorizationHeader().substring("Basic ".length());
    String decoded = new String(Base64.getDecoder().decode(encoded), UTF_8);
    assertThat(decoded).isEqualTo("myuser:mypass");
  }

  @Test
  public void testProxyInfoWithUrlEncodedCredentials() throws Exception {
    // Password contains @ and : which are URL-encoded
    ProxyInfo proxyInfo =
        ProxyHelper.createProxy("http://user:p%40ss%3Aword@proxy.example.com:8080");
    assertThat(proxyInfo.hasCredentials()).isTrue();
    String encoded = proxyInfo.getProxyAuthorizationHeader().substring("Basic ".length());
    String decoded = new String(Base64.getDecoder().decode(encoded), Charset.defaultCharset());
    // URL-encoded characters should be decoded
    assertThat(decoded).isEqualTo("user:p@ss:word");
  }

  @Test
  public void testProxyInfoWithUrlEncodedUsername() throws Exception {
    // Username contains @ which is URL-encoded
    ProxyInfo proxyInfo =
        ProxyHelper.createProxy("http://user%40domain:password@proxy.example.com:8080");
    assertThat(proxyInfo.hasCredentials()).isTrue();
    String encoded = proxyInfo.getProxyAuthorizationHeader().substring("Basic ".length());
    String decoded = new String(Base64.getDecoder().decode(encoded), UTF_8);
    // URL-encoded characters in username should be decoded
    assertThat(decoded).isEqualTo("user@domain:password");
  }

  @Test
  public void testProxyInfoWithUnicodeCredentials() throws Exception {
    // Test Unicode characters in username and password
    // Username: "用户" (Chinese for "user") = %E7%94%A8%E6%88%B7
    // Password: "contraseña" (Spanish with ñ) = contrase%C3%B1a
    ProxyInfo proxyInfo =
        ProxyHelper.createProxy("http://%E7%94%A8%E6%88%B7:contrase%C3%B1a@proxy.example.com:8080");
    assertThat(proxyInfo.hasCredentials()).isTrue();
    String encoded = proxyInfo.getProxyAuthorizationHeader().substring("Basic ".length());
    String decoded = new String(Base64.getDecoder().decode(encoded), StandardCharsets.UTF_8);
    assertThat(decoded).isEqualTo("用户:contraseña");
  }

  @Test
  public void testProxyInfoWithoutCredentials() throws Exception {
    ProxyInfo proxyInfo = ProxyHelper.createProxy("http://proxy.example.com:8080");
    assertThat(proxyInfo.hasCredentials()).isFalse();
    assertThat(proxyInfo.getProxyAuthorizationHeader()).isNull();
  }

  @Test
  public void testProxyInfoNoProxyHasNoCredentials() throws Exception {
    assertThat(ProxyInfo.NO_PROXY.hasCredentials()).isFalse();
    assertThat(ProxyInfo.NO_PROXY.getProxyAuthorizationHeader()).isNull();
    assertThat(ProxyInfo.NO_PROXY.proxy()).isEqualTo(Proxy.NO_PROXY);
  }

  @Test
  public void testProxyInfoFromSystemProperties() throws Exception {
    // Test that credentials can be provided via the systemPropertyUser/Password parameters
    ProxyInfo proxyInfo =
        ProxyHelper.createProxyInfo("http://proxy.example.com:8080", "sysuser", "syspass");
    assertThat(proxyInfo.hasCredentials()).isTrue();
    String encoded = proxyInfo.getProxyAuthorizationHeader().substring("Basic ".length());
    String decoded = new String(Base64.getDecoder().decode(encoded), UTF_8);
    assertThat(decoded).isEqualTo("sysuser:syspass");
  }

  @Test
  public void testProxyInfoUrlCredentialsTakePrecedence() throws Exception {
    // URL credentials should take precedence over system property credentials
    ProxyInfo proxyInfo =
        ProxyHelper.createProxyInfo(
            "http://urluser:urlpass@proxy.example.com:8080", "sysuser", "syspass");
    assertThat(proxyInfo.hasCredentials()).isTrue();
    String encoded = proxyInfo.getProxyAuthorizationHeader().substring("Basic ".length());
    String decoded = new String(Base64.getDecoder().decode(encoded), UTF_8);
    assertThat(decoded).isEqualTo("urluser:urlpass");
  }

  @Test
  public void testProxyInfoSystemPropertiesOnlyUserIgnored() throws Exception {
    // If only username is provided via system properties (no password), credentials should not be
    // set
    ProxyInfo proxyInfo =
        ProxyHelper.createProxyInfo("http://proxy.example.com:8080", "sysuser", null);
    assertThat(proxyInfo.hasCredentials()).isFalse();
  }

  @Test
  public void testProxyInfoSystemPropertiesOnlyPasswordIgnored() throws Exception {
    // If only password is provided via system properties (no username), credentials should not be
    // set
    ProxyInfo proxyInfo =
        ProxyHelper.createProxyInfo("http://proxy.example.com:8080", null, "syspass");
    assertThat(proxyInfo.hasCredentials()).isFalse();
  }

  // Tests for http.nonProxyHosts system property

  @Test
  public void testNonProxyHostsExactMatch() throws Exception {
    String oldValue = System.getProperty("http.nonProxyHosts");
    try {
      System.setProperty("http.nonProxyHosts", "localhost|example.com");
      ProxyHelper helper = new ProxyHelper(ImmutableMap.of("http_proxy", "http://proxy:8080"));

      // Exact match should bypass proxy
      ProxyInfo proxyInfo = helper.createProxyIfNeeded(URI.create("http://example.com/foo"));
      assertThat(proxyInfo.proxy()).isEqualTo(Proxy.NO_PROXY);

      // Non-match should use proxy
      ProxyInfo proxyInfo2 = helper.createProxyIfNeeded(URI.create("http://other.com/foo"));
      assertThat(proxyInfo2.proxy()).isNotEqualTo(Proxy.NO_PROXY);
    } finally {
      if (oldValue != null) {
        System.setProperty("http.nonProxyHosts", oldValue);
      } else {
        System.clearProperty("http.nonProxyHosts");
      }
    }
  }

  @Test
  public void testNonProxyHostsWildcardPrefix() throws Exception {
    String oldValue = System.getProperty("http.nonProxyHosts");
    try {
      System.setProperty("http.nonProxyHosts", "*.example.com");
      ProxyHelper helper = new ProxyHelper(ImmutableMap.of("http_proxy", "http://proxy:8080"));

      // Wildcard match should bypass proxy
      ProxyInfo proxyInfo = helper.createProxyIfNeeded(URI.create("http://foo.example.com/bar"));
      assertThat(proxyInfo.proxy()).isEqualTo(Proxy.NO_PROXY);

      // Non-match should use proxy
      ProxyInfo proxyInfo2 = helper.createProxyIfNeeded(URI.create("http://example.com/bar"));
      assertThat(proxyInfo2.proxy()).isNotEqualTo(Proxy.NO_PROXY);
    } finally {
      if (oldValue != null) {
        System.setProperty("http.nonProxyHosts", oldValue);
      } else {
        System.clearProperty("http.nonProxyHosts");
      }
    }
  }

  @Test
  public void testNonProxyHostsWildcardSuffix() throws Exception {
    String oldValue = System.getProperty("http.nonProxyHosts");
    try {
      System.setProperty("http.nonProxyHosts", "local*");
      ProxyHelper helper = new ProxyHelper(ImmutableMap.of("http_proxy", "http://proxy:8080"));

      // Wildcard match should bypass proxy
      ProxyInfo proxyInfo = helper.createProxyIfNeeded(URI.create("http://localhost/bar"));
      assertThat(proxyInfo.proxy()).isEqualTo(Proxy.NO_PROXY);

      ProxyInfo proxyInfo2 = helper.createProxyIfNeeded(URI.create("http://localserver/bar"));
      assertThat(proxyInfo2.proxy()).isEqualTo(Proxy.NO_PROXY);
    } finally {
      if (oldValue != null) {
        System.setProperty("http.nonProxyHosts", oldValue);
      } else {
        System.clearProperty("http.nonProxyHosts");
      }
    }
  }
}
