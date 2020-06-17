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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import java.io.IOException;
import java.net.Proxy;
import java.net.URL;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link ProxyHelper}.
 */
@RunWith(JUnit4.class)
public class ProxyHelperTest {

  @Test
  public void testCreateIfNeededHttpLowerCase() throws Exception {
    ProxyHelper helper = new ProxyHelper(ImmutableMap.of("http_proxy", "http://my.example.com"));
    Proxy proxy = helper.createProxyIfNeeded(new URL("http://www.something.com"));
    assertThat(proxy.toString()).endsWith("my.example.com:80");
  }

  @Test
  public void testCreateIfNeededHttpUpperCase() throws Exception {
    ProxyHelper helper = new ProxyHelper(ImmutableMap.of("HTTP_PROXY", "http://my.example.com"));
    Proxy proxy = helper.createProxyIfNeeded(new URL("http://www.something.com"));
    assertThat(proxy.toString()).endsWith("my.example.com:80");
  }

  @Test
  public void testCreateIfNeededHttpsLowerCase() throws Exception {
    ProxyHelper helper = new ProxyHelper(ImmutableMap.of("https_proxy", "https://my.example.com"));
    Proxy proxy = helper.createProxyIfNeeded(new URL("https://www.something.com"));
    assertThat(proxy.toString()).endsWith("my.example.com:443");
  }

  @Test
  public void testCreateIfNeededHttpsUpperCase() throws Exception {
    ProxyHelper helper = new ProxyHelper(ImmutableMap.of("HTTPS_PROXY", "https://my.example.com"));
    Proxy proxy = helper.createProxyIfNeeded(new URL("https://www.something.com"));
    assertThat(proxy.toString()).endsWith("my.example.com:443");
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
    Proxy proxy = helper.createProxyIfNeeded(new URL("https://www.example.com"));
    assertThat(proxy).isEqualTo(Proxy.NO_PROXY);
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
    Proxy proxy = helper.createProxyIfNeeded(new URL("https://www.example.com"));
    assertThat(proxy).isEqualTo(Proxy.NO_PROXY);
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
    Proxy proxy = helper.createProxyIfNeeded(new URL("https://www.example.com"));
    assertThat(proxy).isEqualTo(Proxy.NO_PROXY);
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
    Proxy proxy = helper.createProxyIfNeeded(new URL("https://www.example.com"));
    assertThat(proxy).isEqualTo(Proxy.NO_PROXY);
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
    Proxy proxy = helper.createProxyIfNeeded(new URL("https://www.something.com"));
    assertThat(proxy).isEqualTo(Proxy.NO_PROXY);

    Proxy proxy2 = helper.createProxyIfNeeded(new URL("https://www.example.com"));
    assertThat(proxy2).isEqualTo(Proxy.NO_PROXY);

    Proxy proxy3 = helper.createProxyIfNeeded(new URL("https://localhost"));
    assertThat(proxy3).isEqualTo(Proxy.NO_PROXY);
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
    Proxy proxy = helper.createProxyIfNeeded(new URL("https://www.not-example.com"));
    assertThat(proxy.toString()).endsWith("my.example.com:443");
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
    Proxy proxy = helper.createProxyIfNeeded(new URL("https://www.my.something.com"));
    assertThat(proxy).isEqualTo(Proxy.NO_PROXY);
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
    Proxy proxy = helper.createProxyIfNeeded(new URL("https://www.my.subdomain.something.com"));
    assertThat(proxy).isEqualTo(Proxy.NO_PROXY);
  }

  @Test
  public void testNoProxy() throws Exception {
    // Empty address.
    Proxy proxy = ProxyHelper.createProxy(null);
    assertThat(proxy).isEqualTo(Proxy.NO_PROXY);
    proxy = ProxyHelper.createProxy("");
    assertThat(proxy).isEqualTo(Proxy.NO_PROXY);
    Map<String, String> env = ImmutableMap.of();
    ProxyHelper helper = new ProxyHelper(env);
    proxy = helper.createProxyIfNeeded(new URL("https://www.something.com"));
    assertThat(proxy).isEqualTo(Proxy.NO_PROXY);
  }

  @Test
  public void testProxyDefaultPort() throws Exception {
    Proxy proxy = ProxyHelper.createProxy("http://my.example.com");
    assertThat(proxy.type()).isEqualTo(Proxy.Type.HTTP);
    assertThat(proxy.toString()).endsWith(":80");

    proxy = ProxyHelper.createProxy("https://my.example.com");
    assertThat(proxy.toString()).endsWith(":443");
  }

  @Test
  public void testProxyExplicitPort() throws Exception {
    Proxy proxy = ProxyHelper.createProxy("http://my.example.com:12345");
    assertThat(proxy.toString()).endsWith(":12345");

    proxy = ProxyHelper.createProxy("https://my.example.com:12345");
    assertThat(proxy.toString()).endsWith(":12345");
  }

  @Test
  public void testProxyNoProtocol() throws Exception {
    IOException e =
        assertThrows(IOException.class, () -> ProxyHelper.createProxy("my.example.com"));
    assertThat(e).hasMessageThat().contains("Proxy address my.example.com is not a valid URL");
  }

  @Test
  public void testProxyNoProtocolWithPort() throws Exception {
    IOException e =
        assertThrows(IOException.class, () -> ProxyHelper.createProxy("my.example.com:12345"));
    assertThat(e)
        .hasMessageThat()
        .contains("Proxy address my.example.com:12345 is not a valid URL");
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
    Proxy proxy = ProxyHelper.createProxy("http://foo:barbaz@my.example.com");
    assertThat(proxy.type()).isEqualTo(Proxy.Type.HTTP);
    assertThat(proxy.toString()).endsWith(":80");

    proxy = ProxyHelper.createProxy("https://biz:bat@my.example.com");
    assertThat(proxy.toString()).endsWith(":443");
  }

  @Test
  public void testEncodedProxyAuth() throws Exception {
    Proxy proxy = ProxyHelper.createProxy("http://foo:b%40rb%40z@my.example.com");
    assertThat(proxy.type()).isEqualTo(Proxy.Type.HTTP);
    assertThat(proxy.toString()).endsWith(":80");
  }

  @Test
  public void testInvalidAuth() throws Exception {
    IOException e =
        assertThrows(IOException.class, () -> ProxyHelper.createProxy("http://foo@my.example.com"));
    assertThat(e).hasMessageThat().contains("No password given for proxy");
  }

  @Test
  public void testNoProxyAuth() throws Exception {
    Proxy proxy = ProxyHelper.createProxy("http://localhost:3128/");
    assertThat(proxy.type()).isEqualTo(Proxy.Type.HTTP);
    assertThat(proxy.toString()).endsWith(":3128");
  }

  @Test
  public void testTrailingSlash() throws Exception {
    Proxy proxy = ProxyHelper.createProxy("http://foo:bar@example.com:8000/");
    assertThat(proxy.type()).isEqualTo(Proxy.Type.HTTP);
    assertThat(proxy.toString()).endsWith(":8000");
  }
}
