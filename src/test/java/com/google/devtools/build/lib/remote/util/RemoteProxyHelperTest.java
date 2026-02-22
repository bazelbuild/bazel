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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.remote.util.RemoteProxyHelper.ProxyInfo;
import java.io.IOException;
import java.net.URI;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RemoteProxyHelper}. */
@RunWith(JUnit4.class)
public class RemoteProxyHelperTest {

  @Test
  public void parseProxyAddress_httpProxy() throws IOException {
    ProxyInfo proxyInfo = RemoteProxyHelper.parseProxyAddress("http://proxy.example.com:8080");

    assertThat(proxyInfo.hasProxy()).isTrue();
    assertThat(proxyInfo.address().getHostName()).isEqualTo("proxy.example.com");
    assertThat(proxyInfo.address().getPort()).isEqualTo(8080);
    assertThat(proxyInfo.hasCredentials()).isFalse();
    assertThat(proxyInfo.useTls()).isFalse();
  }

  @Test
  public void parseProxyAddress_httpsProxy() throws IOException {
    ProxyInfo proxyInfo = RemoteProxyHelper.parseProxyAddress("https://proxy.example.com:443");

    assertThat(proxyInfo.hasProxy()).isTrue();
    assertThat(proxyInfo.address().getHostName()).isEqualTo("proxy.example.com");
    assertThat(proxyInfo.address().getPort()).isEqualTo(443);
    assertThat(proxyInfo.useTls()).isTrue();
  }

  @Test
  public void parseProxyAddress_withCredentials() throws IOException {
    ProxyInfo proxyInfo =
        RemoteProxyHelper.parseProxyAddress("http://user:password@proxy.example.com:8080");

    assertThat(proxyInfo.hasProxy()).isTrue();
    assertThat(proxyInfo.address().getHostName()).isEqualTo("proxy.example.com");
    assertThat(proxyInfo.address().getPort()).isEqualTo(8080);
    assertThat(proxyInfo.hasCredentials()).isTrue();
    assertThat(proxyInfo.username()).isEqualTo("user");
    assertThat(proxyInfo.password()).isEqualTo("password");
  }

  @Test
  public void parseProxyAddress_withUrlEncodedCredentials() throws IOException {
    // URL-encoded special characters in password
    ProxyInfo proxyInfo =
        RemoteProxyHelper.parseProxyAddress("http://user:p%40ss%3Aword@proxy.example.com:8080");

    assertThat(proxyInfo.hasProxy()).isTrue();
    assertThat(proxyInfo.hasCredentials()).isTrue();
    assertThat(proxyInfo.username()).isEqualTo("user");
    assertThat(proxyInfo.password()).isEqualTo("p@ss:word");
  }

  @Test
  public void parseProxyAddress_defaultHttpPort() throws IOException {
    ProxyInfo proxyInfo = RemoteProxyHelper.parseProxyAddress("http://proxy.example.com");

    assertThat(proxyInfo.hasProxy()).isTrue();
    assertThat(proxyInfo.address().getPort()).isEqualTo(80);
  }

  @Test
  public void parseProxyAddress_defaultHttpsPort() throws IOException {
    ProxyInfo proxyInfo = RemoteProxyHelper.parseProxyAddress("https://proxy.example.com");

    assertThat(proxyInfo.hasProxy()).isTrue();
    assertThat(proxyInfo.address().getPort()).isEqualTo(443);
  }

  @Test
  public void parseProxyAddress_noScheme() throws IOException {
    ProxyInfo proxyInfo = RemoteProxyHelper.parseProxyAddress("proxy.example.com:8080");

    assertThat(proxyInfo.hasProxy()).isTrue();
    assertThat(proxyInfo.address().getHostName()).isEqualTo("proxy.example.com");
    assertThat(proxyInfo.address().getPort()).isEqualTo(8080);
  }

  @Test
  public void parseProxyAddress_nullReturnsNoProxy() throws IOException {
    ProxyInfo proxyInfo = RemoteProxyHelper.parseProxyAddress(null);

    assertThat(proxyInfo.hasProxy()).isFalse();
    assertThat(proxyInfo).isEqualTo(ProxyInfo.NO_PROXY);
  }

  @Test
  public void parseProxyAddress_emptyReturnsNoProxy() throws IOException {
    ProxyInfo proxyInfo = RemoteProxyHelper.parseProxyAddress("");

    assertThat(proxyInfo.hasProxy()).isFalse();
  }

  @Test
  public void parseProxyAddress_invalidScheme() {
    assertThrows(
        IOException.class,
        () -> RemoteProxyHelper.parseProxyAddress("ftp://proxy.example.com:21"));
  }

  @Test
  public void parseProxyAddress_usernameWithoutPassword() {
    assertThrows(
        IOException.class,
        () -> RemoteProxyHelper.parseProxyAddress("http://user@proxy.example.com:8080"));
  }

  @Test
  public void createProxyIfNeeded_httpsProxy() throws IOException {
    RemoteProxyHelper helper =
        new RemoteProxyHelper(ImmutableMap.of("HTTPS_PROXY", "http://proxy.example.com:8080"));

    ProxyInfo proxyInfo = helper.createProxyIfNeeded(URI.create("https://cache.example.com"));

    assertThat(proxyInfo.hasProxy()).isTrue();
    assertThat(proxyInfo.address().getHostName()).isEqualTo("proxy.example.com");
    assertThat(proxyInfo.address().getPort()).isEqualTo(8080);
  }

  @Test
  public void createProxyIfNeeded_httpProxy() throws IOException {
    RemoteProxyHelper helper =
        new RemoteProxyHelper(ImmutableMap.of("HTTP_PROXY", "http://proxy.example.com:8080"));

    ProxyInfo proxyInfo = helper.createProxyIfNeeded(URI.create("http://cache.example.com"));

    assertThat(proxyInfo.hasProxy()).isTrue();
    assertThat(proxyInfo.address().getHostName()).isEqualTo("proxy.example.com");
  }

  @Test
  public void createProxyIfNeeded_lowercaseEnvVar() throws IOException {
    RemoteProxyHelper helper =
        new RemoteProxyHelper(ImmutableMap.of("https_proxy", "http://proxy.example.com:8080"));

    ProxyInfo proxyInfo = helper.createProxyIfNeeded(URI.create("https://cache.example.com"));

    assertThat(proxyInfo.hasProxy()).isTrue();
  }

  @Test
  public void createProxyIfNeeded_noProxyExcludesHost() throws IOException {
    RemoteProxyHelper helper =
        new RemoteProxyHelper(
            ImmutableMap.of(
                "HTTPS_PROXY", "http://proxy.example.com:8080",
                "NO_PROXY", "cache.example.com"));

    ProxyInfo proxyInfo = helper.createProxyIfNeeded(URI.create("https://cache.example.com"));

    assertThat(proxyInfo.hasProxy()).isFalse();
  }

  @Test
  public void createProxyIfNeeded_noProxyWithSubdomain() throws IOException {
    RemoteProxyHelper helper =
        new RemoteProxyHelper(
            ImmutableMap.of(
                "HTTPS_PROXY", "http://proxy.example.com:8080",
                "NO_PROXY", ".example.com"));

    ProxyInfo proxyInfo = helper.createProxyIfNeeded(URI.create("https://cache.example.com"));

    assertThat(proxyInfo.hasProxy()).isFalse();
  }

  @Test
  public void createProxyIfNeeded_noProxyMultipleHosts() throws IOException {
    RemoteProxyHelper helper =
        new RemoteProxyHelper(
            ImmutableMap.of(
                "HTTPS_PROXY", "http://proxy.example.com:8080",
                "NO_PROXY", "other.com,cache.example.com,another.com"));

    ProxyInfo proxyInfo = helper.createProxyIfNeeded(URI.create("https://cache.example.com"));

    assertThat(proxyInfo.hasProxy()).isFalse();
  }

  @Test
  public void createProxyIfNeeded_noEnvVar() throws IOException {
    RemoteProxyHelper helper = new RemoteProxyHelper(ImmutableMap.of());

    ProxyInfo proxyInfo = helper.createProxyIfNeeded(URI.create("https://cache.example.com"));

    assertThat(proxyInfo.hasProxy()).isFalse();
  }

  @Test
  public void createProxyFromFlag_unixSocket() throws IOException {
    ProxyInfo proxyInfo = RemoteProxyHelper.createProxyFromFlag("unix:/path/to/socket");

    // Unix socket proxies return null (they're handled separately)
    assertThat(proxyInfo).isNull();
  }

  @Test
  public void createProxyFromFlag_httpProxy() throws IOException {
    ProxyInfo proxyInfo = RemoteProxyHelper.createProxyFromFlag("http://proxy.example.com:8080");

    assertThat(proxyInfo).isNotNull();
    assertThat(proxyInfo.hasProxy()).isTrue();
  }
}
