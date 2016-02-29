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

package com.google.devtools.build.lib.bazel.repository;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.net.Proxy;

/**
 * Tests for @{link ProxyHelper}.
 */
@RunWith(JUnit4.class)
public class ProxyHelperTest {

  @Test
  public void testNoProxy() throws Exception {
    // Empty address.
    Proxy proxy = ProxyHelper.createProxy(null);
    assertEquals(Proxy.NO_PROXY, proxy);
    proxy = ProxyHelper.createProxy("");
    assertEquals(Proxy.NO_PROXY, proxy);
  }

  @Test
  public void testProxyDefaultPort() throws Exception {
    Proxy proxy = ProxyHelper.createProxy("http://my.example.com");
    assertEquals(Proxy.Type.HTTP, proxy.type());
    assertThat(proxy.toString()).endsWith(":80");
    assertEquals(System.getProperty("http.proxyHost"), "my.example.com");
    assertEquals(System.getProperty("http.proxyPort"), "80");

    proxy = ProxyHelper.createProxy("https://my.example.com");
    assertThat(proxy.toString()).endsWith(":443");
    assertEquals(System.getProperty("https.proxyHost"), "my.example.com");
    assertEquals(System.getProperty("https.proxyPort"), "443");
  }

  @Test
  public void testProxyExplicitPort() throws Exception {
    Proxy proxy = ProxyHelper.createProxy("http://my.example.com:12345");
    assertThat(proxy.toString()).endsWith(":12345");
    assertEquals(System.getProperty("http.proxyHost"), "my.example.com");
    assertEquals(System.getProperty("http.proxyPort"), "12345");

    proxy = ProxyHelper.createProxy("https://my.example.com:12345");
    assertThat(proxy.toString()).endsWith(":12345");
    assertEquals(System.getProperty("https.proxyHost"), "my.example.com");
    assertEquals(System.getProperty("https.proxyPort"), "12345");
  }

  @Test
  public void testProxyNoProtocol() throws Exception {
    try {
      ProxyHelper.createProxy("my.example.com");
      fail("Expected protocol error");
    } catch (IOException e) {
      assertThat(e.getMessage()).contains("Proxy address my.example.com is not a valid URL");
    }
  }

  @Test
  public void testProxyNoProtocolWithPort() throws Exception {
    try {
      ProxyHelper.createProxy("my.example.com:12345");
      fail("Expected protocol error");
    } catch (IOException e) {
      assertThat(e.getMessage()).contains("Proxy address my.example.com:12345 is not a valid URL");
    }
  }

  @Test
  public void testProxyPortParsingError() throws Exception {
    try {
      ProxyHelper.createProxy("http://my.example.com:foo");
      fail("Should have thrown an error for invalid port");
    } catch (IOException e) {
      assertThat(e.getMessage())
          .contains("Proxy address http://my.example.com:foo is not a valid URL");
    }
  }

  @Test
  public void testProxyAuth() throws Exception {
    Proxy proxy = ProxyHelper.createProxy("http://foo:barbaz@my.example.com");
    assertEquals(Proxy.Type.HTTP, proxy.type());
    assertThat(proxy.toString()).endsWith(":80");
    assertEquals(System.getProperty("http.proxyHost"), "my.example.com");
    assertEquals(System.getProperty("http.proxyPort"), "80");
    assertEquals(System.getProperty("http.proxyUser"), "foo");
    assertEquals(System.getProperty("http.proxyPassword"), "barbaz");

    proxy = ProxyHelper.createProxy("https://biz:bat@my.example.com");
    assertThat(proxy.toString()).endsWith(":443");
    assertEquals(System.getProperty("https.proxyHost"), "my.example.com");
    assertEquals(System.getProperty("https.proxyPort"), "443");
    assertEquals(System.getProperty("https.proxyUser"), "biz");
    assertEquals(System.getProperty("https.proxyPassword"), "bat");
  }

  @Test
  public void testEncodedProxyAuth() throws Exception {
    Proxy proxy = ProxyHelper.createProxy("http://foo:b%40rb%40z@my.example.com");
    assertEquals(Proxy.Type.HTTP, proxy.type());
    assertThat(proxy.toString()).endsWith(":80");
    assertEquals(System.getProperty("http.proxyHost"), "my.example.com");
    assertEquals(System.getProperty("http.proxyPort"), "80");
    assertEquals(System.getProperty("http.proxyUser"), "foo");
    assertEquals(System.getProperty("http.proxyPassword"), "b@rb@z");
  }

  @Test
  public void testInvalidAuth() throws Exception {
    try {
      ProxyHelper.createProxy("http://foo@my.example.com");
      fail("Should have thrown an error for invalid auth");
    } catch (IOException e) {
      assertThat(e.getMessage()).contains("No password given for proxy");
    }
  }
}
