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

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link HttpUtils}. */
@RunWith(JUnit4.class)
public class HttpUtilsTest {

  @Rule
  public final ExpectedException thrown = ExpectedException.none();

  private final HttpURLConnection connection = mock(HttpURLConnection.class);

  @Test
  public void getExtension_twoExtensions_returnsLast() throws Exception {
    assertThat(HttpUtils.getExtension("doodle.tar.gz")).isEqualTo("gz");
  }

  @Test
  public void getExtension_isUppercase_returnsLowered() throws Exception {
    assertThat(HttpUtils.getExtension("DOODLE.TXT")).isEqualTo("txt");
  }

  @Test
  public void getLocation_missingInRedirect_throwsIOException() throws Exception {
    thrown.expect(IOException.class);
    when(connection.getURL()).thenReturn(new URL("http://lol.example"));
    HttpUtils.getLocation(connection);
  }

  @Test
  public void getLocation_absoluteInRedirect_returnsNewUrl() throws Exception {
    when(connection.getURL()).thenReturn(new URL("http://lol.example"));
    when(connection.getHeaderField("Location")).thenReturn("http://new.example/hi");
    assertThat(HttpUtils.getLocation(connection)).isEqualTo(new URL("http://new.example/hi"));
  }

  @Test
  public void getLocation_redirectOnlyHasPath_mergesHostFromOriginalUrl() throws Exception {
    when(connection.getURL()).thenReturn(new URL("http://lol.example"));
    when(connection.getHeaderField("Location")).thenReturn("/hi");
    assertThat(HttpUtils.getLocation(connection)).isEqualTo(new URL("http://lol.example/hi"));
  }

  @Test
  public void getLocation_onlyHasPathWithoutSlash_failsToMerge() throws Exception {
    thrown.expect(IOException.class);
    thrown.expectMessage("Could not merge");
    when(connection.getURL()).thenReturn(new URL("http://lol.example"));
    when(connection.getHeaderField("Location")).thenReturn("omg");
    HttpUtils.getLocation(connection);
  }

  @Test
  public void getLocation_hasFragment_prefersNewFragment() throws Exception {
    when(connection.getURL()).thenReturn(new URL("http://lol.example#a"));
    when(connection.getHeaderField("Location")).thenReturn("http://new.example/hi#b");
    assertThat(HttpUtils.getLocation(connection)).isEqualTo(new URL("http://new.example/hi#b"));
  }

  @Test
  public void getLocation_hasNoFragmentButOriginalDoes_mergesOldFragment() throws Exception {
    when(connection.getURL()).thenReturn(new URL("http://lol.example#a"));
    when(connection.getHeaderField("Location")).thenReturn("http://new.example/hi");
    assertThat(HttpUtils.getLocation(connection)).isEqualTo(new URL("http://new.example/hi#a"));
  }

  @Test
  public void getLocation_oldUrlHasPassRedirectingToSameDomain_mergesPassword() throws Exception {
    when(connection.getURL()).thenReturn(new URL("http://a:b@lol.example"));
    when(connection.getHeaderField("Location")).thenReturn("http://lol.example/hi");
    assertThat(HttpUtils.getLocation(connection))
        .isEqualTo(new URL("http://a:b@lol.example/hi"));
    when(connection.getURL()).thenReturn(new URL("http://a:b@lol.example"));
    when(connection.getHeaderField("Location")).thenReturn("/hi");
    assertThat(HttpUtils.getLocation(connection))
        .isEqualTo(new URL("http://a:b@lol.example/hi"));
  }

  @Test
  public void getLocation_oldUrlHasPasswordRedirectingToNewServer_doesntMerge() throws Exception {
    when(connection.getURL()).thenReturn(new URL("http://a:b@lol.example"));
    when(connection.getHeaderField("Location")).thenReturn("http://new.example/hi");
    assertThat(HttpUtils.getLocation(connection)).isEqualTo(new URL("http://new.example/hi"));
    when(connection.getURL()).thenReturn(new URL("http://a:b@lol.example"));
    when(connection.getHeaderField("Location")).thenReturn("http://lol.example:81/hi");
    assertThat(HttpUtils.getLocation(connection))
        .isEqualTo(new URL("http://lol.example:81/hi"));
  }

  @Test
  public void getLocation_redirectToFtp_throwsIOException() throws Exception {
    thrown.expect(IOException.class);
    thrown.expectMessage("Bad Location");
    when(connection.getURL()).thenReturn(new URL("http://lol.example"));
    when(connection.getHeaderField("Location")).thenReturn("ftp://lol.example");
    HttpUtils.getLocation(connection);
  }

  @Test
  public void getLocation_redirectToHttps_works() throws Exception {
    when(connection.getURL()).thenReturn(new URL("http://lol.example"));
    when(connection.getHeaderField("Location")).thenReturn("https://lol.example");
    assertThat(HttpUtils.getLocation(connection)).isEqualTo(new URL("https://lol.example"));
  }

  @Test
  public void getLocation_preservesQuotingIfNotInheriting() throws Exception {
    String redirect =
        "http://redirected.example.org/foo?"
            + "response-content-disposition=attachment%3Bfilename%3D%22bar.tar.gz%22";
    when(connection.getURL()).thenReturn(new URL("http://original.example.org"));
    when(connection.getHeaderField("Location")).thenReturn(redirect);
    assertThat(HttpUtils.getLocation(connection)).isEqualTo(URI.create(redirect).toURL());
  }
}
