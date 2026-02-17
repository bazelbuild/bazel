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

import java.io.IOException;
import java.net.URI;
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
    HttpUtils.getLocation(URI.create("http://lol.example"), null);
  }

  @Test
  public void getLocation_absoluteInRedirect_returnsNewUrl() throws Exception {
    assertThat(HttpUtils.getLocation(URI.create("http://lol.example"), "http://new.example/hi"))
        .isEqualTo(URI.create("http://new.example/hi"));
  }

  @Test
  public void getLocation_redirectOnlyHasPath_mergesHostFromOriginalUrl() throws Exception {
    assertThat(HttpUtils.getLocation(URI.create("http://lol.example"), "/hi"))
        .isEqualTo(URI.create("http://lol.example/hi"));
  }

  @Test
  public void getLocation_onlyHasPathWithoutSlash_failsToMerge() throws Exception {
    thrown.expect(IOException.class);
    thrown.expectMessage("Could not merge");
    HttpUtils.getLocation(URI.create("http://lol.example"), "omg");
  }

  @Test
  public void getLocation_hasFragment_prefersNewFragment() throws Exception {
    assertThat(HttpUtils.getLocation(URI.create("http://lol.example#a"), "http://new.example/hi#b"))
        .isEqualTo(URI.create("http://new.example/hi#b"));
  }

  @Test
  public void getLocation_hasNoFragmentButOriginalDoes_mergesOldFragment() throws Exception {
    assertThat(HttpUtils.getLocation(URI.create("http://lol.example#a"), "http://new.example/hi"))
        .isEqualTo(URI.create("http://new.example/hi#a"));
  }

  @Test
  public void getLocation_oldUrlHasPassRedirectingToSameDomain_mergesPassword() throws Exception {
    assertThat(HttpUtils.getLocation(URI.create("http://a:b@lol.example"), "http://lol.example/hi"))
        .isEqualTo(URI.create("http://a:b@lol.example/hi"));
    assertThat(HttpUtils.getLocation(URI.create("http://a:b@lol.example"), "/hi"))
        .isEqualTo(URI.create("http://a:b@lol.example/hi"));
  }

  @Test
  public void getLocation_oldUrlHasPasswordRedirectingToNewServer_doesntMerge() throws Exception {
    assertThat(HttpUtils.getLocation(URI.create("http://a:b@lol.example"), "http://new.example/hi"))
        .isEqualTo(URI.create("http://new.example/hi"));
    assertThat(
            HttpUtils.getLocation(
                URI.create("http://a:b@lol.example"), "http://lol.example:81/hi"))
        .isEqualTo(URI.create("http://lol.example:81/hi"));
  }

  @Test
  public void getLocation_redirectToFtp_throwsIOException() throws Exception {
    thrown.expect(IOException.class);
    thrown.expectMessage("Bad Location");
    HttpUtils.getLocation(URI.create("http://lol.example"), "ftp://lol.example");
  }

  @Test
  public void getLocation_redirectToHttps_works() throws Exception {
    assertThat(HttpUtils.getLocation(URI.create("http://lol.example"), "https://lol.example"))
        .isEqualTo(URI.create("https://lol.example"));
  }

  @Test
  public void getLocation_preservesQuotingIfNotInheriting() throws Exception {
    String redirect =
        "http://redirected.example.org/foo?"
            + "response-content-disposition=attachment%3Bfilename%3D%22bar.tar.gz%22";
    assertThat(HttpUtils.getLocation(URI.create("http://original.example.org"), redirect))
        .isEqualTo(URI.create(redirect));
  }

  @Test
  public void getLocation_preservesQuotingWithUserIfNotInheriting() throws Exception {
    String redirect =
        "http://redirected.example.org/foo?"
            + "response-content-disposition=attachment%3Bfilename%3D%22bar.tar.gz%22";
    assertThat(HttpUtils.getLocation(URI.create("http://a:b@original.example.org"), redirect))
        .isEqualTo(URI.create(redirect));
  }
}
