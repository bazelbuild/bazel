// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.authandtls;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.net.URI;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link StaticCredentials}. */
@RunWith(JUnit4.class)
public class StaticCredentialsTest {

  @Test
  public void exactUriMatch() throws Exception {
    URI uri = URI.create("https://example.com/path/to/file");
    Map<String, ImmutableList<String>> headers =
        ImmutableMap.of("Authorization", ImmutableList.of("Bearer token123"));
    StaticCredentials credentials =
        new StaticCredentials(ImmutableMap.of(uri, headers));

    assertThat(credentials.getRequestMetadata(uri)).isEqualTo(headers);
  }

  @Test
  public void sameHostDifferentPath_fallback() throws Exception {
    URI originalUri = URI.create("https://example.com/old/path");
    URI redirectUri = URI.create("https://example.com/new/path");
    Map<String, ImmutableList<String>> headers =
        ImmutableMap.of("Authorization", ImmutableList.of("Bearer token123"));
    StaticCredentials credentials =
        new StaticCredentials(ImmutableMap.of(originalUri, headers));

    // Exact match fails, but host-based fallback should succeed.
    assertThat(credentials.getRequestMetadata(redirectUri)).isEqualTo(headers);
  }

  @Test
  public void sameHostDifferentPort_noFallback() throws Exception {
    URI originalUri = URI.create("https://example.com:8080/path");
    URI differentPortUri = URI.create("https://example.com:9090/path");
    Map<String, ImmutableList<String>> headers =
        ImmutableMap.of("Authorization", ImmutableList.of("Bearer token123"));
    StaticCredentials credentials =
        new StaticCredentials(ImmutableMap.of(originalUri, headers));

    // Different port should not match.
    assertThat(credentials.getRequestMetadata(differentPortUri)).isEmpty();
  }

  @Test
  public void differentHost_noFallback() throws Exception {
    URI originalUri = URI.create("https://example.com/path");
    URI differentHostUri = URI.create("https://other.com/path");
    Map<String, ImmutableList<String>> headers =
        ImmutableMap.of("Authorization", ImmutableList.of("Bearer token123"));
    StaticCredentials credentials =
        new StaticCredentials(ImmutableMap.of(originalUri, headers));

    // Different host should not match.
    assertThat(credentials.getRequestMetadata(differentHostUri)).isEmpty();
  }

  @Test
  public void sameHostNoPort_matchesHostWithNoPort() throws Exception {
    URI originalUri = URI.create("https://example.com/path");
    URI redirectUri = URI.create("https://example.com/other");
    Map<String, ImmutableList<String>> headers =
        ImmutableMap.of("Authorization", ImmutableList.of("Bearer token123"));
    StaticCredentials credentials =
        new StaticCredentials(ImmutableMap.of(originalUri, headers));

    assertThat(credentials.getRequestMetadata(redirectUri)).isEqualTo(headers);
  }

  @Test
  public void emptyCredentials_returnsEmpty() throws Exception {
    StaticCredentials credentials = StaticCredentials.EMPTY;
    URI uri = URI.create("https://example.com/path");

    assertThat(credentials.getRequestMetadata(uri)).isEmpty();
  }
}
