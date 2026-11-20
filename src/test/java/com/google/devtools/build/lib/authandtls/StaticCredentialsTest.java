// Copyright 2026 The Bazel Authors. All rights reserved.
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
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link StaticCredentials}. */
@RunWith(JUnit4.class)
public class StaticCredentialsTest {

  private static final Map<String, List<String>> USER_A =
      ImmutableMap.of("Authorization", ImmutableList.of("Basic YTph"));
  private static final Map<String, List<String>> USER_B =
      ImmutableMap.of("Authorization", ImmutableList.of("Basic Yjpi"));

  @Test
  public void exactUriMatch_isReturned() throws Exception {
    StaticCredentials credentials =
        new StaticCredentials(ImmutableMap.of(URI.create("http://localhost:8077/start"), USER_A));

    assertThat(credentials.getRequestMetadata(URI.create("http://localhost:8077/start")))
        .isEqualTo(USER_A);
  }

  @Test
  public void sameOriginRedirectTarget_getsCredentials() throws Exception {
    StaticCredentials credentials =
        new StaticCredentials(ImmutableMap.of(URI.create("http://localhost:8077/start"), USER_A));

    assertThat(credentials.getRequestMetadata(URI.create("http://localhost:8077/protected")))
        .isEqualTo(USER_A);
  }

  @Test
  public void sameOrigin_isCaseInsensitiveAndUsesDefaultPorts() throws Exception {
    StaticCredentials credentials =
        new StaticCredentials(ImmutableMap.of(URI.create("https://Example.COM/a"), USER_A));

    assertThat(credentials.getRequestMetadata(URI.create("https://example.com:443/b")))
        .isEqualTo(USER_A);
  }

  @Test
  public void differentHost_getsNoCredentials() throws Exception {
    StaticCredentials credentials =
        new StaticCredentials(ImmutableMap.of(URI.create("http://a.example.com/start"), USER_A));

    assertThat(credentials.getRequestMetadata(URI.create("http://b.example.com/start"))).isEmpty();
    assertThat(credentials.getRequestMetadata(URI.create("http://sub.a.example.com/start")))
        .isEmpty();
  }

  @Test
  public void differentPort_getsNoCredentials() throws Exception {
    StaticCredentials credentials =
        new StaticCredentials(ImmutableMap.of(URI.create("http://localhost:8077/start"), USER_A));

    assertThat(credentials.getRequestMetadata(URI.create("http://localhost:9999/start"))).isEmpty();
  }

  @Test
  public void schemeDowngrade_getsNoCredentials() throws Exception {
    StaticCredentials credentials =
        new StaticCredentials(ImmutableMap.of(URI.create("https://example.com/start"), USER_A));

    assertThat(credentials.getRequestMetadata(URI.create("http://example.com/start"))).isEmpty();
  }

  @Test
  public void conflictingCredentialsOnSameOrigin_areNotGuessed() throws Exception {
    StaticCredentials credentials =
        new StaticCredentials(
            ImmutableMap.of(
                URI.create("http://example.com/a"), USER_A,
                URI.create("http://example.com/b"), USER_B));

    // Exact matches still work.
    assertThat(credentials.getRequestMetadata(URI.create("http://example.com/a")))
        .isEqualTo(USER_A);
    assertThat(credentials.getRequestMetadata(URI.create("http://example.com/b")))
        .isEqualTo(USER_B);
    // But an unknown path on that origin gets nothing, as there is no way to pick one.
    assertThat(credentials.getRequestMetadata(URI.create("http://example.com/c"))).isEmpty();
  }

  @Test
  public void identicalCredentialsOnSameOrigin_areNotAmbiguous() throws Exception {
    StaticCredentials credentials =
        new StaticCredentials(
            ImmutableMap.of(
                URI.create("http://example.com/a"), USER_A,
                URI.create("http://example.com/b"), USER_A));

    assertThat(credentials.getRequestMetadata(URI.create("http://example.com/c")))
        .isEqualTo(USER_A);
  }

  @Test
  public void empty_returnsNothing() throws Exception {
    assertThat(StaticCredentials.EMPTY.getRequestMetadata(URI.create("http://example.com/a")))
        .isEmpty();
  }
}