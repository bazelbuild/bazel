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

  private static final Map<String, List<String>> AUTH =
      ImmutableMap.of("Authorization", ImmutableList.of("Basic dTpw"));
  private static final Map<String, List<String>> OTHER_AUTH =
      ImmutableMap.of("Authorization", ImmutableList.of("Basic eDp5"));

  @Test
  public void getRequestMetadata_exactMatch_returnsCredentials() throws Exception {
    StaticCredentials credentials =
        new StaticCredentials(ImmutableMap.of(new URI("http://example.com/start"), AUTH));

    assertThat(credentials.getRequestMetadata(new URI("http://example.com/start"))).isEqualTo(AUTH);
  }

  @Test
  public void getRequestMetadata_sameHostDifferentPath_returnsCredentials() throws Exception {
    StaticCredentials credentials =
        new StaticCredentials(ImmutableMap.of(new URI("http://example.com/start"), AUTH));

    assertThat(credentials.getRequestMetadata(new URI("http://example.com/protected")))
        .isEqualTo(AUTH);
  }

  @Test
  public void getRequestMetadata_sameHostMatchingIsCaseInsensitive() throws Exception {
    StaticCredentials credentials =
        new StaticCredentials(ImmutableMap.of(new URI("http://Example.COM/start"), AUTH));

    assertThat(credentials.getRequestMetadata(new URI("http://example.com/protected")))
        .isEqualTo(AUTH);
  }

  @Test
  public void getRequestMetadata_differentHost_returnsEmpty() throws Exception {
    StaticCredentials credentials =
        new StaticCredentials(ImmutableMap.of(new URI("http://example.com/start"), AUTH));

    assertThat(credentials.getRequestMetadata(new URI("http://other.example.com/protected")))
        .isEmpty();
  }

  @Test
  public void getRequestMetadata_differentScheme_returnsEmpty() throws Exception {
    StaticCredentials credentials =
        new StaticCredentials(ImmutableMap.of(new URI("http://example.com/start"), AUTH));

    assertThat(credentials.getRequestMetadata(new URI("https://example.com/start"))).isEmpty();
  }

  @Test
  public void getRequestMetadata_differentPort_returnsEmpty() throws Exception {
    StaticCredentials credentials =
        new StaticCredentials(ImmutableMap.of(new URI("http://example.com:8080/start"), AUTH));

    assertThat(credentials.getRequestMetadata(new URI("http://example.com:9090/protected")))
        .isEmpty();
  }

  @Test
  public void getRequestMetadata_exactMatchTakesPrecedenceOverSameHostFallback() throws Exception {
    // Exact-path matches keep their own credentials; only an unknown path on the same host falls
    // back (to the first registered entry for that host).
    StaticCredentials credentials =
        new StaticCredentials(
            ImmutableMap.of(
                new URI("http://example.com/a"), AUTH,
                new URI("http://example.com/b"), OTHER_AUTH));

    assertThat(credentials.getRequestMetadata(new URI("http://example.com/a"))).isEqualTo(AUTH);
    assertThat(credentials.getRequestMetadata(new URI("http://example.com/b")))
        .isEqualTo(OTHER_AUTH);
    assertThat(credentials.getRequestMetadata(new URI("http://example.com/c"))).isEqualTo(AUTH);
  }

  @Test
  public void getRequestMetadata_sameHostMultiplePaths_returnsCredentials() throws Exception {
    StaticCredentials credentials =
        new StaticCredentials(
            ImmutableMap.of(
                new URI("http://example.com/a"), AUTH,
                new URI("http://example.com/b"), AUTH));

    assertThat(credentials.getRequestMetadata(new URI("http://example.com/c"))).isEqualTo(AUTH);
  }
}
