// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.exec.local;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link WindowsLocalEnvProvider}. */
@RunWith(JUnit4.class)
public final class WindowsLocalEnvProviderTest {

  private static Map<String, String> rewriteEnv(
      WindowsLocalEnvProvider p, ImmutableMap<String, String> env) {
    return p.rewriteLocalEnv(env, null, null);
  }

  private static Map<String, String> rewriteEnv(
      WindowsLocalEnvProvider p, ImmutableMap<String, String> env, String fallback) {
    return p.rewriteLocalEnv(env, null, fallback);
  }

  /** Should use the client environment's TMP envvar if specified. */
  @Test
  public void testRewriteEnvWithClientTmp() throws Exception {
    WindowsLocalEnvProvider p =
        new WindowsLocalEnvProvider(
            ImmutableMap.of("TMP", "client-env/tmp", "TEMP", "ignore/when/tmp/is/present"));

    assertThat(rewriteEnv(p, ImmutableMap.of("key1", "value1", "TMP", "ignore", "TEMP", "ignore")))
        .isEqualTo(
            ImmutableMap.of("key1", "value1", "TMP", "client-env\\tmp", "TEMP", "client-env\\tmp"));

    assertThat(rewriteEnv(p, ImmutableMap.of("key1", "value1", "TMP", "ignore")))
        .isEqualTo(
            ImmutableMap.of("key1", "value1", "TMP", "client-env\\tmp", "TEMP", "client-env\\tmp"));

    assertThat(rewriteEnv(p, ImmutableMap.of("key1", "value1")))
        .isEqualTo(
            ImmutableMap.of("key1", "value1", "TMP", "client-env\\tmp", "TEMP", "client-env\\tmp"));
  }

  /** Should use the client environment's TEMP envvar if TMP is unspecified. */
  @Test
  public void testRewriteEnvWithoutClientTmpWithClientTemp() throws Exception {
    WindowsLocalEnvProvider p =
        new WindowsLocalEnvProvider(ImmutableMap.of("TEMP", "client-env/temp"));

    assertThat(rewriteEnv(p, ImmutableMap.of("key1", "value1", "TMP", "ignore", "TEMP", "ignore")))
        .isEqualTo(
            ImmutableMap.of(
                "key1", "value1", "TMP", "client-env\\temp", "TEMP", "client-env\\temp"));

    assertThat(rewriteEnv(p, ImmutableMap.of("key1", "value1", "TMP", "ignore")))
        .isEqualTo(
            ImmutableMap.of(
                "key1", "value1", "TMP", "client-env\\temp", "TEMP", "client-env\\temp"));

    assertThat(rewriteEnv(p, ImmutableMap.of("key1", "value1")))
        .isEqualTo(
            ImmutableMap.of(
                "key1", "value1", "TMP", "client-env\\temp", "TEMP", "client-env\\temp"));
  }

  /** Should use the fallback temp dir when the client env defines neither TMP nor TEMP. */
  @Test
  public void testRewriteEnvWithFallbackTmp() throws Exception {
    WindowsLocalEnvProvider p = new WindowsLocalEnvProvider(ImmutableMap.<String, String>of());

    assertThat(
            rewriteEnv(
                p,
                ImmutableMap.of("key1", "value1", "TMP", "ignore", "TEMP", "ignore"),
                "fallback/tmp"))
        .isEqualTo(
            ImmutableMap.of("key1", "value1", "TMP", "fallback\\tmp", "TEMP", "fallback\\tmp"));

    assertThat(rewriteEnv(p, ImmutableMap.of("key1", "value1", "TMP", "ignore"), "fallback/tmp"))
        .isEqualTo(
            ImmutableMap.of("key1", "value1", "TMP", "fallback\\tmp", "TEMP", "fallback\\tmp"));

    assertThat(rewriteEnv(p, ImmutableMap.of("key1", "value1"), "fallback/tmp"))
        .isEqualTo(
            ImmutableMap.of("key1", "value1", "TMP", "fallback\\tmp", "TEMP", "fallback\\tmp"));
  }
}
