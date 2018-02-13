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
      WindowsLocalEnvProvider p,
      ImmutableMap<String, String> env,
      String localTmpRoot,
      String fallbackDir) {
    return p.rewriteLocalEnv(env, null, localTmpRoot, fallbackDir, null);
  }

  @Test
  public void testRewriteEnv() throws Exception {
    // localTmpRoot is specified, so ignore everything else.
    assertThat(
            rewriteEnv(
                new WindowsLocalEnvProvider(
                    ImmutableMap.of("TMP", "client/env/tmp", "TEMP", "client/env/temp")),
                ImmutableMap.of("key1", "value1", "TMP", "spawn/tmp", "TEMP", "spawn/temp"),
                "local/tmp",
                "fallback/dir"))
        .isEqualTo(ImmutableMap.of("key1", "value1", "TMP", "local\\tmp", "TEMP", "local\\tmp"));

    // localTmpRoot is empty, fall back to the client environment's TMP.
    assertThat(
            rewriteEnv(
                new WindowsLocalEnvProvider(
                    ImmutableMap.of("TMP", "client/tmp", "TEMP", "client/temp")),
                ImmutableMap.of("key1", "value1", "TMP", "spawn/tmp", "TEMP", "spawn/temp"),
                "",
                "fallback/dir"))
        .isEqualTo(ImmutableMap.of("key1", "value1", "TMP", "client\\tmp", "TEMP", "client\\tmp"));

    // localTmpRoot and the client environment's TMP are empty, fall back to TEMP.
    assertThat(
            rewriteEnv(
                new WindowsLocalEnvProvider(ImmutableMap.of("TMP", "", "TEMP", "client/temp")),
                ImmutableMap.of("key1", "value1", "TMP", "spawn/tmp", "TEMP", "spawn/temp"),
                "",
                "fallback/dir"))
        .isEqualTo(
            ImmutableMap.of("key1", "value1", "TMP", "client\\temp", "TEMP", "client\\temp"));

    // localTmpRoot and the client environment's TMP and TEMP are empty, fall back to fallbackDir.
    assertThat(
            rewriteEnv(
                new WindowsLocalEnvProvider(ImmutableMap.of("TMP", "", "TEMP", "")),
                ImmutableMap.of("key1", "value1", "TMP", "spawn/tmp", "TEMP", "spawn/temp"),
                "",
                "fallback/dir"))
        .isEqualTo(
            ImmutableMap.of("key1", "value1", "TMP", "fallback\\dir", "TEMP", "fallback\\dir"));
  }
}
