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

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.exec.BinTools;
import java.util.Map;

/** {@link LocalEnvProvider} implementation for actions running on Windows. */
public final class WindowsLocalEnvProvider implements LocalEnvProvider {
  private final Map<String, String> clientEnv;

  /**
   * Create a new {@link WindowsLocalEnvProvider}.
   *
   * <p>Use {@link LocalEnvProvider#forCurrentOs(Map)} to instantiate this.
   *
   * @param clientEnv a map of the current Bazel command's environment
   */
  public WindowsLocalEnvProvider(Map<String, String> clientEnv) {
    this.clientEnv = clientEnv;
  }

  /**
   * Compute an environment map for local actions on Windows.
   *
   * <p>Returns a map with the same keys and values as {@code env}. Overrides the value of TMP and
   * TEMP (or adds them if not present in {@code env}) by the same value, which is:
   *
   * <ul>
   *   <li>the value of {@code clientEnv.get("TMP")}, or if that's empty or null, then
   *   <li>the value of {@code clientEnv.get("TEMP")}, or if that's empty or null, then
   *   <li>the value of {@code fallbackTmpDir}.
   * </ul>
   *
   * <p>The values for TMP and TEMP will use backslashes as directory separators.
   */
  @Override
  public ImmutableMap<String, String> rewriteLocalEnv(
      Map<String, String> env, BinTools binTools, String fallbackTmpDir) {
    ImmutableMap.Builder<String, String> result = ImmutableMap.builder();
    result.putAll(Maps.filterKeys(env, k -> !k.equals("TMP") && !k.equals("TEMP")));
    String p = clientEnv.get("TMP");
    if (Strings.isNullOrEmpty(p)) {
      p = clientEnv.get("TEMP");
      if (Strings.isNullOrEmpty(p)) {
        p = fallbackTmpDir;
      }
    }
    p = p.replace('/', '\\');
    result.put("TMP", p);
    result.put("TEMP", p);
    return result.build();
  }
}
