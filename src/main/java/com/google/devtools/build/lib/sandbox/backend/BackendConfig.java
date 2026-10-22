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

package com.google.devtools.build.lib.sandbox.backend;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Launch configuration of one sandbox backend: the binary and the options passed to it on its
 * command line. Two configurations that are equal launch identical backends, so equality is what
 * decides whether a running backend has to be restarted.
 *
 * @param binary absolute path of the backend binary, or a bare name resolved against {@code PATH}
 * @param options the {@code --sandbox_backend_opt} values for this backend, in flag order
 */
public record BackendConfig(String binary, ImmutableList<String> options) {

  /**
   * Resolves the {@code --sandbox_backend} and {@code --sandbox_backend_opt} flags into one {@link
   * BackendConfig} per backend name, in registration order.
   *
   * <p>When a name is registered more than once (for example in an rc file and on the command
   * line), the last registration wins, as for any other flag. Options accumulate per name in flag
   * order.
   *
   * @throws IOException if a {@code --sandbox_backend_opt} names a backend that no {@code
   *     --sandbox_backend} registers, which would otherwise silently drop the option
   */
  public static ImmutableMap<String, BackendConfig> fromFlags(
      List<Map.Entry<String, String>> backends, List<Map.Entry<String, String>> opts)
      throws IOException {
    Map<String, String> binaries = new LinkedHashMap<>();
    for (Map.Entry<String, String> backend : backends) {
      if (backend.getKey().isEmpty() || backend.getValue().isEmpty()) {
        throw new IOException(
            String.format(
                "--sandbox_backend=%s=%s must have the form <name>=<binary>",
                backend.getKey(), backend.getValue()));
      }
      binaries.put(backend.getKey(), backend.getValue());
    }
    Map<String, List<String>> options = new LinkedHashMap<>();
    for (Map.Entry<String, String> opt : opts) {
      if (!binaries.containsKey(opt.getKey())) {
        throw new IOException(
            String.format(
                "--sandbox_backend_opt=%s=%s refers to the unknown sandbox backend '%s'; %s",
                opt.getKey(),
                opt.getValue(),
                opt.getKey(),
                binaries.isEmpty()
                    ? "no backends are registered with --sandbox_backend"
                    : "registered backends: " + String.join(", ", binaries.keySet())));
      }
      options.computeIfAbsent(opt.getKey(), unused -> new ArrayList<>()).add(opt.getValue());
    }
    ImmutableMap.Builder<String, BackendConfig> result = ImmutableMap.builder();
    for (Map.Entry<String, String> binary : binaries.entrySet()) {
      result.put(
          binary.getKey(),
          new BackendConfig(
              binary.getValue(),
              ImmutableList.copyOf(options.getOrDefault(binary.getKey(), ImmutableList.of()))));
    }
    return result.buildOrThrow();
  }

  /**
   * Returns the executable to launch, or null if the binary doesn't exist or isn't executable. A
   * bare name is resolved against the {@code PATH} in {@code clientEnv}, the first match wins.
   */
  public String resolveBinary(ImmutableMap<String, String> clientEnv) {
    if (binary.contains("/")) {
      return new File(binary).canExecute() ? binary : null;
    }
    String path = clientEnv.get("PATH");
    if (path == null) {
      return null;
    }
    for (String dir : path.split(File.pathSeparator)) {
      if (dir.isEmpty()) {
        continue;
      }
      File candidate = new File(dir, binary);
      if (candidate.canExecute()) {
        return candidate.getPath();
      }
    }
    return null;
  }
}
