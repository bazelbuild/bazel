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

package com.google.devtools.build.lib.sandbox;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;

/**
 * Static helpers for the {@code sandbox-backend} strategy. Manifest construction lives in
 * {@link SandboxBackendManifest}; this class hosts the controller availability probe and the
 * resolution of the {@code --sandbox_backend} / {@code --sandbox_backend_opt} flags into per-backend
 * launch configurations.
 */
public final class SandboxBackendUtil {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private SandboxBackendUtil() {}

  /**
   * Launch configuration of one registered backend: the controller binary and the options relayed
   * to it via the Negotiate handshake. Two configs with equal fields launch identical servers, so
   * the record's equality is what decides whether a running server must be restarted.
   *
   * @param binary controller binary; absolute path or bare name resolved against {@code PATH}
   * @param options {@code --sandbox_backend_opt} values for this backend, in flag order
   */
  record BackendConfig(PathFragment binary, ImmutableList<String> options) {}

  /**
   * Resolves the {@code --sandbox_backend} and {@code --sandbox_backend_opt} flags into one {@link
   * BackendConfig} per backend name, in registration order.
   *
   * <p>When the same name is registered more than once (e.g. once in an rc file and once on the
   * command line), the last binary wins, matching how a later flag overrides an earlier one. Options
   * accumulate per name in flag order.
   *
   * @throws IOException if a {@code --sandbox_backend_opt} names a backend that no {@code
   *     --sandbox_backend} registers; the option would otherwise be silently dropped
   */
  static ImmutableMap<String, BackendConfig> configuredBackends(
      List<Map.Entry<String, String>> backends, List<Map.Entry<String, String>> opts)
      throws IOException {
    Map<String, PathFragment> binaries = new LinkedHashMap<>();
    for (Map.Entry<String, String> backend : backends) {
      binaries.put(backend.getKey(), PathFragment.create(backend.getValue()));
    }
    Map<String, List<String>> options = new LinkedHashMap<>();
    for (Map.Entry<String, String> opt : opts) {
      if (!binaries.containsKey(opt.getKey())) {
        throw new IOException(
            String.format(
                "--sandbox_backend_opt=%s=%s refers to unknown sandbox backend '%s'; %s",
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
    for (Map.Entry<String, PathFragment> binary : binaries.entrySet()) {
      result.put(
          binary.getKey(),
          new BackendConfig(
              binary.getValue(),
              ImmutableList.copyOf(options.getOrDefault(binary.getKey(), ImmutableList.of()))));
    }
    return result.buildOrThrow();
  }

  /**
   * Checks whether the controller binary exists and is executable.
   *
   * <p>Pure filesystem check, no subprocess. A path containing {@code /} must exist and be
   * executable; a bare name is resolved against {@code PATH} (from {@code clientEnv}), first
   * executable match wins. Otherwise returns {@code false} silently so the strategy declines and
   * Bazel falls through to the next {@code --spawn_strategy}.
   *
   * @param binary controller binary; absolute path or bare name
   * @param clientEnv environment supplying {@code PATH} for bare-name resolution
   * @return {@code true} if the binary exists and is executable
   */
  public static boolean isAvailable(PathFragment binary, ImmutableMap<String, String> clientEnv) {
    if (binary.isEmpty()) {
      return false;
    }
    String binaryStr = binary.getPathString();
    if (binaryStr.contains("/")) {
      File f = new File(binaryStr);
      if (f.canExecute()) {
        return true;
      }
      logger.at(Level.FINE).log(
          "sandbox backend at %s does not exist or is not executable", binaryStr);
      return false;
    }
    String pathEnv = clientEnv.get("PATH");
    if (pathEnv == null || pathEnv.isEmpty()) {
      logger.at(Level.FINE).log(
          "sandbox backend %s requested by bare name but PATH is unset", binaryStr);
      return false;
    }
    for (String dir : pathEnv.split(File.pathSeparator)) {
      if (dir.isEmpty()) {
        continue;
      }
      File candidate = new File(dir, binaryStr);
      if (candidate.canExecute()) {
        return true;
      }
    }
    logger.at(Level.FINE).log(
        "sandbox backend %s not found on PATH (%s)", binaryStr, pathEnv);
    return false;
  }
}
