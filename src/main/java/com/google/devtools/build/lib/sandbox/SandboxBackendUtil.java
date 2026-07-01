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

import com.google.common.collect.ImmutableMap;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.File;
import java.util.logging.Level;

/**
 * Static helpers for the {@code sandbox-backend} strategy. Manifest construction lives in
 * {@link SandboxBackendManifest}; this class hosts the controller availability probe.
 */
public final class SandboxBackendUtil {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private SandboxBackendUtil() {}

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
