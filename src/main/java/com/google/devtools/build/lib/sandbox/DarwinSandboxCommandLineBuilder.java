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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.vfs.Path;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Builds the macOS Seatbelt confinement for an action: the {@code .sb} sandbox profile and the
 * {@code sandbox-exec} command line that wraps the action argv. Shared by the {@code darwin-sandbox}
 * strategy and by sandbox backends that materialize the filesystem themselves but reuse Bazel's
 * confinement.
 */
public final class DarwinSandboxCommandLineBuilder {
  private DarwinSandboxCommandLineBuilder() {}

  /** Wraps {@code commandToRun} in {@code sandbox-exec} applying the profile at {@code profile}. */
  public static ImmutableList<String> wrapCommand(
      String sandboxExecBinary, Path profile, List<String> commandToRun) {
    return ImmutableList.<String>builder()
        .add(sandboxExecBinary)
        .add("-f")
        .add(profile.getPathString())
        .addAll(commandToRun)
        .build();
  }

  /**
   * Writes a Seatbelt profile to {@code profilePath}: read-allowed by default, writes denied except
   * under {@code writableDirs} (and {@code statisticsPath}), network denied unless {@code
   * allowNetwork}, and reads denied under {@code inaccessiblePaths}.
   */
  public static void writeProfile(
      Path profilePath,
      Set<Path> writableDirs,
      Set<Path> inaccessiblePaths,
      boolean allowNetwork,
      @Nullable Path statisticsPath)
      throws IOException {
    try (PrintWriter out =
        new PrintWriter(
            new BufferedWriter(
                new OutputStreamWriter(profilePath.getOutputStream(), UTF_8)))) {
      // Note: In Apple's sandbox configuration language, the *last* matching rule wins.
      out.println("(version 1)");
      out.println("(debug deny)");
      out.println("(allow default)");
      out.println("(allow process-exec (with no-sandbox) (literal \"/bin/ps\"))");

      if (!allowNetwork) {
        out.println("(deny network*)");
        // Binding to any local address must be allowed: servers commonly bind to the wildcard
        // address (0.0.0.0 or ::), which the "localhost" filter does not match. Outbound
        // traffic stays restricted to loopback by the rules below, so a sandboxed spawn still
        // cannot reach another host.
        //
        // Inbound traffic cannot be restricted the same way. macOS decides network-inbound
        // from the local address alone, and "localhost" there matches every address assigned
        // to the machine rather than just 127.0.0.1 and ::1, so a peer elsewhere on the
        // network can reach a spawn that listens on a non-loopback address. Filtering
        // network-inbound on the remote address is not an option: such a rule compiles but
        // has no effect. The Linux sandbox does not have this gap because its network
        // namespace only ever has a loopback interface.
        // See https://github.com/bazelbuild/bazel/issues/14828.
        out.println("(allow network-bind (local ip \"*:*\"))");
        out.println("(allow network-inbound (local ip \"localhost:*\"))");
        out.println("(allow network* (remote ip \"localhost:*\"))");
        // Unix domain sockets are filesystem-scoped IPC and cannot reach other hosts, so
        // allow them in both directions, just like the Linux network namespace does.
        out.println("(allow network* (local unix-socket))");
        out.println("(allow network* (remote unix-socket))");
      }

      // By default, everything is read-only.
      out.println("(deny file-write*)");

      out.println("(allow file-write*");
      for (Path path : writableDirs) {
        out.println("    (subpath \"" + escapeSchemeString(path.getPathString()) + "\")");
      }
      if (statisticsPath != null) {
        out.println("    (literal \"" + escapeSchemeString(statisticsPath.getPathString()) + "\")");
      }
      out.println(")");

      if (!inaccessiblePaths.isEmpty()) {
        out.println("(deny file-read*");
        // The sandbox configuration file is not part of a cache key and sandbox-exec doesn't care
        // about ordering of paths in expressions, so it's fine if the iteration order is random.
        for (Path inaccessiblePath : inaccessiblePaths) {
          out.println("    (subpath \"" + escapeSchemeString(inaccessiblePath.toString()) + "\")");
        }
        out.println(")");
      }
    }
  }

  /** Escapes quotes and backslashes for Apple SBPL Scheme string literals. */
  private static String escapeSchemeString(String s) {
    if (s == null) {
      return "";
    }
    return s.replace("\\", "\\\\").replace("\"", "\\\"");
  }
}
