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

package com.google.devtools.build.lib.bazel.bzlmod.modcommand;

import static java.nio.charset.StandardCharsets.US_ASCII;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.bazel.bzlmod.Version;
import com.google.devtools.build.lib.util.io.AnsiTerminal;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Comparator;
import java.util.List;
import javax.annotation.Nullable;

/** Renders the output table for the {@code mod upgrade} subcommand. */
public final class VersionsRenderer {

  /** A module version entry with its installed version, latest available, and dependency type. */
  public record ModuleVersionEntry(
      String name, Version installed, @Nullable Version latest, boolean isDirect) {}

  private final PrintWriter out;
  private final String reset;
  private final String green;
  private final String yellow;
  private final String bold;
  private final String dim;
  private final boolean useUtf8;

  public VersionsRenderer(OutputStream output, boolean useColor, boolean useUtf8) {
    Charset charset = useUtf8 ? StandardCharsets.UTF_8 : US_ASCII;
    this.out = new PrintWriter(new OutputStreamWriter(output, charset), /* autoFlush= */ true);
    this.reset = useColor ? new String(AnsiTerminal.Color.DEFAULT.getEscapeSeq(), US_ASCII) : "";
    this.green = useColor ? new String(AnsiTerminal.Color.GREEN.getEscapeSeq(), US_ASCII) : "";
    this.yellow = useColor ? new String(AnsiTerminal.Color.YELLOW.getEscapeSeq(), US_ASCII) : "";
    this.bold = useColor ? new String(AnsiTerminal.Color.BOLD.getEscapeSeq(), US_ASCII) : "";
    this.dim = useColor ? new String(AnsiTerminal.Color.DIM.getEscapeSeq(), US_ASCII) : "";
    this.useUtf8 = useUtf8;
  }

  /**
   * Finds the latest stable version from a list of available versions. Prefers the highest
   * non-prerelease version; falls back to the highest overall version if all are prereleases.
   */
  @Nullable
  public static Version findLatestStable(List<Version> versions) {
    if (versions.isEmpty()) {
      return null;
    }
    return versions.stream()
        .filter(v -> !v.isPrerelease())
        .max(Comparator.naturalOrder())
        .orElseGet(() -> versions.stream().max(Comparator.naturalOrder()).orElseThrow());
  }

  /**
   * Renders the versions table to the given output stream.
   *
   * @param directDeps direct dependency entries
   * @param transitiveDeps transitive dependency entries
   */
  public void render(List<ModuleVersionEntry> directDeps, List<ModuleVersionEntry> transitiveDeps) {
    // Sort each group alphabetically by module name.
    ImmutableList<ModuleVersionEntry> sortedDirectDeps =
        ImmutableList.sortedCopyOf(Comparator.comparing(ModuleVersionEntry::name), directDeps);
    ImmutableList<ModuleVersionEntry> sortedTransitiveDeps =
        ImmutableList.sortedCopyOf(Comparator.comparing(ModuleVersionEntry::name), transitiveDeps);

    int totalCount = sortedDirectDeps.size() + sortedTransitiveDeps.size();
    if (totalCount == 0) {
      out.println("No external module dependencies found.");
      out.flush();
      return;
    }

    // "Module (Indirect)" is the widest possible header (17 chars).
    int nameWidth = "Module (Indirect)".length();
    int currentWidth = "Current".length();
    int latestWidth = "Latest".length();
    int statusWidth = "upgrade available".length();
    for (ModuleVersionEntry info : Iterables.concat(sortedDirectDeps, sortedTransitiveDeps)) {
      nameWidth = Math.max(nameWidth, info.name().length());
      currentWidth = Math.max(currentWidth, info.installed().toString().length());
      if (info.latest() != null) {
        latestWidth = Math.max(latestWidth, info.latest().toString().length());
      }
    }

    // Add one whitespace left and right as padding.
    nameWidth += 2;
    currentWidth += 2;
    latestWidth += 2;
    statusWidth += 2;

    int totalWidth = nameWidth + currentWidth + latestWidth + statusWidth + 3;
    String fmt =
        String.format(
            "%%-%ds %%-%ds %%-%ds %%-%ds%%n", nameWidth, currentWidth, latestWidth, statusWidth);

    String sep = useUtf8 ? "\u2500" : "-";
    String separator = sep.repeat(totalWidth);
    int upgradeable = 0;
    int unknown = 0;

    // Print direct dependencies section.
    if (!sortedDirectDeps.isEmpty()) {
      out.printf(bold + fmt + reset, "Module", "Current", "Latest", "Status");
      out.println(separator);
      for (ModuleVersionEntry info : sortedDirectDeps) {
        switch (printRow(fmt, info)) {
          case UPGRADE_AVAILABLE -> upgradeable++;
          case UNKNOWN -> unknown++;
          default -> {}
        }
      }
    }

    // Print transitive dependencies section.
    if (!sortedTransitiveDeps.isEmpty()) {
      if (!sortedDirectDeps.isEmpty()) {
        out.println();
      }
      out.printf(bold + fmt + reset, "Module (Indirect)", "Current", "Latest", "Status");
      out.println(separator);
      for (ModuleVersionEntry info : sortedTransitiveDeps) {
        switch (printRow(fmt, info)) {
          case UPGRADE_AVAILABLE -> upgradeable++;
          case UNKNOWN -> unknown++;
          default -> {}
        }
      }
    }

    // Print summary.
    out.println();
    out.printf("%s%d%s modules total, ", bold, totalCount, reset);
    if (upgradeable > 0) {
      out.printf("%s%s%d%s with upgrades available", yellow, bold, upgradeable, reset);
      if (unknown > 0) {
        out.printf(", %s%d with unknown versions%s", dim, unknown, reset);
      }
      out.println();
    } else if (unknown > 0) {
      out.printf(
          "%sall up to date%s (%s%d with unknown versions%s)%n", green, reset, dim, unknown, reset);
    } else {
      out.printf("%sall up to date%s%n", green, reset);
    }

    out.flush();
  }

  private enum VersionStatus {
    UP_TO_DATE,
    UPGRADE_AVAILABLE,
    UNKNOWN
  }

  private VersionStatus printRow(String fmt, ModuleVersionEntry info) {
    String currentStr = info.installed().toString();
    String latestStr;
    String status;
    String lineColor;
    VersionStatus result;

    if (info.latest() == null) {
      latestStr = "?";
      status = "unknown";
      lineColor = dim;
      result = VersionStatus.UNKNOWN;
    } else if (info.latest().compareTo(info.installed()) > 0) {
      latestStr = info.latest().toString();
      status = "upgrade available";
      lineColor = yellow;
      result = VersionStatus.UPGRADE_AVAILABLE;
    } else {
      latestStr = info.latest().toString();
      status = "up to date";
      lineColor = dim;
      result = VersionStatus.UP_TO_DATE;
    }

    out.printf(lineColor + fmt + reset, info.name(), currentStr, latestStr, status);
    return result;
  }

  /** Renders a hint message suggesting upgrade commands. */
  public void renderHint() {
    out.println();
    out.printf(
        "%sHint: Run 'bazel mod upgrade <module>' to upgrade specific modules,%n"
            + "      or 'bazel mod upgrade --all' to upgrade all direct dependencies.%s%n",
        dim, reset);
    out.flush();
  }
}
