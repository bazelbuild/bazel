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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.nio.charset.StandardCharsets.US_ASCII;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.bazel.bzlmod.Version;
import com.google.devtools.build.lib.util.StringEncoding;
import com.google.devtools.build.lib.util.io.AnsiTerminal;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.Comparator;
import java.util.List;
import javax.annotation.Nullable;

/** Renders the output table for the {@code mod upgrade} subcommand. */
public final class VersionsRenderer {

  /**
   * A module version entry with its installed version, latest available, and dependency type.
   *
   * <p>{@code isPinned} marks a module whose version is fixed by a {@code single_version_override};
   * such modules are shown (so a stale pin stays visible) but never upgraded automatically.
   */
  public record ModuleVersionEntry(
      String name, Version installed, @Nullable Version latest, boolean isDirect, boolean isPinned) {
    public ModuleVersionEntry(
        String name, Version installed, @Nullable Version latest, boolean isDirect) {
      this(name, installed, latest, isDirect, /* isPinned= */ false);
    }
  }

  private final PrintWriter out;
  private final String reset;
  private final String green;
  private final String yellow;
  private final String cyan;
  private final String bold;
  private final String dim;
  private final boolean useUtf8;

  public VersionsRenderer(OutputStream output, boolean useColor, boolean useUtf8) {
    // Internal strings are raw bytes stored in Latin-1 (see StringEncoding), so always write
    // through a Latin-1 stream to emit them faithfully. The charset option only selects the
    // separator glyph (see render()), not the stream encoding.
    this.out = new PrintWriter(new OutputStreamWriter(output, ISO_8859_1), /* autoFlush= */ true);
    this.reset = useColor ? new String(AnsiTerminal.Color.DEFAULT.getEscapeSeq(), US_ASCII) : "";
    this.green = useColor ? new String(AnsiTerminal.Color.GREEN.getEscapeSeq(), US_ASCII) : "";
    this.yellow = useColor ? new String(AnsiTerminal.Color.YELLOW.getEscapeSeq(), US_ASCII) : "";
    this.cyan = useColor ? new String(AnsiTerminal.Color.CYAN.getEscapeSeq(), US_ASCII) : "";
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

    // Filter out up-to-date modules — only show entries that need attention.
    ImmutableList<ModuleVersionEntry> displayDirectDeps =
        sortedDirectDeps.stream().filter(e -> !isUpToDate(e)).collect(toImmutableList());
    ImmutableList<ModuleVersionEntry> displayTransitiveDeps =
        sortedTransitiveDeps.stream().filter(e -> !isUpToDate(e)).collect(toImmutableList());
    int upToDateCount =
        totalCount - displayDirectDeps.size() - displayTransitiveDeps.size();

    // "Module (Indirect)" is the widest possible header (17 chars).
    int nameWidth = "Module (Indirect)".length();
    int currentWidth = "Current".length();
    int latestWidth = "Latest".length();
    int statusWidth = "upgrade available".length();
    for (ModuleVersionEntry info :
        Iterables.concat(displayDirectDeps, displayTransitiveDeps)) {
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

    String sep = useUtf8 ? StringEncoding.unicodeToInternal("\u2500") : "-";
    String separator = sep.repeat(totalWidth);
    int upgradeable = 0;
    int unknown = 0;
    int pinned = 0;

    // Print direct dependencies section.
    if (!displayDirectDeps.isEmpty()) {
      out.printf(bold + fmt + reset, "Module", "Current", "Latest", "Status");
      out.println(separator);
      for (ModuleVersionEntry info : displayDirectDeps) {
        switch (printRow(fmt, info)) {
          case UPGRADE_AVAILABLE -> upgradeable++;
          case UNKNOWN -> unknown++;
          case PINNED -> pinned++;
          default -> {}
        }
      }
    }

    // Print transitive dependencies section.
    if (!displayTransitiveDeps.isEmpty()) {
      if (!displayDirectDeps.isEmpty()) {
        out.println();
      }
      out.printf(bold + fmt + reset, "Module (Indirect)", "Current", "Latest", "Status");
      out.println(separator);
      for (ModuleVersionEntry info : displayTransitiveDeps) {
        switch (printRow(fmt, info)) {
          case UPGRADE_AVAILABLE -> upgradeable++;
          case UNKNOWN -> unknown++;
          case PINNED -> pinned++;
          default -> {}
        }
      }
    }

    // Print summary.
    out.println();
    out.printf("%s%d%s modules total", bold, totalCount, reset);
    if (upgradeable > 0 || pinned > 0) {
      if (upgradeable > 0) {
        out.printf(", %s%s%d%s with upgrades available", yellow, bold, upgradeable, reset);
      }
      if (pinned > 0) {
        out.printf(", %s%d pinned%s", cyan, pinned, reset);
      }
      if (unknown > 0) {
        out.printf(", %s%d with unknown versions%s", dim, unknown, reset);
      }
      if (upToDateCount > 0) {
        out.printf(", %s%d up to date%s", dim, upToDateCount, reset);
      }
      out.println();
    } else if (unknown > 0) {
      out.printf(
          ", %sall up to date%s (%s%d with unknown versions%s)%n",
          green, reset, dim, unknown, reset);
    } else {
      out.printf(", %sall up to date%s%n", green, reset);
    }

    out.flush();
  }

  private static boolean isUpToDate(ModuleVersionEntry entry) {
    return entry.latest() != null && entry.latest().compareTo(entry.installed()) <= 0;
  }

  private enum VersionStatus {
    UP_TO_DATE,
    UPGRADE_AVAILABLE,
    UNKNOWN,
    PINNED
  }

  private VersionStatus printRow(String fmt, ModuleVersionEntry info) {
    String currentStr = info.installed().toString();
    String latestStr;
    String status;
    String lineColor;
    VersionStatus result;

    if (info.isPinned()) {
      // Version is fixed by an override; surface that a newer version exists, but don't present it
      // as an actionable upgrade. Use a distinct color from the yellow "upgrade available" rows.
      latestStr = info.latest() != null ? info.latest().toString() : "?";
      status = "pinned";
      lineColor = cyan;
      result = VersionStatus.PINNED;
    } else if (info.latest() == null) {
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
