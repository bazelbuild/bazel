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
package com.google.devtools.build.lib.analysis.starlark;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.util.ShellEscaper;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import java.util.TreeSet;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkValue;

/**
 * The result of {@code ctx.expand_location(..., lazy = True)}: a compact recipe for the expanded
 * string that can be added to {@code ctx.actions.args()} in place of the eagerly expanded string.
 *
 * <p>Unlike eager expansion, no string containing artifact paths is created during analysis: the
 * recipe only references the original input string (which is typically an attribute value that is
 * retained by the rule anyway), a small {@code int[]} describing the location expression sites,
 * and the resolved artifacts. Paths are rendered only when the consuming action's command line is
 * expanded, at which point they are subject to path mapping.
 */
@StarlarkBuiltin(
    name = "lazy_location_expansion",
    documented = false,
    doc = "An opaque, lazily rendered location expansion that can be added to Args.")
public final class LazyLocationExpansion implements StarlarkValue {

  // Path rendering modes, mirroring LocationExpander.LocationFunction.PathType.
  public static final int MODE_LOCATION = 0;
  public static final int MODE_EXEC = 1;
  public static final int MODE_RLOCATION = 2;

  // The name of the runfiles directory of the main repository ("_main" for Bazel). This is a
  // constant that only depends on the product, so the choice is recorded once when the product's
  // ConfiguredRuleClassProvider is constructed rather than stored per expansion, which keeps both
  // the instance layout and the retained encoding in Args free of it.
  private static volatile String workspaceRunfilesDirectory;

  /** Records the product's main repository runfiles directory name. */
  public static void setWorkspaceRunfilesDirectory(String directory) {
    workspaceRunfilesDirectory = directory;
  }

  private final String original;
  // Triples of (site start offset, site end offset, mode), one per location expression in
  // original. The literal segments between sites are rendered directly from original.
  private final int[] layout;
  // One element per site: an Artifact or an ImmutableList<Artifact>.
  private final Object[] values;

  public LazyLocationExpansion(String original, int[] layout, Object[] values) {
    this.original = original;
    this.layout = layout;
    this.values = values;
  }

  String original() {
    return original;
  }

  int[] layout() {
    return layout;
  }

  Object[] values() {
    return values;
  }

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public void repr(Printer printer, StarlarkSemantics semantics) {
    printer.append("lazy_location_expansion(").repr(original, semantics).append(")");
  }

  /**
   * Renders the expansion, reading the per-site values from {@code values} starting at {@code
   * valuesStart}. Exec paths are subject to {@code pathMapper}.
   */
  static String render(
      String original, int[] layout, List<Object> values, int valuesStart, PathMapper pathMapper) {
    StringBuilder result = new StringBuilder(original.length());
    int numSites = layout.length / 3;
    int previousEnd = 0;
    for (int i = 0; i < numSites; i++) {
      result.append(original, previousEnd, layout[3 * i]);
      renderSite(result, values.get(valuesStart + i), layout[3 * i + 2], pathMapper);
      previousEnd = layout[3 * i + 1];
    }
    result.append(original, previousEnd, original.length());
    return result.toString();
  }

  @SuppressWarnings("unchecked")
  private static void renderSite(
      StringBuilder result, Object value, int mode, PathMapper pathMapper) {
    // Mirrors LocationExpander.LocationFunction#getPaths and #joinPaths: paths are deduplicated
    // and sorted after rendering and joined with spaces after shell escaping.
    TreeSet<String> paths = new TreeSet<>();
    if (value instanceof Artifact artifact) {
      paths.add(renderPath(artifact, mode, pathMapper));
    } else {
      for (Artifact artifact : (ImmutableList<Artifact>) value) {
        paths.add(renderPath(artifact, mode, pathMapper));
      }
    }
    boolean first = true;
    for (String path : paths) {
      if (!first) {
        result.append(' ');
      }
      result.append(ShellEscaper.escapeString(path));
      first = false;
    }
  }

  private static String renderPath(Artifact artifact, int mode, PathMapper pathMapper) {
    // Mirrors LocationExpander.LocationFunction#getPath. Only exec paths are subject to path
    // mapping: runfiles and rlocation paths do not contain a configuration prefix.
    PathFragment path =
        switch (mode) {
          case MODE_LOCATION -> artifact.getRunfilesPath();
          case MODE_EXEC -> pathMapper.map(artifact.getExecPath());
          case MODE_RLOCATION -> {
            PathFragment runfilesPath = artifact.getRunfilesPath();
            if (runfilesPath.startsWith(LabelConstants.EXTERNAL_RUNFILES_PATH_PREFIX)) {
              yield runfilesPath.relativeTo(LabelConstants.EXTERNAL_RUNFILES_PATH_PREFIX);
            } else {
              yield PathFragment.create(workspaceRunfilesDirectory).getRelative(runfilesPath);
            }
          }
          default -> throw new IllegalStateException("unknown mode: " + mode);
        };
    return path.getCallablePathString();
  }
}
