// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import com.google.common.base.Function;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import javax.annotation.Nullable;

/**
 * A helper class for android rules.
 *
 * <p>Helps create the java compilation as well as handling the exporting of the java compilation
 * artifacts to the other rules.
 */
public class AndroidCommon {

  /** Set of allowable android directories prefixes. */
  // Based on com.android.resources.ResourceFolderType
  private static final ImmutableSet<String> RESOURCE_DIRECTORY_TYPES =
      ImmutableSet.of(
          "anim",
          "animator",
          "color",
          "drawable",
          "font",
          "interpolator",
          "layout",
          "menu",
          "mipmap",
          "navigation",
          "raw",
          "transition",
          "values",
          "xml");

  /**
   * Finds and validates the resource directory PathFragment from the artifact Path.
   *
   * <p>If the artifact is not a Fileset, the resource directory is presumed to be the second
   * directory from the end. Filesets are expect to have the last directory as the resource
   * directory.
   */
  @Nullable
  private static PathFragment findResourceDir(Artifact artifact) {
    PathFragment fragment = artifact.getExecPath();
    int segmentCount = fragment.segmentCount();
    if (segmentCount < 3) {
      return null;
    }
    // TODO(bazel-team): Expand Fileset to verify, or remove Fileset as an option for resources.
    if (artifact.isFileset() || artifact.isTreeArtifact()) {
      return fragment.subFragment(segmentCount - 1);
    }

    // Check the resource folder type layout.
    // get the prefix of the parent folder of the fragment.
    String parentDirectory = fragment.getSegment(segmentCount - 2);
    int dashIndex = parentDirectory.indexOf('-');
    String androidFolder =
        dashIndex == -1 ? parentDirectory : parentDirectory.substring(0, dashIndex);
    if (!RESOURCE_DIRECTORY_TYPES.contains(androidFolder)) {
      return null;
    }

    return fragment.subFragment(segmentCount - 3, segmentCount - 2);
  }

  @Nullable
  static PathFragment getSourceDirectoryRelativePathFromResource(Artifact resource) {
    PathFragment resourceDir = findResourceDir(resource);
    if (resourceDir == null) {
      return null;
    }
    return trimTo(resource.getRootRelativePath(), resourceDir);
  }

  /**
   * Finds the rightmost occurrence of the needle and returns subfragment of the haystack from left
   * to the end of the occurrence inclusive of the needle.
   *
   * <pre>
   * `Example:
   *   Given the haystack:
   *     res/research/handwriting/res/values/strings.xml
   *   And the needle:
   *     res
   *   Returns:
   *     res/research/handwriting/res
   * </pre>
   */
  static PathFragment trimTo(PathFragment haystack, PathFragment needle) {
    if (needle.equals(PathFragment.EMPTY_FRAGMENT)) {
      return haystack;
    }
    List<String> needleSegments = needle.splitToListOfSegments();
    // Compute the overlap offset for duplicated parts of the needle.
    int[] overlap = new int[needleSegments.size() + 1];
    // Start overlap at -1, as it will cancel out the increment in the search.
    // See http://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm for the
    // details.
    overlap[0] = -1;
    for (int i = 0, j = -1; i < needleSegments.size(); j++, i++, overlap[i] = j) {
      while (j >= 0 && !needleSegments.get(i).equals(needleSegments.get(j))) {
        // Walk the overlap until the bound is found.
        j = overlap[j];
      }
    }
    // TODO(corysmith): reverse the search algorithm.
    // Keep the index of the found so that the rightmost index is taken.
    List<String> haystackSegments = haystack.splitToListOfSegments();
    int found = -1;
    for (int i = 0, j = 0; i < haystackSegments.size(); i++) {

      while (j >= 0 && !haystackSegments.get(i).equals(needleSegments.get(j))) {
        // Not matching, walk the needle index to attempt another match.
        j = overlap[j];
      }
      j++;
      // Needle index is exhausted, so the needle must match.
      if (j == needleSegments.size()) {
        // Record the found index + 1 to be inclusive of the end index.
        found = i + 1;
        // Subtract one from the needle index to restart the search process
        j = j - 1;
      }
    }
    if (found != -1) {
      // Return the subsection of the haystack.
      return haystack.subFragment(0, found);
    }
    throw new IllegalArgumentException(String.format("%s was not found in %s", needle, haystack));
  }

  /** Returns {@link AndroidConfiguration} in given context. */
  private static AndroidConfiguration getAndroidConfig(RuleContext context) {
    return context.getConfiguration().getFragment(AndroidConfiguration.class);
  }

  private static class FlagMatcher implements Predicate<String> {
    private final ImmutableList<String> matching;

    FlagMatcher(ImmutableList<String> matching) {
      this.matching = matching;
    }

    @Override
    public boolean apply(String input) {
      for (String match : matching) {
        if (input.contains(match)) {
          return true;
        }
      }
      return false;
    }
  }

  private enum FlagConverter implements Function<String, String> {
    DX_TO_DEXBUILDER;

    @Override
    public String apply(String input) {
      return input.replace("--no-", "--no");
    }
  }

  private static ImmutableSet<String> normalizeDexopts(Iterable<String> tokenizedDexopts) {
    // Sort and use ImmutableSet to drop duplicates and get fixed (sorted) order.  Fixed order is
    // important so we generate one dex archive per set of flag in create() method, regardless of
    // how those flags are listed in all the top-level targets being built.
    return Streams.stream(tokenizedDexopts)
        .map(FlagConverter.DX_TO_DEXBUILDER)
        .sorted()
        .collect(ImmutableSet.toImmutableSet()); // collector with dedupe
  }

  /**
   * Derives options to use in DexFileMerger actions from the given context and dx flags, where the
   * latter typically come from a {@code dexopts} attribute on a top-level target.
   */
  public static ImmutableSet<String> mergerDexopts(
      RuleContext ruleContext, Iterable<String> tokenizedDexopts) {
    // We don't need an ordered set but might as well.  Note we don't need to worry about coverage
    // builds since the merger doesn't use --no-locals.
    return normalizeDexopts(
        Iterables.filter(
            tokenizedDexopts,
            new FlagMatcher(getAndroidConfig(ruleContext).getDexoptsSupportedInDexMerger())));
  }
}
