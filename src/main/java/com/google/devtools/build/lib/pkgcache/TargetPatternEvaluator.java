// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.pkgcache;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadHostile;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * A parser for target patterns.  Target patterns are a generalisation of
 * labels to include wildcards for finding all packages recursively
 * beneath some root, and for finding all targets within a package.
 *
 * <p>A list of target patterns implies a union of all the labels of each
 * pattern.  Each item in a list of target patterns may include a prefix
 * negation operator, indicating that the sets of targets for this pattern
 * should be subtracted from the set of targets for the preceding patterns (note
 * this means that order matters).  Thus, the following list of target patterns:
 * <pre>foo/... -foo/bar:all</pre>
 * means "all targets beneath <tt>foo</tt> except for those targets in
 * package <tt>foo/bar</tt>.
 */
@ThreadSafety.ConditionallyThreadSafe // as long as you don't call updateOffset.
public interface TargetPatternEvaluator {
  static FilteringPolicy DEFAULT_FILTERING_POLICY = FilteringPolicies.NO_FILTER;

  /**
   * Attempts to parse an ordered list of target patterns, computing the union of the set of targets
   * represented by each pattern, unless it is preceded by "-", in which case the set difference is
   * computed. Implements the specification described in the class-level comment.
   */
  ResolvedTargets<Target> parseTargetPatternList(
      ExtendedEventHandler eventHandler,
      List<String> targetPatterns,
      FilteringPolicy policy,
      boolean keepGoing)
      throws TargetParsingException, InterruptedException;

  /**
   * Attempts to parse a single target pattern while consulting the package cache to check for the
   * existence of packages and directories and the build targets in them. Implements the
   * specification described in the class-level comment. Returns a {@link ResolvedTargets} object.
   *
   * <p>If an error is encountered, a {@link TargetParsingException} is thrown, unless {@code
   * keepGoing} is set to true. In that case, the returned object will have its error bit set.
   */
  ResolvedTargets<Target> parseTargetPattern(
      ExtendedEventHandler eventHandler, String pattern, boolean keepGoing)
      throws TargetParsingException, InterruptedException;

  /**
   * Attempts to parse and load the given collection of patterns; the returned map contains the
   * results for each pattern successfully parsed.
   *
   * <p>If an error is encountered, a {@link TargetParsingException} is thrown, unless {@code
   * keepGoing} is set to true. In that case, the patterns that failed to load have the error flag
   * set.
   */
  Map<String, ResolvedTargets<Target>> preloadTargetPatterns(
      ExtendedEventHandler eventHandler, Collection<String> patterns, boolean keepGoing)
      throws TargetParsingException, InterruptedException;

  /**
   * Update the parser's offset, given the workspace and working directory.
   *
   * @param relativeWorkingDirectory the working directory relative to the workspace
   */
  @ThreadHostile
  void updateOffset(PathFragment relativeWorkingDirectory);

  /**
   * @return the offset of this parser from the root of the workspace.
   *         Non-absolute package-names will be resolved relative
   *         to this offset.
   */
  @VisibleForTesting
  String getOffset();
}
