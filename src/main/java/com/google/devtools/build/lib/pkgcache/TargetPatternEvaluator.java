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

import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;

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
// TODO(ulfjack): Delete this interface - it's only used in tests.
public interface TargetPatternEvaluator {
  static FilteringPolicy DEFAULT_FILTERING_POLICY = FilteringPolicies.NO_FILTER;

  /**
   * Attempts to parse an ordered list of target patterns, computing the union of the set of targets
   * represented by each pattern, unless it is preceded by "-", in which case the set difference is
   * computed. Implements the specification described in the class-level comment.
   */
  ResolvedTargets<Target> parseTargetPatternList(
      PathFragment relativeWorkingDirectory,
      ExtendedEventHandler eventHandler,
      List<String> targetPatterns,
      FilteringPolicy policy,
      boolean keepGoing)
      throws TargetParsingException, InterruptedException;
}
