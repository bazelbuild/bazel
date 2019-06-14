// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.Map;

/**
 * A preloader for target patterns. Target patterns are a generalisation of labels to include
 * wildcards for finding all packages recursively beneath some root, and for finding all targets
 * within a package.
 *
 * <p>A list of target patterns implies a union of all the labels of each pattern. Each item in a
 * list of target patterns may include a prefix negation operator, indicating that the sets of
 * targets for this pattern should be subtracted from the set of targets for the preceding patterns
 * (note this means that order matters). Thus, the following list of target patterns:
 *
 * <pre>foo/... -foo/bar:all</pre>
 *
 * means "all targets beneath <tt>foo</tt> except for those targets in package <tt>foo/bar</tt>.
 */
@ThreadSafety.ThreadSafe
public interface TargetPatternPreloader {
  /**
   * Attempts to parse and load the given collection of patterns; the returned map contains the
   * results for each pattern successfully parsed. As a side effect, calling this method populates
   * the Skyframe graph, so subsequent calls are faster.
   *
   * <p>If an error is encountered, a {@link TargetParsingException} is thrown, unless {@code
   * keepGoing} is set to true. In that case, the patterns that failed to load have the error flag
   * set.
   */
  Map<String, Collection<Target>> preloadTargetPatterns(
      ExtendedEventHandler eventHandler,
      PathFragment relativeWorkingDirectory,
      Collection<String> patterns,
      boolean keepGoing,
      boolean useForkJoinPool)
      throws TargetParsingException, InterruptedException;
}
