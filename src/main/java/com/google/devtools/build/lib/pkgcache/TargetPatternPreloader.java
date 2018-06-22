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

import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.Map;

/**
 * A preloader for target patterns. See {@link TargetPatternEvaluator} for more details.
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
  Map<String, ResolvedTargets<Target>> preloadTargetPatterns(
      ExtendedEventHandler eventHandler,
      PathFragment relativeWorkingDirectory,
      Collection<String> patterns,
      boolean keepGoing)
          throws TargetParsingException, InterruptedException;
}
