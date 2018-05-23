// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.config.transitions;

import com.google.common.base.Verify;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import java.util.List;

/**
 * A configuration split transition; this should be used to transition to multiple configurations
 * simultaneously. Note that the corresponding rule implementations must have special support to
 * handle this.
 */
@ThreadSafety.Immutable
@FunctionalInterface
public interface SplitTransition extends ConfigurationTransition {
  /**
   * Returns the list of {@code BuildOptions} after splitting, or the original options if this
   * split is a noop.
   *
   * <p>Returning an empty or null list triggers a {@link RuntimeException}.
   */
  List<BuildOptions> split(BuildOptions buildOptions);

  /**
   * Calls {@link #split} and throws a {@link RuntimeException} if the split output is empty.
   */
  default List<BuildOptions> checkedSplit(BuildOptions buildOptions) {
    List<BuildOptions> splitOptions = split(buildOptions);
    Verify.verifyNotNull(splitOptions, "Split transition output may not be null");
    Verify.verify(!splitOptions.isEmpty(), "Split transition output may not be empty");
    return splitOptions;
  }

  /**
   * Returns true iff {@code option} and {@splitOptions} are equal.
   *
   * <p>This can be used to determine if a split is a noop.
   */
  static boolean equals(BuildOptions options, List<BuildOptions> splitOptions) {
    return splitOptions.size() == 1 && Iterables.getOnlyElement(splitOptions).equals(options);
  }
}
