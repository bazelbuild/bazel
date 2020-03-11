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
import java.util.Collection;
import java.util.Map;

/**
 * A configuration transition that maps a single input {@link BuildOptions} to possibly multiple
 * output {@link BuildOptions}. This provides the ability to transition to multiple configurations
 * simultaneously.
 *
 * <p>Also see {@link PatchTransition}, which maps a single input {@BuildOptions} to a single
 * output. If your transition never needs to produce multiple outputs, you should use a
 * {@link PatchTransition}.
 *
 * Corresponding rule implementations may require special support to handle this in an organized
 * way (e.g. for determining which CPU corresponds to which dep for a multi-arch split dependency).
 */
@ThreadSafety.Immutable
@FunctionalInterface
public interface SplitTransition extends ConfigurationTransition {
  /**
   * Returns the map of {@code BuildOptions} after splitting, or the original options if this split
   * is a noop. The key values are used as dict keys in ctx.split_attr, so human-readable strings
   * are recommended.
   *
   * <p>Returning an empty or null list triggers a {@link RuntimeException}.
   */
  Map<String, BuildOptions> split(BuildOptions buildOptions);

  /**
   * Returns true iff {@code option} and {@splitOptions} are equal.
   *
   * <p>This can be used to determine if a split is a noop.
   */
  static boolean equals(BuildOptions options, Collection<BuildOptions> splitOptions) {
    return splitOptions.size() == 1 && Iterables.getOnlyElement(splitOptions).equals(options);
  }

  @Override
  default Map<String, BuildOptions> apply(BuildOptions buildOptions) {
    Map<String, BuildOptions> splitOptions = split(buildOptions);
    Verify.verifyNotNull(splitOptions, "Split transition output may not be null");
    Verify.verify(!splitOptions.isEmpty(), "Split transition output may not be empty");
    return splitOptions;
  }

  @Override
  default String reasonForOverride() {
    return "This is a fundamental transition modeling the need for multiply configured deps";
  }
}
