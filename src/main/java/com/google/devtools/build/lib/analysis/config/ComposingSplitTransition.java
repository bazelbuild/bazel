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

package com.google.devtools.build.lib.analysis.config;

import com.google.common.base.Verify;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.Attribute.SplitTransition;
import com.google.devtools.build.lib.packages.Attribute.Transition;
import java.util.List;

/**
 * A split transition that combines a Transition with a {@link PatchTransition}.  The patch is
 * applied first, followed by the Transition.
 *
 * <p>We implement a {@link SplitTransition} here since that abstraction can capture all possible
 * composed transitions - both those that produce multiple output configurations and those that
 * do not.
 */
public class ComposingSplitTransition implements SplitTransition<BuildOptions> {

  private PatchTransition patch;
  private Transition transition;

  /**
   * Creates a {@link ComposingSplitTransition} with the given {@link Transition} and
   * {@link PatchTransition}.
   */
  public ComposingSplitTransition(PatchTransition patch, Transition transition) {
    this.patch = patch;
    this.transition = transition;
  }

  @Override
  public List<BuildOptions> split(BuildOptions buildOptions) {
    BuildOptions patchedOptions = patch.apply(buildOptions);
    if (transition == ConfigurationTransition.NONE) {
      return ImmutableList.<BuildOptions>of(patchedOptions);
    } else if (transition instanceof PatchTransition) {
      return ImmutableList.<BuildOptions>of(((PatchTransition) transition).apply(patchedOptions));
    } else if (transition instanceof SplitTransition) {
      SplitTransition split = (SplitTransition<BuildOptions>) transition;
      List<BuildOptions> splitOptions = split.split(patchedOptions);
      if (splitOptions.isEmpty()) {
        Verify.verify(split.defaultsToSelf());
        return ImmutableList.of(patchedOptions);
      } else {
        return splitOptions;
      }
    } else {
      throw new IllegalStateException(
          String.format("Unsupported composite transition type: %s",
              transition.getClass().getName()));
    }
  }

  @Override
  public boolean defaultsToSelf() {
    return true;
  }
}

