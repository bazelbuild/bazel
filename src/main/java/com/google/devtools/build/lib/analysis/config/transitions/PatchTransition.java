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
package com.google.devtools.build.lib.analysis.config.transitions;

import com.google.devtools.build.lib.analysis.config.BuildOptions;
import java.util.Collections;
import java.util.Map;

/**
 * A configuration transition that maps a single input {@link BuildOptions} to a single output
 * {@link BuildOptions}.
 *
 * <p>Also see {@link SplitTransition}, which maps a single input {@link BuildOptions} to possibly
 * multiple {@link BuildOptions}.
 *
 * <p>The concept is simple: given the input configuration's build options, the
 * transition does whatever it wants to them and returns the modified result.
 *
 * <p>Implementations must be stateless: the output must exclusively depend on the
 * input build options and any immutable member fields. Implementations must also override
 * {@link Object#equals} and {@link Object#hashCode} unless exclusively accessed as
 * singletons. For example:
 *
 * <pre>
 * public class MyTransition implements PatchTransition {
 *   public MyTransition INSTANCE = new MyTransition();
 *
 *   private MyTransition() {}
 *
 *   {@literal @}Override
 *   public BuildOptions patch(BuildOptions options) {
 *     BuildOptions toOptions = options.clone();
 *     // Change some setting on toOptions
 *     return toOptions;
 *   }
 * }
 * </pre>
 *
 * <p>For performance reasons, the input options are passed as a <i>reference</i>, not a
 * <i>copy</i>. Implementations should <i>always</i> treat these as immutable, and call
 * {@link com.google.devtools.build.lib.analysis.config.BuildOptions#clone}
 * before making changes. Unfortunately,
 * {@link com.google.devtools.build.lib.analysis.config.BuildOptions} doesn't currently
 * enforce immutability. So care must be taken not to modify the wrong instance.
 */
@FunctionalInterface
public interface PatchTransition extends ConfigurationTransition {
  /**
   * Applies the transition.
   *
   * @param options the options representing the input configuration to this transition. <b>DO NOT
   *     MODIFY THIS VARIABLE WITHOUT CLONING IT FIRST!</b>
   * @return the options representing the desired post-transition configuration
   */
  BuildOptions patch(BuildOptions options);

  @Override
  default Map<String, BuildOptions> apply(BuildOptions buildOptions) {
    return Collections.singletonMap(PATCH_TRANSITION_KEY, patch(buildOptions));
  }

  @Override
  default String reasonForOverride() {
    return "This is a fundamental transition modeling the simple, common case 1-1 options mapping";
  }
}
