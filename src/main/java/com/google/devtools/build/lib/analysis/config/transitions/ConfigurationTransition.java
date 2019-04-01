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

import com.google.devtools.build.lib.analysis.config.BuildOptions;
import java.util.List;

/**
 * A configuration transition.
 */
public interface ConfigurationTransition {
  /**
   * Returns the list of {@code BuildOptions} after applying this transition.
   *
   * <p>Returning an empty or null list triggers a {@link RuntimeException}.
   */
  List<BuildOptions> apply(BuildOptions buildOptions);

  /**
   * We want to keep the number of transition interfaces no larger than what's necessary to maintain
   * a clear configuration API.
   *
   * <p>This method provides a speed bump against creating new interfaces too casually. While we
   * could provide stronger enforcement by making {@link ConfigurationTransition} an abstract class
   * with a limited access constructor, keeping it as an interface supports defining transitions
   * with lambdas.
   *
   * <p>If you're considering adding a new override, contact bazel-dev@googlegroups.com to discuss.
   */
  @SuppressWarnings("unused")
  String reasonForOverride();

  /**
   * Does this transition switch to a "host" configuration?
   */
  default boolean isHostTransition() {
    return false;
  }

  default String getName() {
    return this.getClass().getSimpleName();
  }

  /** Allows the given {@link Visitor} to inspect this transition. */
  default <E extends Exception> void visit(Visitor<E> visitor) throws E {
    visitor.accept(this);
  }

  /** Helper object that can be used to inspect {@link ConfigurationTransition} instances. */
  @FunctionalInterface
  interface Visitor<E extends Exception> {
    void accept(ConfigurationTransition transition) throws E;
  }
}
