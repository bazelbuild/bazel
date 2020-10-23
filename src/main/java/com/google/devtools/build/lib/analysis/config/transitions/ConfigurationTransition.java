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

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.ClassName;
import java.util.Map;

/**
 * A configuration transition.
 */
public interface ConfigurationTransition {
  /**
   * A designated key string for patch transitions. See {@link ConfigurationTransition#apply} for
   * its usage.
   */
  String PATCH_TRANSITION_KEY = "";

  /**
   * Declares the {@link FragmentOptions} this transition may read.
   *
   * <p>Blaze throws an {@link IllegalArgumentException} if {@link #apply} is called on an options
   * fragment that isn't declared here.
   */
  default ImmutableSet<Class<? extends FragmentOptions>> requiresOptionFragments() {
    return ImmutableSet.of();
  }

  /**
   * {@link #requiresOptionFragments()} variation for Starlark transitions, which need a {@link
   * BuildOptions} instance to map required options to their {@link FragmentOptions}.
   *
   * <p>This version pre-translates the fragments to their final string representations. This is
   * because Starlark transitions can depend on both native fragments and Starlark flags. The latter
   * are reported directly, not as part of a larger fragment.
   *
   * <p>Non-Starlark transitions should override {@link #requiresOptionFragments()} and ignore this.
   *
   * <p>Callers may ignore this method if they know they're not calling into a Starlark transition.
   */
  default ImmutableSet<String> requiresOptionFragments(BuildOptions options) {
    return requiresOptionFragments().stream()
        .map(ClassName::getSimpleNameWithOuter)
        .collect(toImmutableSet());
  }

  /**
   * Returns the map of {@code BuildOptions} after applying this transition. The returned map keys
   * are only used for dealing with split transitions. Patch transitions, including internal, native
   * Patch transitions, should return a single entry map with key {@code PATCH_TRANSITION_KEY}.
   *
   * <p>Blaze throws an {@link IllegalArgumentException} if this method reads any options fragment
   * not declared in {@link #requiresOptionFragments}.
   *
   * <p>Returning an empty or null map triggers a {@link RuntimeException}.
   */
  Map<String, BuildOptions> apply(BuildOptionsView buildOptions, EventHandler eventHandler)
      throws InterruptedException;

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
