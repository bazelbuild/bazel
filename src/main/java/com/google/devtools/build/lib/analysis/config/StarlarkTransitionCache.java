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

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionUtil;
import com.google.devtools.build.lib.analysis.starlark.StarlarkBuildSettingsDetailsValue;
import com.google.devtools.build.lib.analysis.starlark.StarlarkTransition;
import com.google.devtools.build.lib.analysis.starlark.StarlarkTransition.TransitionException;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import java.util.Map;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * Caches the application of transitions that use Starlark.
 *
 * <p>This trivially includes {@link StarlarkTransition}s. But it also includes transitions that
 * delegate to {@link StarlarkTransition}s, like some {@link
 * com.google.devtools.build.lib.analysis.config.transitions.ComposingTransition}s.
 *
 * <p>This cache was added to keep builds that heavily rely on Starlark transitions performant. The
 * inspiring build is a large Apple binary that heavily relies on {@code objc_library.bzl}, which
 * applies a self-transition. The build applies this transition ~600,000 times. Each application has
 * a cost, mostly from setup in translating Java objects to Starlark objects in {@link
 * com.google.devtools.build.lib.analysis.starlark.FunctionTransitionUtil#applyAndValidate}. This
 * cache saves most of that work, reducing analysis phase CPU time by 17%.
 */
public final class StarlarkTransitionCache {

  private Cache<Key, Value> cache = Caffeine.newBuilder().softValues().build();

  /**
   * Applies a Starlark transition, possibly returning a cached result.
   *
   * @param fromOptions source options before the transition
   * @param transition the transition itself
   * @param details information from packages about Starlark build settings needed by transition
   * @param eventHandler handler for errors evaluating the transition.
   * @return transition output
   */
  public Map<String, BuildOptions> computeIfAbsent(
      BuildOptions fromOptions,
      ConfigurationTransition transition,
      StarlarkBuildSettingsDetailsValue details,
      ExtendedEventHandler eventHandler)
      throws TransitionException, InterruptedException {
    Key cacheKey = new Key(transition, fromOptions, details);
    Value cachedResult = cache.getIfPresent(cacheKey);
    if (cachedResult != null) {
      if (cachedResult.nonErrorEvents != null) {
        cachedResult.nonErrorEvents.replayOn(eventHandler);
      }
      return cachedResult.result;
    }
    // All code below here only executes on a cache miss and thus should rely only on values that
    // are part of the above cache key or constants that exist throughout the lifetime of the
    // Blaze server instance.
    BuildOptions adjustedOptions =
        StarlarkTransition.addDefaultStarlarkOptions(fromOptions, transition, details);
    // TODO(bazel-team): Add safety-check that this never mutates fromOptions.
    StoredEventHandler handlerWithErrorStatus = new StoredEventHandler();
    Map<String, BuildOptions> result =
        transition.apply(
            TransitionUtil.restrict(transition, adjustedOptions), handlerWithErrorStatus);

    // We use a temporary StoredEventHandler instead of the caller's event handler because
    // StarlarkTransition.validate assumes no errors occurred. We need a StoredEventHandler to be
    // able to check that, and fail out early if there are errors.
    //
    // TODO(bazel-team): harden StarlarkTransition.validate so we can eliminate this step.
    // StarlarkRuleTransitionProviderTest#testAliasedBuildSetting_outputReturnMismatch shows the
    // effect.
    handlerWithErrorStatus.replayOn(eventHandler);
    if (handlerWithErrorStatus.hasErrors()) {
      throw new TransitionException("Errors encountered while applying Starlark transition");
    }
    result = StarlarkTransition.validate(transition, details, result);
    // If the transition errored (like bad Starlark code), this method already exited with an
    // exception so the results won't go into the cache. We still want to collect non-error events
    // like print() output.
    StoredEventHandler nonErrorEvents =
        !handlerWithErrorStatus.isEmpty() ? handlerWithErrorStatus : null;
    cache.put(cacheKey, new Value(result, nonErrorEvents));
    return result;
  }

  public void clear() {
    cache = Caffeine.newBuilder().softValues().build();
  }

  private static final class Key {
    private final ConfigurationTransition transition;
    private final BuildOptions fromOptions;
    private final StarlarkBuildSettingsDetailsValue details;
    private final int hashCode;

    private Key(
        ConfigurationTransition transition,
        BuildOptions fromOptions,
        StarlarkBuildSettingsDetailsValue details) {
      // For rule self-transitions, the transition instance encapsulates both the transition logic
      // and attributes of the target it's attached to. This is important: the same transition in
      // the same configuration applied to distinct targets may produce different outputs. See
      // StarlarkRuleTransitionProvider.FunctionPatchTransition for details.
      this.transition = transition;
      this.fromOptions = fromOptions;
      this.details = details;
      this.hashCode = Objects.hash(transition, fromOptions, details);
    }

    @Override
    public boolean equals(Object other) {
      if (other == this) {
        return true;
      }
      if (!(other instanceof Key otherKey)) {
        return false;
      }
      return this.transition.equals(otherKey.transition)
          && this.fromOptions.equals(otherKey.fromOptions)
          && this.details.equals(otherKey.details);
    }

    @Override
    public int hashCode() {
      return hashCode;
    }
  }

  private static final class Value {
    private final Map<String, BuildOptions> result;
    /**
     * Stores events for successful transitions. Transitions that fail aren't added to the cache.
     * This is meant for non-error events like Starlark {@code print()} output. See {@link
     * com.google.devtools.build.lib.starlark.StarlarkIntegrationTest#testPrintFromTransitionImpl}
     * for a test that covers this.
     *
     * <p>This is null if the transition lacks non-error events.
     */
    @Nullable private final StoredEventHandler nonErrorEvents;

    Value(Map<String, BuildOptions> result, @Nullable StoredEventHandler nonErrorEvents) {
      this.result = result;
      this.nonErrorEvents = nonErrorEvents;
    }
  }
}
