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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.StarlarkDefinedConfigTransition.Settings;
import com.google.devtools.build.lib.analysis.config.transitions.ComposingTransitionFactory;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionUtil;
import com.google.devtools.build.lib.analysis.starlark.StarlarkBuildSettingsDetailsValue;
import com.google.devtools.build.lib.analysis.starlark.StarlarkBuildSettingsDetailsValue.CustomExecScopeValue;
import com.google.devtools.build.lib.analysis.starlark.StarlarkTransition;
import com.google.devtools.build.lib.analysis.starlark.StarlarkTransition.StarlarkTransitionVisitor;
import com.google.devtools.build.lib.analysis.starlark.StarlarkTransition.TransitionException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.skyframe.config.ParsedFlagsValue;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Caches the application of transitions that use Starlark.
 *
 * <p>This trivially includes {@link StarlarkTransition}s. But it also includes transitions that
 * delegate to {@link StarlarkTransition}s, like some {@link
 * com.google.devtools.build.lib.analysis.config.transitions.ComposingTransitionFactory} instances.
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
   * Cache of the set of Starlark build settings referenced by a {@link ConfigurationTransition}.
   *
   * <p>This is a separate cache because even if a transition value is evaluated, its Starlark build
   * settings are computed multiple times by {@link TransitionApplier}.
   *
   * <p>Since `--flag_alias` is non-configurable, we can assume that during a single build, the set
   * of starlark build settings for a given transition won't change.
   */
  private Cache<ConfigurationTransition, ImmutableSet<Label>> starlarkBuildSettingsCache =
      Caffeine.newBuilder().softValues().build();

  /**
   * Given a {@link ConfigurationTransition}, decompose (if possible) and find all referenced
   * Starlark build settings.
   *
   * <p>If a transition references a build setting via an alias, this set includes the alias' label
   * and *does not* include the actual label i.e. this method returns all referenced labels exactly
   * as they are.
   *
   * <p>If a flag alias (defined via --flag_alias) is used in the transition, include the starlark
   * flag mapped to this alias.
   */
  public ImmutableSet<Label> getAllStarlarkBuildSettings(
      ConfigurationTransition root, ImmutableMap<String, Label> flagsAliases) {
    var cachedValue = starlarkBuildSettingsCache.getIfPresent(root);
    if (cachedValue != null) {
      return cachedValue;
    }

    ImmutableSet.Builder<Label> keyBuilder = new ImmutableSet.Builder<>();
    try {
      root.visit(
          (StarlarkTransitionVisitor)
              transition ->
                  keyBuilder.addAll(
                      StarlarkTransition.getRelevantStarlarkSettingsFromTransition(
                          transition, flagsAliases, Settings.INPUTS_AND_OUTPUTS)));
    } catch (TransitionException e) {
      // Not actually thrown in the visitor, but declared.
    }

    ImmutableSet<Label> result = keyBuilder.build();
    starlarkBuildSettingsCache.put(root, result);
    return result;
  }

  /**
   * Adds the default values for a transition's input build settings to its input build options. An
   * alias gets the value of the setting it refers to instead if that is set.
   */
  private BuildOptions addDefaultStarlarkOptions(
      BuildOptions fromOptions,
      ImmutableMap<String, Label> flagsAliases,
      ConfigurationTransition transition,
      StarlarkBuildSettingsDetailsValue details) {
    if (details.buildSettingToDefault().isEmpty() && details.customExecScopeValues().isEmpty()) {
      // No need to traverse the transition to find its Starlark flag inputs. There are none.
      return fromOptions;
    }

    BuildOptions.Builder optionsWithDefaults = null;
    for (Label maybeAliasSetting : getAllStarlarkBuildSettings(transition, flagsAliases)) {
      // details will only have the defaults of the actual setting so must unalias
      Label setting = details.aliasToActual().getOrDefault(maybeAliasSetting, maybeAliasSetting);
      if (!fromOptions.getStarlarkOptions().containsKey(maybeAliasSetting)) {
        if (optionsWithDefaults == null) {
          optionsWithDefaults = fromOptions.toBuilder();
        }
        optionsWithDefaults.addStarlarkOption(
            maybeAliasSetting,
            fromOptions
                .getStarlarkOptions()
                .getOrDefault(setting, details.buildSettingToDefault().get(setting)));
      }
    }

    if (ExecutionTransitionFactory.isExecTransition(transition)) {
      return getDefaultStarlarkOptionsForCustomExec(optionsWithDefaults, details, fromOptions);
    }
    return optionsWithDefaults == null ? fromOptions : optionsWithDefaults.build();
  }

  /**
   * Adds the default values of unset flags with a custom exec scope, so that the exec transition
   * sets them to the value of the flag they reference.
   */
  public static BuildOptions getDefaultStarlarkOptionsForCustomExec(
      BuildOptions.Builder optionsWithDefaults,
      StarlarkBuildSettingsDetailsValue details,
      BuildOptions fromOptions) {
    for (CustomExecScopeValue customExecScopeValue : details.customExecScopeValues().values()) {
      if (!fromOptions.getStarlarkOptions().containsKey(customExecScopeValue.flag())) {
        if (optionsWithDefaults == null) {
          optionsWithDefaults = fromOptions.toBuilder();
        }
        optionsWithDefaults.addStarlarkOption(
            customExecScopeValue.flag(), customExecScopeValue.flagDefault());
      }
    }

    return optionsWithDefaults == null ? fromOptions : optionsWithDefaults.build();
  }

  /**
   * The flags of a target platform.
   *
   * @param flags the flags, which are empty for a platform without flags
   * @param details the details of the Starlark flags in {@code flags} if the transition composes
   *     the exec transition, which propagates them based on their scope, or {@link
   *     StarlarkBuildSettingsDetailsValue#EMPTY}
   */
  public record PlatformFlags(
      Optional<ParsedFlagsValue> flags, StarlarkBuildSettingsDetailsValue details) {}

  /** The result of {@link #computeIfAbsent}. */
  public sealed interface Result {}

  /**
   * The options a transition results in, by key.
   *
   * @param platformChangedKeys the keys of the options whose target platform changed since the
   *     flags of a platform were last applied to them, see {@link #changesTargetPlatform}
   */
  public record Applied(
      ImmutableMap<String, BuildOptions> options, ImmutableSet<String> platformChangedKeys)
      implements Result {}

  /**
   * The platforms whose flags have to be looked up before the transition can be applied.
   *
   * @param events the events of the incomplete application of the transition, which callers replay
   *     if they can't look up the platforms
   */
  public record MissingPlatforms(ImmutableSet<Label> platforms, @Nullable StoredEventHandler events)
      implements Result {}

  /**
   * Applies a Starlark transition, possibly returning a cached result.
   *
   * <p>If a transition in a composition changes the target platform, the flags of the new platform
   * are applied to its output before the next transition is applied. Callers apply them to the
   * output of the last transition, see {@link Applied#platformChangedKeys}.
   *
   * @param fromOptions source options before the transition
   * @param transition the transition itself
   * @param details information from packages about Starlark build settings needed by transition
   * @param platformFlags the flags of the target platforms looked up so far
   * @param eventHandler handler for errors evaluating the transition.
   * @return the transition output or the platforms whose flags have to be added to {@code
   *     platformFlags} first
   */
  public Result computeIfAbsent(
      BuildOptions fromOptions,
      ConfigurationTransition transition,
      StarlarkBuildSettingsDetailsValue details,
      Map<Label, PlatformFlags> platformFlags,
      ExtendedEventHandler eventHandler)
      throws TransitionException, InterruptedException {
    Key cacheKey = new Key(transition, fromOptions, details);
    Value cachedResult = cache.getIfPresent(cacheKey);
    if (cachedResult != null) {
      // The result also depends on the flags of the platforms it records, which callers have to
      // look up so that they depend on them too. They are checked in the order in which they were
      // applied since the first one that differs may change which platforms follow.
      boolean upToDate = true;
      for (var entry : cachedResult.platformFlags.entrySet()) {
        PlatformFlags flags = platformFlags.get(entry.getKey());
        if (flags == null) {
          return new MissingPlatforms(ImmutableSet.of(entry.getKey()), /* events= */ null);
        }
        if (!flags.equals(entry.getValue())) {
          upToDate = false;
          break;
        }
      }
      if (upToDate) {
        if (cachedResult.nonErrorEvents != null) {
          cachedResult.nonErrorEvents.replayOn(eventHandler);
        }
        return cachedResult.result;
      }
    }

    ImmutableMap<String, Label> flagsAliases =
        fromOptions.get(CoreOptions.class).getCommandLineFlagAliasesMap();

    // All code below here only executes on a cache miss and thus should rely only on values that
    // are part of the above cache key, the platform flags recorded in the cached value or
    // constants that exist throughout the lifetime of the Blaze server instance.
    StoredEventHandler handlerWithErrorStatus = new StoredEventHandler();
    Map<Label, PlatformFlags> usedPlatformFlags = new LinkedHashMap<>();
    Set<Label> missingPlatforms = new LinkedHashSet<>();
    Map<String, TransitionedOptions> result;
    boolean incomplete = false;
    try {
      // Each transition in a composition is applied and validated on its own. This is also
      // necessary since the exec transition drops the Starlark options it doesn't propagate.
      result =
          ComposingTransitionFactory.applyComposed(
              transition,
              new TransitionedOptions(
                  fromOptions,
                  /* platformChanged= */ false,
                  StarlarkBuildSettingsDetailsValue.EMPTY),
              (composedTransition, input) -> {
                BuildOptions options = input.options();
                StarlarkBuildSettingsDetailsValue platformFlagDetails = input.platformFlagDetails();
                Label platform = input.platformChanged() ? targetPlatform(options) : null;
                if (platform != null) {
                  PlatformFlags flags = platformFlags.get(platform);
                  if (flags == null) {
                    missingPlatforms.add(platform);
                    return null;
                  }
                  usedPlatformFlags.put(platform, flags);
                  options = applyPlatformFlags(platform, flags, options);
                  if (platformFlagDetails == StarlarkBuildSettingsDetailsValue.EMPTY) {
                    platformFlagDetails = flags.details();
                  } else if (flags.details() != StarlarkBuildSettingsDetailsValue.EMPTY) {
                    platformFlagDetails = platformFlagDetails.union(flags.details());
                  }
                }
                ConfigurationTransition transitionToApply = composedTransition;
                if (composedTransition != transition
                    && ExecutionTransitionFactory.isExecTransition(composedTransition)) {
                  // The exec transition only has the details of the source configuration's flags,
                  // but earlier transitions in the composition and the platforms they change to
                  // may set others.
                  transitionToApply =
                      ExecutionTransitionFactory.withAdditionalScopeDetails(
                          composedTransition,
                          platformFlagDetails == StarlarkBuildSettingsDetailsValue.EMPTY
                              ? details
                              : details.union(platformFlagDetails));
                }
                Map<String, BuildOptions> outputs =
                    applyAndValidate(
                        transitionToApply, options, flagsAliases, details, handlerWithErrorStatus);
                ImmutableMap.Builder<String, TransitionedOptions> transitionedOptions =
                    ImmutableMap.builderWithExpectedSize(outputs.size());
                for (Map.Entry<String, BuildOptions> output : outputs.entrySet()) {
                  transitionedOptions.put(
                      output.getKey(),
                      new TransitionedOptions(
                          output.getValue(),
                          changesTargetPlatform(composedTransition, options, output.getValue()),
                          platformFlagDetails));
                }
                return transitionedOptions.buildOrThrow();
              });
      incomplete = result == null;
    } finally {
      if (!incomplete) {
        handlerWithErrorStatus.replayOn(eventHandler);
      }
    }
    if (incomplete) {
      return new MissingPlatforms(ImmutableSet.copyOf(missingPlatforms), handlerWithErrorStatus);
    }
    Applied applied =
        new Applied(
            ImmutableMap.copyOf(Maps.transformValues(result, TransitionedOptions::options)),
            ImmutableSet.copyOf(
                Maps.filterValues(result, TransitionedOptions::platformChanged).keySet()));
    // If the transition errored (like bad Starlark code), this method already exited with an
    // exception so the results won't go into the cache. We still want to collect non-error events
    // like print() output.
    StoredEventHandler nonErrorEvents =
        !handlerWithErrorStatus.isEmpty() ? handlerWithErrorStatus : null;
    cache.put(cacheKey, new Value(applied, ImmutableMap.copyOf(usedPlatformFlags), nonErrorEvents));
    return applied;
  }

  /**
   * Returns whether {@code transition}, which isn't composed, changed the target platform from
   * {@code fromOptions} to {@code toOptions}. This includes the exec transition, which resets
   * options to their defaults.
   */
  public static boolean changesTargetPlatform(
      ConfigurationTransition transition, BuildOptions fromOptions, BuildOptions toOptions) {
    return ExecutionTransitionFactory.isExecTransition(transition)
        || !targetPlatforms(fromOptions).equals(targetPlatforms(toOptions));
  }

  private static List<Label> targetPlatforms(BuildOptions options) {
    PlatformOptions platformOptions = options.get(PlatformOptions.class);
    return platformOptions == null ? ImmutableList.of() : platformOptions.getPlatforms();
  }

  /** Returns the single target platform of {@code options}, or null if there isn't one. */
  @Nullable
  private static Label targetPlatform(BuildOptions options) {
    List<Label> platforms = targetPlatforms(options);
    return platforms.size() == 1 ? platforms.getFirst() : null;
  }

  private static BuildOptions applyPlatformFlags(
      Label platform, PlatformFlags flags, BuildOptions options) throws TransitionException {
    if (flags.flags().isEmpty()) {
      return options;
    }
    BuildOptions result;
    try {
      result = flags.flags().get().mergeWith(options).getOptions();
    } catch (OptionsParsingException e) {
      throw new TransitionException(e.getMessage(), e);
    }
    // Later transitions in the composition read flags through the aliases of the source options.
    if (!result
        .get(CoreOptions.class)
        .getCommandLineFlagAliases()
        .equals(options.get(CoreOptions.class).getCommandLineFlagAliases())) {
      throw new TransitionException(
          String.format(
              "%s can't set --flag_alias since a transition changes the target platform to it"
                  + " before applying another transition",
              platform));
    }
    return result;
  }

  /**
   * Options produced by a transition in a composition.
   *
   * @param platformFlagDetails the details of the Starlark flags set by the platforms whose flags
   *     were applied to the options
   */
  private record TransitionedOptions(
      BuildOptions options,
      boolean platformChanged,
      StarlarkBuildSettingsDetailsValue platformFlagDetails) {}

  private Map<String, BuildOptions> applyAndValidate(
      ConfigurationTransition transition,
      BuildOptions fromOptions,
      ImmutableMap<String, Label> flagsAliases,
      StarlarkBuildSettingsDetailsValue details,
      StoredEventHandler handlerWithErrorStatus)
      throws TransitionException, InterruptedException {
    BuildOptions adjustedOptions =
        addDefaultStarlarkOptions(fromOptions, flagsAliases, transition, details);
    // TODO(bazel-team): Add safety-check that this never mutates fromOptions.
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
    if (handlerWithErrorStatus.hasErrors()) {
      throw new TransitionException("Errors encountered while applying Starlark transition");
    }
    return StarlarkTransition.validate(transition, details, flagsAliases, result);
  }

  public void clear() {
    cache = Caffeine.newBuilder().softValues().build();
    starlarkBuildSettingsCache = Caffeine.newBuilder().softValues().build();
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
    private final Applied result;

    /** The flags of the platforms that transitions in the composition changed to. */
    private final ImmutableMap<Label, PlatformFlags> platformFlags;

    /**
     * Stores events for successful transitions. Transitions that fail aren't added to the cache.
     * This is meant for non-error events like Starlark {@code print()} output. See {@link
     * com.google.devtools.build.lib.starlark.StarlarkIntegrationTest#testPrintFromTransitionImpl}
     * for a test that covers this.
     *
     * <p>This is null if the transition lacks non-error events.
     */
    @Nullable private final StoredEventHandler nonErrorEvents;

    Value(
        Applied result,
        ImmutableMap<Label, PlatformFlags> platformFlags,
        @Nullable StoredEventHandler nonErrorEvents) {
      this.result = result;
      this.platformFlags = platformFlags;
      this.nonErrorEvents = nonErrorEvents;
    }
  }
}
