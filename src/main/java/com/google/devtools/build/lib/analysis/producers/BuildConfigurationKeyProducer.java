// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.producers;

import com.google.common.base.Preconditions;
import com.google.common.base.Verify;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.Scope;
import com.google.devtools.build.lib.analysis.config.transitions.BaselineOptionsValue;
import com.google.devtools.build.lib.analysis.platform.PlatformValue;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.BuildOptionsScopeFunction.BuildOptionsScopeFunctionException;
import com.google.devtools.build.lib.skyframe.BuildOptionsScopeValue;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.config.ParsedFlagsValue;
import com.google.devtools.build.lib.skyframe.config.PlatformMappingException;
import com.google.devtools.build.lib.skyframe.config.PlatformMappingValue;
import com.google.devtools.build.lib.skyframe.toolchains.PlatformLookupUtil.InvalidPlatformException;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.StateMachine;
import com.google.devtools.build.skyframe.state.StateMachine.ValueOrExceptionSink;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/**
 * Creates the needed {@link BuildConfigurationKey} instance for a single {@link BuildOptions},
 * including merging in any platform-based flags or a platform mapping.
 *
 * <p>Platform-based flags and platform mappings are mutually exclusive: only one will be applied if
 * they are present. Trying to mix and match would be possible but confusing, especially if they try
 * to change the same flag. The logic is:
 *
 * <ul>
 *   <li>If {@link PlatformOptions#platforms} specifies a target platform, look up the {@link
 *       PlatformValue}. If it specifies {@linkplain PlatformValue#parsedFlags flags}, use {@link
 *       ParsedFlagsValue#mergeWith}.
 *   <li>If {@link PlatformOptions#platforms} does not specify a target platform, or if the target
 *       platform does not specify {@linkplain PlatformValue#parsedFlags flags}, look up the {@link
 *       PlatformMappingValue} and use {@link PlatformMappingValue#map}.
 * </ul>
 *
 * <p>Scopes for starlark flags also get applied before producing the final BuildConfigurationKey.
 * Scopes are applied after platform-based flags or platform mappings are applied. The logic is:
 *
 * <ul>
 *   <li>The scopes of the starlark flags are taken from the configuration the transition was
 *       applied to, which almost always sets the same flags. Otherwise they are looked up via
 *       {@link BuildOptionsScopeFunction}.
 *   <li>If no starlark flag has ScopeType.PROJECT, no further processing is done.
 *   <li>If the ScopeType for a flag is ScopeType.PROJECT, and the flag is not in the scope of the
 *       current package, the flag is reset to its baseline value if it is present in the baseline.
 *       If the flag is not present in the baseline, it is removed. This is to ensure that we do not
 *       trigger an addition ST-<hash>, which defeats the purpose of scoping.
 *   <li>If the ScopeType for a flag is ScopeType.PROJECT, and the flag is in the scope of the
 *       current package, the flag keeps its current value.
 * </ul>
 *
 * @param <C> The type of the context variable that the producer will pass via the {@link
 *     ResultSink} so that consumers can identify which options are which.
 */
public final class BuildConfigurationKeyProducer<C>
    implements StateMachine,
        ValueOrExceptionSink<PlatformMappingException>,
        Consumer<SkyValue>,
        PlatformProducer.ResultSink {

  /** Interface for clients to accept results of this computation. */
  public interface ResultSink<C> {

    void acceptOptionsParsingError(OptionsParsingException e);

    void acceptPlatformMappingError(PlatformMappingException e);

    void acceptPlatformFlagsError(InvalidPlatformException error);

    void acceptBuildOptionsScopeFunctionError(BuildOptionsScopeFunctionException e);

    void acceptTransitionedConfiguration(C context, BuildConfigurationKey transitionedOptionKey);
  }

  // -------------------- Input --------------------
  private final ResultSink<C> sink;
  private final StateMachine runAfter;
  private final C context;
  private final BuildOptions options;
  private final boolean forBaseline;
  private final BuildOptionsScopeValue scopesFromSourceConfiguration;
  private final Label label;

  // -------------------- Internal State --------------------
  private PlatformValue targetPlatformValue;
  private PlatformMappingValue platformMappingValue;
  private BuildOptionsScopeValue buildOptionsScopeValue;
  private BuildOptions postPlatformProcessedOptions;
  private BuildOptions baselineConfiguration;

  /**
   * @param scopesFromSourceConfiguration the scopes already resolved for the configuration {@code
   *     options} was transitioned from, or {@link BuildOptionsScopeValue#EMPTY} if the caller has
   *     none. Transitions rarely introduce Starlark flags, so this usually covers {@code options}
   *     too and spares this producer a Skyframe lookup on every dependency edge.
   */
  BuildConfigurationKeyProducer(
      ResultSink<C> sink,
      StateMachine runAfter,
      C context,
      BuildOptions options,
      boolean forBaseline,
      BuildOptionsScopeValue scopesFromSourceConfiguration,
      Label label) {
    this.sink = sink;
    this.runAfter = runAfter;
    this.context = context;
    this.options = options;
    this.forBaseline = forBaseline;
    this.scopesFromSourceConfiguration = scopesFromSourceConfiguration;
    this.label = label;
  }

  @Override
  public StateMachine step(Tasks tasks) {
    // Resolve the scopes of the flags set before platform processing here rather than in
    // findBuildOptionsScopes so that any lookup shares a Skyframe batch with the platform lookups
    // below instead of adding a round of its own. Platform-based flags and platform mappings
    // hardly ever add Starlark flags; findBuildOptionsScopes requests the right value if they do.
    requestScopes(tasks, options);

    // Short-circuit if there are no platform options.
    var platformOptions = options.get(PlatformOptions.class);
    if (platformOptions == null) {
      this.postPlatformProcessedOptions = options;
      return this::findBuildOptionsScopes;
    }

    List<Label> targetPlatforms = platformOptions.getPlatforms();
    if (targetPlatforms.size() == 1) {
      // TODO: https://github.com/bazelbuild/bazel/issues/19807 - We define this flag to only use
      //  the first value and ignore any subsequent ones. Remove this check as part of cleanup.
      tasks.enqueue(
          new PlatformProducer(
              targetPlatforms.getFirst(),
              options.get(CoreOptions.class).getCommandLineFlagAliasesMap(),
              this,
              this::checkTargetPlatformFlags));
      return runAfter;
    } else {
      Verify.verify(targetPlatforms.isEmpty());
      return this::mergeFromPlatformMapping;
    }
  }

  /**
   * Determine whether to update the BuildOptions with platform-based flags via {@link
   * ParsedFlagsValue#mergeWith} or with platform mappings via {@link PlatformMappingValue#map}
   * based on the presence of {@link ParsedFlagsValue}.
   */
  private StateMachine checkTargetPlatformFlags(Tasks tasks) {
    if (targetPlatformValue == null) {
      return DONE; // Error.
    }
    Optional<ParsedFlagsValue> parsedFlags = targetPlatformValue.parsedFlags();
    if (parsedFlags.isPresent()) {
      this.postPlatformProcessedOptions = parsedFlags.get().mergeWith(options).getOptions();
      return this::findBuildOptionsScopes;
    } else {
      return this::mergeFromPlatformMapping;
    }
  }

  /**
   * Makes the {@link Scope}s of {@code buildOptions}' Starlark flags available to {@link
   * #possiblyApplyScopes}, looking up a {@link BuildOptionsScopeValue} via {@link
   * BuildOptionsScopeFunction} unless a value that answers for all of them is already at hand.
   */
  private void requestScopes(Tasks tasks, BuildOptions buildOptions) {
    ImmutableSet<Label> starlarkFlags = buildOptions.getStarlarkOptions().keySet();
    if (starlarkFlags.isEmpty() || forBaseline) {
      return;
    }
    if (scopesFromSourceConfiguration.covers(starlarkFlags)) {
      this.buildOptionsScopeValue = scopesFromSourceConfiguration;
      return;
    }
    tasks.lookUp(BuildOptionsScopeValue.Key.create(starlarkFlags), (Consumer<SkyValue>) this);
  }

  /**
   * Requests the scopes of the Starlark flags {@link postPlatformProcessedOptions} gained during
   * platform processing, if any: {@link #step} only requested the ones set before it.
   */
  private StateMachine findBuildOptionsScopes(Tasks tasks) {
    Preconditions.checkNotNull(this.postPlatformProcessedOptions);
    if (buildOptionsScopeValue == null
        || !buildOptionsScopeValue.covers(
            postPlatformProcessedOptions.getStarlarkOptions().keySet())) {
      requestScopes(tasks, postPlatformProcessedOptions);
    }
    return this::possiblyApplyScopes;
  }

  /**
   * Performs a lookup for {@link PlatformMappingValue} via {@link PlatformMappingFunction} given
   * {@link options} and will transform the input {@link BuildOptions} with any matching platform
   * mappings.
   */
  private StateMachine mergeFromPlatformMapping(Tasks tasks) {
    tasks.lookUp(
        options.get(PlatformOptions.class).getPlatformMappingKey(),
        PlatformMappingException.class,
        this);
    return this::applyPlatformMapping;
  }

  private StateMachine applyPlatformMapping(Tasks tasks) {
    if (platformMappingValue == null) {
      return DONE; // Error.
    }
    try {
      this.postPlatformProcessedOptions = platformMappingValue.map(options).getOptions();
      return this::findBuildOptionsScopes;
    } catch (OptionsParsingException e) {
      sink.acceptOptionsParsingError(e);
      return runAfter;
    }
  }

  // Handles results from the PlatformMappingValueKey lookup.
  @Override
  public void acceptValueOrException(
      @Nullable SkyValue value, @Nullable PlatformMappingException exception) {
    if (value == null && exception == null) {
      throw new IllegalStateException("No value or exception was provided");
    }
    if (value != null && exception != null) {
      throw new IllegalStateException("Both value and exception were provided");
    }

    if (exception != null) {
      sink.acceptPlatformMappingError(exception);
    } else {
      this.platformMappingValue = (PlatformMappingValue) value;
    }
  }

  @Override
  public void acceptPlatformValue(PlatformValue value) {
    this.targetPlatformValue = value;
  }

  @Override
  public void acceptPlatformInfoError(InvalidPlatformException error) {
    sink.acceptPlatformFlagsError(error);
  }

  @Override
  public void acceptOptionsParsingError(OptionsParsingException error) {
    sink.acceptOptionsParsingError(error);
  }

  @Override
  public void accept(SkyValue value) {
    this.buildOptionsScopeValue = (BuildOptionsScopeValue) value;
  }

  private StateMachine possiblyApplyScopes(Tasks tasks) {
    // A null value here doesn't mean a Skyframe lookUp came back empty: it means no scopes were
    // requested at all, because these options set no Starlark flags or this is the baseline.
    if (buildOptionsScopeValue == null
        || postPlatformProcessedOptions.getStarlarkOptions().isEmpty()) {
      return finishConfigurationKeyProcessing(postPlatformProcessedOptions);
    }

    // The scopes may have been resolved for a superset of these options' Starlark flags, so check
    // that at least one of the flags actually set here is project-scoped.
    ImmutableMap<Label, Scope> projectScopes = buildOptionsScopeValue.projectScopes();
    boolean shouldApplyScopes =
        !projectScopes.isEmpty()
            && postPlatformProcessedOptions.getStarlarkOptions().keySet().stream()
                .anyMatch(projectScopes::containsKey);

    if (!shouldApplyScopes) {
      return finishConfigurationKeyProcessing(postPlatformProcessedOptions);
    }

    tasks.lookUp(
        BaselineOptionsValue.key(
            postPlatformProcessedOptions.get(CoreOptions.class).getIsExec(),
            !postPlatformProcessedOptions.contains(TestConfiguration.TestOptions.class),
            /* newPlatform= */ null),
        val -> this.baselineConfiguration = ((BaselineOptionsValue) val).toOptions());
    return this::applyScopes;
  }

  private StateMachine applyScopes(Tasks tasks) {
    BuildOptions finalBuildOptions =
        baselineConfiguration
                .getStarlarkOptions()
                .equals(postPlatformProcessedOptions.getStarlarkOptions())
            ? postPlatformProcessedOptions
            : resetFlags(
                buildOptionsScopeValue, postPlatformProcessedOptions, baselineConfiguration, label);
    return finishConfigurationKeyProcessing(finalBuildOptions);
  }

  private StateMachine finishConfigurationKeyProcessing(BuildOptions finalBuildOptions) {
    sink.acceptTransitionedConfiguration(context, BuildConfigurationKey.create(finalBuildOptions));
    return runAfter;
  }

  /**
   * If a flag is considered to be out of scope, resetFlags does either of the following:
   *
   * <ul>
   *   <li>If the flag is not present in the baseline configuration, remove the flag from the {@link
   *       BuildOptions}.
   *   <li>If the flag is present in the baseline configuration, set the flag to the baseline value.
   *       <p>This is to ensure that we do not trigger an additional ST-<hash>, which defeats the
   *       <p>purpose of scoping.
   * </ul>
   *
   * This method returns the final {@link BuildOptions} after scoping is applied and the object only
   * has the {@link Scope.ScopeType} information for all starlark flags.
   */
  private static BuildOptions resetFlags(
      BuildOptionsScopeValue buildOptionsScopeValue,
      BuildOptions transitionedOptions,
      BuildOptions baselineConfiguration,
      @Nullable Label label) {
    Preconditions.checkNotNull(buildOptionsScopeValue);

    // If there are no scopes, short circuit.
    if (buildOptionsScopeValue.projectScopes().isEmpty()) {
      return transitionedOptions;
    }

    Preconditions.checkNotNull(baselineConfiguration);
    boolean flagsRemoved = false;
    boolean flagsResetToBaseline = false;
    BuildOptions.Builder optionsBuilder = transitionedOptions.toBuilder();
    for (Map.Entry<Label, Object> flagEntry : transitionedOptions.getStarlarkOptions().entrySet()) {
      Label flagLabel = flagEntry.getKey();
      // Only project-scoped flags are in the map, so a hit means the flag has to be scoped.
      Scope scope = buildOptionsScopeValue.projectScopes().get(flagLabel);
      if (scope != null) {
        Object flagValue = flagEntry.getValue();
        Object baselineValue = baselineConfiguration.getStarlarkOptions().get(flagLabel);
        if (flagValue != baselineValue && !isInScope(label, scope.getScopeDefinition())) {
          if (baselineValue == null) {
            optionsBuilder.removeStarlarkOption(flagLabel);
            flagsRemoved = true;
          } else {
            optionsBuilder.addStarlarkOption(flagLabel, baselineValue);
            flagsResetToBaseline = true;
          }
        }
      }
    }

    if (!flagsRemoved && !flagsResetToBaseline) {
      return transitionedOptions;
    }

    BuildOptions scopedBuildOptions = optionsBuilder.build();
    if (scopedBuildOptions.equals(baselineConfiguration)) {
      return baselineConfiguration;
    }

    return scopedBuildOptions;
  }

  private static boolean isInScope(
      @Nullable Label label, @Nullable Scope.ScopeDefinition scopeDefinition) {
    // A null scopeDefinition means the flag's package has no PROJECT.scl file. Treat the target
    // as not in scope so the flag resets to its baseline value.
    // Also, if the label is null, we are evaluating a configuration without a target, so we also
    // treat it
    // as out of scope.
    if (scopeDefinition == null || label == null) {
      return false;
    }
    for (String path : scopeDefinition.getOwnedCodePaths()) {
      if (label.getCanonicalForm().startsWith(path)) {
        return true;
      }
    }
    return false;
  }
}
