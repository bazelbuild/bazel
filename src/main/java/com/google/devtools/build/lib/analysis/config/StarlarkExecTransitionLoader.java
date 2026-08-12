// Copyright 2023 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static java.util.Objects.requireNonNull;

import com.google.common.base.Splitter;
import com.google.common.base.Verify;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.transitions.SplitTransition;
import com.google.devtools.build.lib.analysis.starlark.StarlarkAttributeTransitionProvider;
import com.google.devtools.build.lib.analysis.starlark.StarlarkBuildSettingsDetailsValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.AttributeTransitionData;
import com.google.devtools.build.lib.rules.config.FeatureFlagValue;
import com.google.devtools.build.lib.skyframe.BzlLoadFailedException;
import com.google.devtools.build.lib.skyframe.BzlLoadValue;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import javax.annotation.Nullable;

/**
 * Utility class for loading a Starlark exec transition from source and making it available as an
 * {@link StarlarkAttributeTransitionProvider}.
 */
public final class StarlarkExecTransitionLoader {

  /** Thrown when the Starlark transition failed to load. */
  public static class StarlarkExecTransitionLoadingException extends Exception {
    public StarlarkExecTransitionLoadingException(String context, String ref, String message) {
      this(
          String.format(
              "Bad Starlark transition reference from %s: %s. %s.", context, ref, message));
    }

    public StarlarkExecTransitionLoadingException(String message) {
      super(message);
    }

    public StarlarkExecTransitionLoadingException(Throwable cause) {
      super(cause);
    }
  }

  /** Caller-provided logic for Skyframe-evaluating {@link BzlLoadValue.Key}s. */
  public interface BzlFileLoader {
    /**
     * Loads the given {@link BzlLoadValue.Key}. Returns null if not all Skyframe deps are ready.
     */
    @Nullable
    BzlLoadValue getValue(BzlLoadValue.Key key)
        throws BzlLoadFailedException, InterruptedException, StarlarkExecTransitionLoadingException;
  }

  /**
   * Returns the key for the scope info the Starlark exec transition needs to transition {@code
   * options}, or null if it needs none.
   *
   * <p>{@code BuildConfigurationFunction} uses this to resolve the value once per configuration and
   * store it on the {@code BuildConfigurationValue}, so that the exec transition can read it from
   * the configuration rather than resolving it once per configured target. Computing the key isn't
   * free: it scans the configuration's Starlark flags and interns a SkyKey, which during evaluation
   * routes through the graph's node map.
   */
  @Nullable
  public static StarlarkBuildSettingsDetailsValue.Key execScopeDetailsKey(BuildOptions options) {
    CoreOptions coreOptions = options.get(CoreOptions.class);
    if (coreOptions == null || coreOptions.getStarlarkExecConfig() == null) {
      // The exec transition is implemented by native logic, which doesn't consult scopes.
      return null;
    }
    // All Starlark build setting flags in the config, minus feature flags, which have no scopes.
    ImmutableSet<Label> starlarkFlags = flagsWithScopes(options.getStarlarkOptions());
    // Host flags declared by users in the blazerc/MODULE.bazel files with an alias pointing to the
    // Starlark definition. These determine exec propagation for flags scoped "exec:--".
    ImmutableSet<Label> hostFlags = coreOptions.getHostFlagAliases();
    if (starlarkFlags.isEmpty() && hostFlags.isEmpty()) {
      return null;
    }
    return StarlarkBuildSettingsDetailsValue.Key.create(starlarkFlags, hostFlags);
  }

  /**
   * Loads the Starlark transition that implements execution transition logic according to {@link
   * CoreOptions#starlarkExecConfig}.
   *
   * @param options the current configured target's {@link BuildOptions}. This is used to find the
   *     value for {@link CoreOptions#starlarkExecConfig}.
   * @param bzlFileLoader caller-provided logic for loading {@link BzlLoadValue.Key} skyvalues.
   * @param scopeDetails the scope info for {@code options}' Starlark flags, which callers holding
   *     the corresponding {@code BuildConfigurationValue} read off it with {@code
   *     starlarkExecScopeDetails()}. Null if the exec transition needs none, i.e. whenever {@link
   *     #execScopeDetailsKey} returns null for {@code options}.
   * @return null if Skyframe deps need loading. A filled {@link Optional} if this build implements
   *     the exec transition with a Starlark transition. An empty {@link Optional} if this build
   *     implements the exec transition with native logic.
   * @throws StarlarkExecTransitionLoadingException if the desired transition isn't a valid Starlark
   *     exec transition.
   */
  @Nullable
  public static Optional<StarlarkAttributeTransitionProvider> loadStarlarkExecTransition(
      @Nullable BuildOptions options,
      BzlFileLoader bzlFileLoader,
      @Nullable StarlarkBuildSettingsDetailsValue scopeDetails)
      throws StarlarkExecTransitionLoadingException, InterruptedException {
    if (options == null || options.equals(CommonOptions.EMPTY_OPTIONS)) {
      return Optional.empty();
    }
    String userRef =
        Verify.verifyNotNull(
            options.get(CoreOptions.class).getStarlarkExecConfig(),
            "Cannot apply the exec transition since no transition is defined for this build.");
    final String flagName = "--experimental_exec_config";
    TransitionReference parsedRef = TransitionReference.create(userRef, flagName);
    BzlLoadValue bzlValue;
    try {
      bzlValue =
          bzlFileLoader.getValue(
              Objects.equals(parsedRef.bzlFile().getRepository(), RepositoryName.BUILTINS)
                  ? BzlLoadValue.keyForBuiltins(parsedRef.bzlFile())
                  : BzlLoadValue.keyForBuild(parsedRef.bzlFile()));
    } catch (BzlLoadFailedException e) {
      throw new StarlarkExecTransitionLoadingException(flagName, userRef, e.getMessage());
    }
    if (bzlValue == null) {
      return null;
    }
    Object transition = bzlValue.getModule().getGlobal(parsedRef.starlarkSymbolName());
    if (transition == null) {
      throw new StarlarkExecTransitionLoadingException(
          flagName,
          userRef,
          String.format("%s not found in %s", parsedRef.starlarkSymbolName(), parsedRef.bzlFile()));
    } else if (!(transition instanceof StarlarkDefinedConfigTransition)) {
      throw new StarlarkExecTransitionLoadingException(
          flagName, userRef, parsedRef.starlarkSymbolName() + " is not a Starlark transition");
    }

    return Optional.of(
        new StarlarkExecTransitionProvider(
            (StarlarkDefinedConfigTransition) transition, scopeDetails));
  }

  /**
   * Returns the labels of the Starlark flags in {@code starlarkOptions} that can have scopes, i.e.
   * all of them except feature flags.
   *
   * <p>Avoids allocating a new set in the common case where no feature flags are set. Returning the
   * map's cached key set also lets the {@link StarlarkBuildSettingsDetailsValue.Key} interner
   * short-circuit on reference equality.
   */
  private static ImmutableSet<Label> flagsWithScopes(ImmutableMap<Label, Object> starlarkOptions) {
    for (Object value : starlarkOptions.values()) {
      if (value instanceof FeatureFlagValue) {
        return starlarkOptions.entrySet().stream()
            .filter(e -> !(e.getValue() instanceof FeatureFlagValue))
            .map(Map.Entry::getKey)
            .collect(toImmutableSet());
      }
    }
    return starlarkOptions.keySet();
  }

  /** A marker class to distinguish the exec transition from other starlark transitions. */
  static class StarlarkExecTransitionProvider extends StarlarkAttributeTransitionProvider {
    @Nullable private final StarlarkBuildSettingsDetailsValue scopeDetails;

    private final int hashCode;

    StarlarkExecTransitionProvider(
        StarlarkDefinedConfigTransition execTransition,
        @Nullable StarlarkBuildSettingsDetailsValue scopeDetails) {
      super(execTransition);
      this.scopeDetails = scopeDetails;
      this.hashCode = Objects.hash(super.hashCode(), scopeDetails);
    }

    @Override
    public SplitTransition create(AttributeTransitionData data) {
      return createWithScopeDetails(data, scopeDetails);
    }

    @Override
    public boolean allowImmutableFlagChanges() {
      // The exec transition must be allowed to change otherwise immutable flags.
      return true;
    }

    @Override
    public boolean isExecTransitionProvider() {
      return true;
    }

    @Override
    public int hashCode() {
      return hashCode;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof StarlarkExecTransitionProvider that)) {
        return false;
      }
      return super.equals(o) && Objects.equals(scopeDetails, that.scopeDetails);
    }
  }

  /**
   * Structured form of a Starlark transition reference.
   *
   * <p>In other words, structured form of <code>//pkg:def.bzl%transition_name</code>
   *
   * @param bzlFile The .bzl file where this transition is defined.
   * @param starlarkSymbolName The transition's Starlark symbol name.
   */
  record TransitionReference(Label bzlFile, String starlarkSymbolName) {
    TransitionReference {
      requireNonNull(bzlFile, "bzlFile");
      requireNonNull(starlarkSymbolName, "starlarkSymbolName");
    }

    /**
     * Returns a structured form of a user-specified Starlark transition reference.
     *
     * @throws StarlarkExecTransitionLoadingException on parsing errors.
     */
    static TransitionReference create(String userRef, String context)
        throws StarlarkExecTransitionLoadingException {
      List<String> splitval = Splitter.on('%').splitToList(userRef);
      if (splitval.size() < 2 || splitval.get(1).isEmpty()) {
        throw new StarlarkExecTransitionLoadingException(
            context, userRef, "Doesn't match expected form //pkg:file.bzl%%symbol");
      }
      try {
        return new TransitionReference(Label.parseCanonical(splitval.get(0)), splitval.get(1));
      } catch (LabelSyntaxException e) {
        throw new StarlarkExecTransitionLoadingException(
            context, userRef, String.format("Bad label %s: %s", splitval.get(0), e.getMessage()));
      }
    }
  }

  private StarlarkExecTransitionLoader() {}
}
