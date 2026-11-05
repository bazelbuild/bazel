// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.starlark;

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static java.util.Objects.requireNonNull;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.Scope;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.config.FeatureFlagValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyKey.SkyKeyInterner;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.errorprone.annotations.CheckReturnValue;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * This contains information about a list of given Starlark build options, specifically their
 * defaults, scopes and the (final) actual values of alias {@link Label}.
 *
 * <p>For memory-efficiency reasons, aliasToActual contains only aliases in keys. Other attributes
 * contain only actual build setting as keys.
 *
 * <p>Potentially aliased targets can be unaliased with aliasToActual().getWithDefault(raw, raw);
 *
 * @param buildSettings The (possibly aliased) build settings in the {@link Key} this value was
 *     computed for. Used by {@link #covers}.
 * @param buildSettingToDefault Map from each build option to its default value. Does not include
 *     aliases.
 * @param buildSettingToType Map from each build option to its type information. Does not include
 *     aliases.
 * @param buildSettingIsAllowsMultiple If build option is in this set, is an allows_multiple option.
 *     Does not include aliases.
 * @param aliasToActual Map from an alias Label to actual Label it points to.
 * @param customExecScopeValues Map from a build setting Label to its custom exec scope value, which
 *     includes the referenced flag's default. Host flag aliases have entries even if neither flag
 *     is set.
 * @param buildSettingToScopeType Map from each build option to its scope type. Does not include
 *     aliases.
 * @param buildSettingToOnLeaveScopeValue Map from each build option to its on_leave_scope value, if
 *     explicitly set. Does not include aliases.
 * @param projectScopes Map from each build option with {@link Scope.ScopeType#PROJECT} scope to its
 *     {@link Scope}, including the {@link Scope.ScopeDefinition} from the PROJECT.scl file
 *     governing the option's package, if any. Does not include aliases.
 */
@CheckReturnValue
@Immutable
@ThreadSafe
@AutoCodec
public record StarlarkBuildSettingsDetailsValue(
    ImmutableSet<Label> buildSettings,
    ImmutableMap<Label, Object> buildSettingToDefault,
    ImmutableMap<Label, Type<?>> buildSettingToType,
    ImmutableSet<Label> buildSettingIsAllowsMultiple,
    ImmutableMap<Label, Label> aliasToActual,
    ImmutableMap<Label, CustomExecScopeValue> customExecScopeValues,
    ImmutableMap<Label, Scope.ScopeType> buildSettingToScopeType,
    ImmutableMap<Label, Object> buildSettingToOnLeaveScopeValue,
    ImmutableMap<Label, Scope> projectScopes)
    implements SkyValue {
  public StarlarkBuildSettingsDetailsValue {
    requireNonNull(buildSettings, "buildSettings");
    requireNonNull(buildSettingToDefault, "buildSettingToDefault");
    requireNonNull(buildSettingToType, "buildSettingToType");
    requireNonNull(buildSettingIsAllowsMultiple, "buildSettingIsAllowsMultiple");
    requireNonNull(aliasToActual, "aliasToActual");
    requireNonNull(customExecScopeValues, "customExecScopeValues");
    requireNonNull(buildSettingToScopeType, "buildSettingToScopeType");
    requireNonNull(buildSettingToOnLeaveScopeValue, "buildSettingToOnLeaveScopeValue");
    requireNonNull(projectScopes, "projectScopes");
  }

  /**
   * Create a single StarlarkBuildSettingsDetailsValue that can be quickly returned for transitions
   * that use no Starlark build settings
   */
  public static final StarlarkBuildSettingsDetailsValue EMPTY =
      create(
          ImmutableSet.of(),
          ImmutableMap.of(),
          ImmutableMap.of(),
          ImmutableSet.of(),
          ImmutableMap.of(),
          ImmutableMap.of(),
          ImmutableMap.of(),
          ImmutableMap.of(),
          ImmutableMap.of());

  public static StarlarkBuildSettingsDetailsValue create(
      Set<Label> buildSettings,
      Map<Label, Object> buildSettingDefaults,
      Map<Label, Type<?>> buildSettingToType,
      Set<Label> buildSettingIsAllowsMultiple,
      Map<Label, Label> aliasToActual,
      Map<Label, CustomExecScopeValue> customExecScopeValues,
      Map<Label, Scope.ScopeType> buildSettingToScopeType,
      Map<Label, Object> buildSettingToOnLeaveScopeValue,
      Map<Label, Scope> projectScopes) {
    return new StarlarkBuildSettingsDetailsValue(
        ImmutableSet.copyOf(buildSettings),
        ImmutableMap.copyOf(buildSettingDefaults),
        ImmutableMap.copyOf(buildSettingToType),
        ImmutableSet.copyOf(buildSettingIsAllowsMultiple),
        ImmutableMap.copyOf(aliasToActual),
        ImmutableMap.copyOf(customExecScopeValues),
        ImmutableMap.copyOf(buildSettingToScopeType),
        ImmutableMap.copyOf(buildSettingToOnLeaveScopeValue),
        ImmutableMap.copyOf(projectScopes));
  }

  public static Key key(Set<Label> buildSettings, Set<Label> hostFlags) {
    return Key.create(ImmutableSet.copyOf(buildSettings), ImmutableSet.copyOf(hostFlags));
  }

  /**
   * Returns the key for the details of a configuration's Starlark build settings and host flag
   * aliases, or null if it has neither.
   *
   * @param starlarkOptions the configuration's Starlark options
   * @param hostFlags the Starlark flags that {@code --flag_alias} maps to a {@code host_}-prefixed
   *     name, see {@code CoreOptions#getHostFlagAliases}. Included since a flag scoped {@code
   *     exec:--host_foo} takes the value of {@code --host_foo} in exec configurations, even if
   *     neither is set.
   */
  @Nullable
  public static Key keyForStarlarkOptions(
      ImmutableMap<Label, Object> starlarkOptions, ImmutableSet<Label> hostFlags) {
    ImmutableSet<Label> buildSettings = scopedBuildSettings(starlarkOptions);
    if (buildSettings.isEmpty() && hostFlags.isEmpty()) {
      return null;
    }
    return Key.create(buildSettings, hostFlags);
  }

  /**
   * Returns the labels of the Starlark options in {@code starlarkOptions} that aren't feature
   * flags, which are the ones with scopes.
   */
  public static ImmutableSet<Label> scopedBuildSettings(
      ImmutableMap<Label, Object> starlarkOptions) {
    for (Object value : starlarkOptions.values()) {
      if (value instanceof FeatureFlagValue) {
        return starlarkOptions.entrySet().stream()
            .filter(e -> !(e.getValue() instanceof FeatureFlagValue))
            .map(Map.Entry::getKey)
            .collect(toImmutableSet());
      }
    }
    // The map's own key set avoids an allocation and lets covers and the Key interner compare by
    // reference.
    return starlarkOptions.keySet();
  }

  /**
   * Returns whether this value has details for all of {@code buildSettings}, in which case it can
   * be used instead of a value computed for exactly those settings.
   */
  public boolean covers(Set<Label> buildSettings) {
    // This runs on every transitioned dependency edge, so first check for the very set this value
    // was computed for.
    return this.buildSettings == buildSettings
        || buildSettings.isEmpty()
        || (this.buildSettings.size() >= buildSettings.size()
            && this.buildSettings.containsAll(buildSettings));
  }

  /** Returns whether any build setting in this value has {@link Scope.ScopeType#PROJECT} scope. */
  public boolean hasProjectScopedBuildSettings() {
    return !projectScopes.isEmpty();
  }

  /**
   * Returns the {@link Scope} of the possibly aliased {@code buildSetting} if it has {@link
   * Scope.ScopeType#PROJECT} scope, or null otherwise.
   */
  @Nullable
  public Scope projectScopeOf(Label buildSetting) {
    return projectScopes.get(aliasToActual.getOrDefault(buildSetting, buildSetting));
  }

  /**
   * Returns the scope type of the build setting with the actual (non-alias) label {@code
   * buildSetting}, or {@link Scope.ScopeType#DEFAULT} if this value has no details for it.
   */
  public String scopeTypeOf(Label buildSetting) {
    Scope.ScopeType scopeType = buildSettingToScopeType.get(buildSetting);
    return scopeType != null ? scopeType.scopeType() : Scope.ScopeType.DEFAULT;
  }

  /**
   * Represents a custom exec scope value for a Starlark build setting.
   *
   * @param flag the label of the build setting, e.g. //:foo
   * @param flagDefault the default value of the build setting
   * @param hostFlag the label of the host flag, e.g. //:host_foo
   * @param hostFlagDefault the default value of the host flag, which is the value that will be used
   *     for the build setting in the exec configuration.
   */
  @AutoCodec
  @Immutable
  @ThreadSafe
  public record CustomExecScopeValue(
      Label flag, Object flagDefault, Label hostFlag, Object hostFlagDefault) {}

  /** {@link SkyKey} implementation used for {@link StarlarkBuildSettingsDetailsValue}. */
  @CheckReturnValue
  @Immutable
  @ThreadSafe
  @AutoCodec
  public record Key(ImmutableSet<Label> buildSettings, ImmutableSet<Label> hostFlags)
      implements SkyKey {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    public Key {
      requireNonNull(buildSettings, "buildSettings");
      requireNonNull(hostFlags, "hostFlags");
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.STARLARK_BUILD_SETTINGS_DETAILS;
    }

    @AutoCodec.Instantiator
    public static Key create(ImmutableSet<Label> buildSettings, ImmutableSet<Label> hostFlags) {
      return interner.intern(new Key(buildSettings, hostFlags));
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }
}
