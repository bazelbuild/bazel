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

import static java.util.Objects.requireNonNull;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.Scope;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyKey.SkyKeyInterner;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.errorprone.annotations.CheckReturnValue;
import java.util.Map;
import java.util.Set;

/**
 * This contains information about a list of given Starlark build options, specifically their
 * defaults and the (final) actual values of alias {@link Label}.
 *
 * <p>For memory-efficiency reasons, aliasToActual contains only aliases in keys. Other attributes
 * contain only actual build setting as keys.
 *
 * <p>Potentially aliased targets can be unaliased with aliasToActual().getWithDefault(raw, raw);
 *
 * @param buildSettingToDefault Map from each build option to its default value. Does not include
 *     aliases.
 * @param buildSettingToType Map from each build option to its type information. Does not include
 *     aliases.
 * @param buildSettingIsAllowsMultiple If build option is in this set, is an allows_multiple option.
 *     Does not include aliases.
 * @param aliasToActual Map from an alias Label to actual Label it points to.
 * @param customExecScopeValues Map from a build setting Label to the custom exec scope value for
 *     that setting. This contains [--foo, default_foo, --host_foo, default_host_foo,
 *     scope_type_foo, scope_type_host_foo]
 * @param buildSettingToScopeType Map from each build option to its scope type. Does not include
 *     aliases.
 * @param buildSettingToOnLeaveScopeValue Map from each build option to its on_leave_scope value, if
 *     explicitly set. Does not include aliases.
 */
@CheckReturnValue
@Immutable
@ThreadSafe
@AutoCodec
public record StarlarkBuildSettingsDetailsValue(
    ImmutableMap<Label, Object> buildSettingToDefault,
    ImmutableMap<Label, Type<?>> buildSettingToType,
    ImmutableSet<Label> buildSettingIsAllowsMultiple,
    ImmutableMap<Label, Label> aliasToActual,
    ImmutableMap<Label, CustomExecScopeValue> customExecScopeValues,
    ImmutableMap<Label, Scope.ScopeType> buildSettingToScopeType,
    ImmutableMap<Label, Object> buildSettingToOnLeaveScopeValue)
    implements SkyValue {
  public StarlarkBuildSettingsDetailsValue {
    requireNonNull(buildSettingToDefault, "buildSettingToDefault");
    requireNonNull(buildSettingToType, "buildSettingToType");
    requireNonNull(buildSettingIsAllowsMultiple, "buildSettingIsAllowsMultiple");
    requireNonNull(aliasToActual, "aliasToActual");
    requireNonNull(customExecScopeValues, "customExecScopeValues");
    requireNonNull(buildSettingToScopeType, "buildSettingToScopeType");
    requireNonNull(buildSettingToOnLeaveScopeValue, "buildSettingToOnLeaveScopeValue");
  }

  /**
   * Create a single StarlarkBuildSettingsDetailsValue that can be quickly returned for transitions
   * that use no Starlark build settings
   */
  public static final StarlarkBuildSettingsDetailsValue EMPTY =
      create(
          ImmutableMap.of(),
          ImmutableMap.of(),
          ImmutableSet.of(),
          ImmutableMap.of(),
          ImmutableMap.of(),
          ImmutableMap.of(),
          ImmutableMap.of());

  public static StarlarkBuildSettingsDetailsValue create(
      Map<Label, Object> buildSettingDefaults,
      Map<Label, Type<?>> buildSettingToType,
      Set<Label> buildSettingIsAllowsMultiple,
      Map<Label, Label> aliasToActual,
      Map<Label, CustomExecScopeValue> customExecScopeValues,
      Map<Label, Scope.ScopeType> buildSettingToScopeType,
      Map<Label, Object> buildSettingToOnLeaveScopeValue) {
    return new StarlarkBuildSettingsDetailsValue(
        ImmutableMap.copyOf(buildSettingDefaults),
        ImmutableMap.copyOf(buildSettingToType),
        ImmutableSet.copyOf(buildSettingIsAllowsMultiple),
        ImmutableMap.copyOf(aliasToActual),
        ImmutableMap.copyOf(customExecScopeValues),
        ImmutableMap.copyOf(buildSettingToScopeType),
        ImmutableMap.copyOf(buildSettingToOnLeaveScopeValue));
  }

  public static Key key(Set<Label> buildSettings, Set<Label> hostFlags) {
    return Key.create(ImmutableSet.copyOf(buildSettings), ImmutableSet.copyOf(hostFlags));
  }

  /**
   * Represents a custom exec scope value for a Starlark build setting.
   *
   * @param flag the label of the build setting, e.g. //:foo
   * @param flagDefault the default value of the build setting
   * @param hostFlag the label of the host flag, e.g. //:host_foo
   * @param hostFlagDefault the default value of the host flag, which is the value that will be used
   *     for the build setting in the exec configuration.
   * @param flagScopeType the scope type of the build setting, e.g. "exec:--host_foo"
   * @param hostFlagScopeType the scope type of the host flag, e.g. "default" or "target"
   */
  @AutoCodec
  @Immutable
  @ThreadSafe
  public record CustomExecScopeValue(
      Label flag,
      Object flagDefault,
      Label hostFlag,
      Object hostFlagDefault,
      String flagScopeType,
      String hostFlagScopeType) {}

  /**
   * {@link SkyKey} implementation used for {@link StarlarkBuildSettingsDetailsValue}.
   *
   * <p>Instances are interned: the exec transition requests this key once per configured target and
   * Skyframe retains the requesting key instance in each node's dependency list. Without interning,
   * every configured target would retain its own copy of the (potentially large) label sets.
   */
  @CheckReturnValue
  @Immutable
  @ThreadSafe
  @AutoCodec
  public static final class Key implements SkyKey {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    private final ImmutableSet<Label> buildSettings;
    private final ImmutableSet<Label> hostFlags;
    private final int hashCode;

    private Key(ImmutableSet<Label> buildSettings, ImmutableSet<Label> hostFlags) {
      this.buildSettings = requireNonNull(buildSettings, "buildSettings");
      this.hostFlags = requireNonNull(hostFlags, "hostFlags");
      this.hashCode = 31 * buildSettings.hashCode() + hostFlags.hashCode();
    }

    public static Key create(ImmutableSet<Label> buildSettings, ImmutableSet<Label> hostFlags) {
      return interner.intern(new Key(buildSettings, hostFlags));
    }

    @VisibleForSerialization
    @AutoCodec.Interner
    static Key intern(Key key) {
      return interner.intern(key);
    }

    public ImmutableSet<Label> buildSettings() {
      return buildSettings;
    }

    public ImmutableSet<Label> hostFlags() {
      return hostFlags;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.STARLARK_BUILD_SETTINGS_DETAILS;
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == this) {
        return true;
      }
      return obj instanceof Key other
          && hashCode == other.hashCode
          && buildSettings.equals(other.buildSettings)
          && hostFlags.equals(other.hostFlags);
    }

    @Override
    public int hashCode() {
      return hashCode;
    }

    @Override
    public String toString() {
      return "Key[buildSettings=%s, hostFlags=%s]".formatted(buildSettings, hostFlags);
    }
  }
}
