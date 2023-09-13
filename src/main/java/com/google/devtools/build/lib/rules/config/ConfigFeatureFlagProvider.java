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

package com.google.devtools.build.lib.rules.config;

import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.RequiredProviders;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.starlarkbuildapi.config.ConfigFeatureFlagProviderApi;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.StarlarkValue;

/** Provider for exporting value and valid value predicate of feature flags to consuming targets. */
// TODO(adonovan): rename this to *Info and its constructor to *Provider.
@Immutable
public class ConfigFeatureFlagProvider extends NativeInfo implements ConfigFeatureFlagProviderApi {

  /** Name used in Starlark for accessing ConfigFeatureFlagProvider. */
  static final String STARLARK_NAME = "FeatureFlagInfo";

  /**
   * Constructor and identifier for ConfigFeatureFlagProvider. This is the value of {@code
   * config_common.FeatureFlagInfo}.
   */
  static final BuiltinProvider<ConfigFeatureFlagProvider> STARLARK_CONSTRUCTOR = new Constructor();

  static final RequiredProviders REQUIRE_CONFIG_FEATURE_FLAG_PROVIDER =
      RequiredProviders.acceptAnyBuilder().addStarlarkSet(ImmutableSet.of(id())).build();

  private final String value;
  @Nullable private final String potentialError;
  private final Predicate<String> validityPredicate;

  private ConfigFeatureFlagProvider(
      String value, @Nullable String potentialError, Predicate<String> validityPredicate) {
    this.value = value;
    this.potentialError = potentialError;
    this.validityPredicate = validityPredicate;
  }

  @Override
  public BuiltinProvider<ConfigFeatureFlagProvider> getProvider() {
    return STARLARK_CONSTRUCTOR;
  }

  /** Creates a new ConfigFeatureFlagProvider with the given value and valid value predicate. */
  public static ConfigFeatureFlagProvider create(
      String value, @Nullable String potentialError, Predicate<String> isValidValue) {
    return new ConfigFeatureFlagProvider(value, potentialError, isValidValue);
  }

  /**
   * A constructor callable from Starlark for OutputGroupInfo: {@code
   * config_common.FeatureFlagInfo(value="...")}
   */
  @StarlarkBuiltin(name = "FeatureFlagInfo", documented = false)
  @Immutable
  private static final class Constructor extends BuiltinProvider<ConfigFeatureFlagProvider>
      implements StarlarkValue {

    Constructor() {
      super(STARLARK_NAME, ConfigFeatureFlagProvider.class);
    }

    @StarlarkMethod(
        name = "FeatureFlagInfo",
        documented = false,
        parameters = {@Param(name = "value", named = true)},
        selfCall = true)
    public ConfigFeatureFlagProvider selfcall(String value) {
      return create(value, null, Predicates.alwaysTrue());
    }

    @Override
    public void repr(Printer printer) {
      printer.append("<function FeatureFlagInfo>");
    }
  }

  public static StarlarkProviderIdentifier id() {
    return STARLARK_CONSTRUCTOR.id();
  }

  /** Retrieves and casts the provider from the given target. */
  public static ConfigFeatureFlagProvider fromTarget(TransitiveInfoCollection target) {
    return target.get(STARLARK_CONSTRUCTOR);
  }

  /**
   * Gets the current value of the flag in the flag's current configuration.
   *
   * <p>Throws EvalException when getError() is non-empty.
   */
  @Override
  @Nullable
  public String getFlagValue() {
    if (!Strings.isNullOrEmpty(potentialError)) {
      return null;
    }
    return value;
  }

  /** Gets the current value of the flag in the flag's current configuration. */
  @Override
  @Nullable
  public String getError() {
    return potentialError;
  }

  /** Returns whether this value is valid for this flag. */
  @Override
  public boolean isValidValue(String value) {
    return validityPredicate.apply(value);
  }

  // ConfigFeatureFlagProvider instances should all be unique, so we override the default
  // equals and hashCode from Info to ensure that. SCO's toString is fine, however.
  @Override
  public boolean equals(Object other) {
    return other == this;
  }

  @Override
  public int hashCode() {
    return System.identityHashCode(this);
  }
}
