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
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.packages.RequiredProviders;
import com.google.devtools.build.lib.packages.SkylarkProviderIdentifier;
import com.google.devtools.build.lib.skylarkbuildapi.config.ConfigFeatureFlagProviderApi;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import java.util.Map;

/** Provider for exporting value and valid value predicate of feature flags to consuming targets. */
@Immutable
public class ConfigFeatureFlagProvider extends NativeInfo implements ConfigFeatureFlagProviderApi {

  /** Name used in Skylark for accessing ConfigFeatureFlagProvider. */
  static final String SKYLARK_NAME = "FeatureFlagInfo";

  /** Skylark constructor and identifier for ConfigFeatureFlagProvider. */
  static final NativeProvider<ConfigFeatureFlagProvider> SKYLARK_CONSTRUCTOR = new Constructor();

  static final RequiredProviders REQUIRE_CONFIG_FEATURE_FLAG_PROVIDER =
      RequiredProviders.acceptAnyBuilder().addSkylarkSet(ImmutableSet.of(id())).build();

  private final String value;
  private final Predicate<String> validityPredicate;

  private ConfigFeatureFlagProvider(String value, Predicate<String> validityPredicate) {
    super(SKYLARK_CONSTRUCTOR);

    this.value = value;
    this.validityPredicate = validityPredicate;
  }

  /** Creates a new ConfigFeatureFlagProvider with the given value and valid value predicate. */
  public static ConfigFeatureFlagProvider create(String value, Predicate<String> isValidValue) {
    return new ConfigFeatureFlagProvider(value, isValidValue);
  }

  /** A constructor callable from Skylark for OutputGroupInfo. */
  private static class Constructor extends NativeProvider<ConfigFeatureFlagProvider> {

    private Constructor() {
      super(ConfigFeatureFlagProvider.class, SKYLARK_NAME);
    }

    @Override
    protected ConfigFeatureFlagProvider createInstanceFromSkylark(
        Object[] args, Environment env, Location loc) throws EvalException {

      @SuppressWarnings("unchecked")
      Map<String, Object> kwargs = (Map<String, Object>) args[0];

      if (!kwargs.containsKey("value") || !(kwargs.get("value") instanceof String)) {
        throw new EvalException(loc, "FeatureFlagInfo requires 'value' to be set to a string");
      }
      return create((String) kwargs.get("value"), Predicates.alwaysTrue());
    }
}

  public static SkylarkProviderIdentifier id() {
    return SKYLARK_CONSTRUCTOR.id();
  }

  /** Retrieves and casts the provider from the given target. */
  public static ConfigFeatureFlagProvider fromTarget(TransitiveInfoCollection target) {
    return target.get(SKYLARK_CONSTRUCTOR);
  }

  /** Gets the current value of the flag in the flag's current configuration. */
  @Override
  public String getFlagValue() {
    return value;
  }

  /** Returns whether this value is valid for this flag. */
  @Override
  public boolean isValidValue(String value) {
    return validityPredicate.apply(value);
  }

  // ConfigFeatureFlagProvider instances should all be unique, so we override the default
  // equals and hashCode from InfoInterface to ensure that. SCO's toString is fine, however.
  @Override
  public boolean equals(Object other) {
    return other == this;
  }

  @Override
  public int hashCode() {
    return System.identityHashCode(this);
  }
}
