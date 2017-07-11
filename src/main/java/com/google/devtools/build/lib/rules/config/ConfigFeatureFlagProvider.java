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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.NativeClassObjectConstructor;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;

/** Provider for exporting value and valid value predicate of feature flags to consuming targets. */
@SkylarkModule(
  name = "FeatureFlagInfo",
  doc = "A provider used to access information about config_feature_flag rules."
)
@Immutable
public class ConfigFeatureFlagProvider extends SkylarkClassObject {

  /** Name used in Skylark for accessing ConfigFeatureFlagProvider. */
  static final String SKYLARK_NAME = "FeatureFlagInfo";

  /** Skylark constructor and identifier for ConfigFeatureFlagProvider. */
  public static final NativeClassObjectConstructor<ConfigFeatureFlagProvider> SKYLARK_CONSTRUCTOR =
      new NativeClassObjectConstructor<ConfigFeatureFlagProvider>(
          ConfigFeatureFlagProvider.class, SKYLARK_NAME) {};

  private final String value;
  private final Predicate<String> validityPredicate;

  private ConfigFeatureFlagProvider(String value, Predicate<String> validityPredicate) {
    super(SKYLARK_CONSTRUCTOR, ImmutableMap.<String, Object>of("value", value));

    this.value = value;
    this.validityPredicate = validityPredicate;
  }

  /** Creates a new ConfigFeatureFlagProvider with the given value and valid value predicate. */
  public static ConfigFeatureFlagProvider create(String value, Predicate<String> isValidValue) {
    return new ConfigFeatureFlagProvider(value, isValidValue);
  }

  /** Retrieves and casts the provider from the given target. */
  public static ConfigFeatureFlagProvider fromTarget(TransitiveInfoCollection target) {
    return target.get(SKYLARK_CONSTRUCTOR);
  }

  /** Gets the current value of the flag in the flag's current configuration. */
  public String getValue() {
    return value;
  }

  /** Returns whether this value is valid for this flag. */
  @SkylarkCallable(
    name = "is_valid_value",
    doc = "The value of the flag in the configuration used by the flag rule.",
    parameters = {
      @Param(
        name = "value",
        type = String.class,
        doc = "String, the value to check for validity for this flag."
      ),
    }
  )
  public boolean isValidValue(String value) {
    return validityPredicate.apply(value);
  }

  // ConfigFeatureFlagProvider instances should all be unique, so we override the default
  // equals and hashCode from SkylarkClassObject to ensure that. SCO's toString is fine, however.
  @Override
  public boolean equals(Object other) {
    return other == this;
  }

  @Override
  public int hashCode() {
    return System.identityHashCode(this);
  }
}
