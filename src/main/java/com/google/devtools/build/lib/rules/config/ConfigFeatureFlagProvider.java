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

import com.google.auto.value.AutoValue;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.ClassObjectConstructor;
import com.google.devtools.build.lib.packages.NativeClassObjectConstructor;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.packages.SkylarkProviderIdentifier;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.util.Preconditions;

/** Provider for exporting value and valid value predicate of feature flags to consuming targets. */
@SkylarkModule(
  name = "FeatureFlagInfo",
  doc = "A provider used to access information about config_feature_flag rules."
)
@AutoValue
@Immutable
public abstract class ConfigFeatureFlagProvider extends SkylarkClassObject
    implements TransitiveInfoProvider {

  /** Name used in Skylark for accessing ConfigFeatureFlagProvider. */
  static final String SKYLARK_NAME = "FeatureFlagInfo";

  /** Skylark constructor and identifier for ConfigFeatureFlagProvider. */
  static final ClassObjectConstructor SKYLARK_CONSTRUCTOR =
      new NativeClassObjectConstructor(SKYLARK_NAME) {};

  /** Identifier used to retrieve this provider from rules which export it. */
  static final SkylarkProviderIdentifier SKYLARK_IDENTIFIER =
      SkylarkProviderIdentifier.forKey(SKYLARK_CONSTRUCTOR.getKey());

  ConfigFeatureFlagProvider() {
    super(SKYLARK_CONSTRUCTOR, ImmutableMap.<String, Object>of());
  }

  /** Creates a new ConfigFeatureFlagProvider with the given value and valid value predicate. */
  public static ConfigFeatureFlagProvider create(String value, Predicate<String> isValidValue) {
    return new AutoValue_ConfigFeatureFlagProvider(value, isValidValue);
  }

  /** Retrieves and casts the provider from the given target. */
  public static ConfigFeatureFlagProvider fromTarget(TransitiveInfoCollection target) {
    Object provider = target.get(SKYLARK_IDENTIFIER);
    if (provider == null) {
      return null;
    }
    Preconditions.checkState(provider instanceof ConfigFeatureFlagProvider);
    return (ConfigFeatureFlagProvider) provider;
  }

  /** Gets the current value of the flag in the flag's current configuration. */
  @SkylarkCallable(
    name = "value",
    structField = true,
    doc = "The value of the flag in the configuration used by the flag rule."
  )
  public abstract String getValue();

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
    return getIsValidValue().apply(value);
  }

  /** Gets the predicate which determines valid values for this flag. */
  abstract Predicate<String> getIsValidValue();
}
