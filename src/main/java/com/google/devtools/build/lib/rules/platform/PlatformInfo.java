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

package com.google.devtools.build.lib.rules.platform;

import com.google.common.base.Function;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.ClassObjectConstructor;
import com.google.devtools.build.lib.packages.NativeClassObjectConstructor;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.packages.SkylarkProviderIdentifier;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Provider for a platform, which is a group of constraints and values. */
@SkylarkModule(
  name = "PlatformInfo",
  doc = "Provides access to data about a specific platform.",
  category = SkylarkModuleCategory.PROVIDER
)
@Immutable
public class PlatformInfo extends SkylarkClassObject {

  /** Name used in Skylark for accessing this provider. */
  static final String SKYLARK_NAME = "PlatformInfo";

  /** Skylark constructor and identifier for this provider. */
  static final ClassObjectConstructor SKYLARK_CONSTRUCTOR =
      new NativeClassObjectConstructor(SKYLARK_NAME) {};

  /** Identifier used to retrieve this provider from rules which export it. */
  public static final SkylarkProviderIdentifier SKYLARK_IDENTIFIER =
      SkylarkProviderIdentifier.forKey(SKYLARK_CONSTRUCTOR.getKey());

  private final ImmutableList<ConstraintValueInfo> constraints;
  private final ImmutableMap<String, String> remoteExecutionProperties;

  private PlatformInfo(
      ImmutableList<ConstraintValueInfo> constraints,
      ImmutableMap<String, String> remoteExecutionProperties) {
    super(SKYLARK_CONSTRUCTOR, ImmutableMap.<String, Object>of("constraints", constraints));

    this.constraints = constraints;
    this.remoteExecutionProperties = remoteExecutionProperties;
  }

  public ImmutableList<ConstraintValueInfo> constraints() {
    return constraints;
  }

  @SkylarkCallable(
    name = "remoteExecutionProperties",
    doc = "Properties that are available for the use of remote execution.",
    structField = true
  )
  public ImmutableMap<String, String> remoteExecutionProperties() {
    return remoteExecutionProperties;
  }

  /** Retrieves and casts the provider from the given target. */
  public static PlatformInfo fromTarget(TransitiveInfoCollection target) {
    Object provider = target.get(SKYLARK_IDENTIFIER);
    if (provider == null) {
      return null;
    }
    Preconditions.checkState(provider instanceof PlatformInfo);
    return (PlatformInfo) provider;
  }

  /** Retrieves and casts the providers from the given targets. */
  public static Iterable<PlatformInfo> fromTargets(
      Iterable<? extends TransitiveInfoCollection> targets) {
    return Iterables.transform(
        targets,
        new Function<TransitiveInfoCollection, PlatformInfo>() {
          @Override
          public PlatformInfo apply(TransitiveInfoCollection target) {
            return fromTarget(target);
          }
        });
  }

  /** Returns a new {@link Builder} for creating a fresh {@link PlatformInfo} instance. */
  public static Builder builder() {
    return new Builder();
  }

  /** Builder class to facilitate creating valid {@link PlatformInfo} instances. */
  public static class Builder {
    private final List<ConstraintValueInfo> constraints = new ArrayList<>();
    private final Map<String, String> remoteExecutionProperties = new HashMap<>();

    /**
     * Adds the given constraint value to the constraints that define this {@link PlatformInfo}.
     *
     * @param constraint the constraint to add
     * @return the {@link Builder} instance for method chaining
     */
    public Builder addConstraint(ConstraintValueInfo constraint) {
      this.constraints.add(constraint);
      return this;
    }

    /**
     * Adds the given constraint values to the constraints that define this {@link PlatformInfo}.
     *
     * @param constraints the constraints to add
     * @return the {@link Builder} instance for method chaining
     */
    public Builder addConstraints(Iterable<ConstraintValueInfo> constraints) {
      for (ConstraintValueInfo constraint : constraints) {
        this.addConstraint(constraint);
      }

      return this;
    }

    /**
     * Adds the given key/value pair to the data being sent to a potential remote executor. If the
     * key already exists in the map, the previous value will be overwritten.
     *
     * @param key the key to be used
     * @param value the value to be used
     * @return the {@link Builder} instance for method chaining
     */
    public Builder addRemoteExecutionProperty(String key, String value) {
      this.remoteExecutionProperties.put(key, value);
      return this;
    }

    /**
     * Adds the given properties to the data being sent to a potential remote executor. If any key
     * already exists in the map, the previous value will be overwritten.
     *
     * @param properties the properties to be added
     * @return the {@link Builder} instance for method chaining
     */
    public Builder addRemoteExecutionProperties(Map<String, String> properties) {
      for (Map.Entry<String, String> entry : properties.entrySet()) {
        this.addRemoteExecutionProperty(entry.getKey(), entry.getValue());
      }
      return this;
    }

    /**
     * Returns the new {@link PlatformInfo} instance.
     *
     * @throws DuplicateConstraintException if more than one constraint value exists for the same
     *     constraint setting
     */
    public PlatformInfo build() throws DuplicateConstraintException {
      ImmutableList<ConstraintValueInfo> validatedConstraints = validateConstraints(constraints);
      return new PlatformInfo(validatedConstraints, ImmutableMap.copyOf(remoteExecutionProperties));
    }

    private ImmutableList<ConstraintValueInfo> validateConstraints(
        Iterable<ConstraintValueInfo> constraintValues) throws DuplicateConstraintException {
      Multimap<ConstraintSettingInfo, ConstraintValueInfo> constraints = ArrayListMultimap.create();

      for (ConstraintValueInfo constraintValue : constraintValues) {
        constraints.put(constraintValue.constraint(), constraintValue);
      }

      // Are there any settings with more than one value?
      for (ConstraintSettingInfo constraintSetting : constraints.keySet()) {
        if (constraints.get(constraintSetting).size() > 1) {
          // Only reports the first case of this error.
          throw new DuplicateConstraintException(
              constraintSetting, constraints.get(constraintSetting));
        }
      }

      return ImmutableList.copyOf(constraints.values());
    }
  }

  /**
   * Exception class used when more than one {@link ConstraintValueInfo} for the same {@link
   * ConstraintSettingInfo} is added to a {@link Builder}.
   */
  public static class DuplicateConstraintException extends Exception {
    private final ImmutableSet<ConstraintValueInfo> duplicateConstraints;

    public DuplicateConstraintException(
        ConstraintSettingInfo constraintSetting,
        Iterable<ConstraintValueInfo> duplicateConstraintValues) {
      super(formatError(constraintSetting, duplicateConstraintValues));
      this.duplicateConstraints = ImmutableSet.copyOf(duplicateConstraintValues);
    }

    public ImmutableSet<ConstraintValueInfo> duplicateConstraints() {
      return duplicateConstraints;
    }

    private static String formatError(
        ConstraintSettingInfo constraintSetting,
        Iterable<ConstraintValueInfo> duplicateConstraints) {
      StringBuilder constraintValuesDescription = new StringBuilder();
      for (ConstraintValueInfo constraintValue : duplicateConstraints) {
        if (constraintValuesDescription.length() > 0) {
          constraintValuesDescription.append(", ");
        }
        constraintValuesDescription.append(constraintValue.label());
      }
      return String.format(
          "Duplicate constraint_values for constraint_setting %s: %s",
          constraintSetting.label(), constraintValuesDescription.toString());
    }
  }
}
