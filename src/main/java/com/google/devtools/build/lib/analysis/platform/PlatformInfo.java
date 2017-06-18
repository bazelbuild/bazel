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

package com.google.devtools.build.lib.analysis.platform;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.ClassObjectConstructor;
import com.google.devtools.build.lib.packages.NativeClassObjectConstructor;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.packages.SkylarkProviderIdentifier;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkType;
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
  public static final String SKYLARK_NAME = "PlatformInfo";

  private static final FunctionSignature.WithValues<Object, SkylarkType> SIGNATURE =
      FunctionSignature.WithValues.create(
          FunctionSignature.of(
              /*numMandatoryPositionals=*/ 2,
              /*numOptionalPositionals=*/ 0,
              /*numMandatoryNamedOnly*/ 0,
              /*starArg=*/ false,
              /*kwArg=*/ false,
              /*names=*/ "label",
              "constraint_values"),
          /*defaultValues=*/ null,
          /*types=*/ ImmutableList.<SkylarkType>of(
              SkylarkType.of(Label.class),
              SkylarkType.Combination.of(
                  SkylarkType.LIST, SkylarkType.of(ConstraintValueInfo.class))));

  /** Skylark constructor and identifier for this provider. */
  public static final ClassObjectConstructor SKYLARK_CONSTRUCTOR =
      new NativeClassObjectConstructor(SKYLARK_NAME, SIGNATURE) {
        @Override
        protected PlatformInfo createInstanceFromSkylark(Object[] args, Location loc)
            throws EvalException {
          // Based on SIGNATURE above, the args are label, constraint_values.

          Label label = (Label) args[0];
          SkylarkList<ConstraintValueInfo> constraintValues =
              (SkylarkList<ConstraintValueInfo>) args[1];
          try {
            return builder()
                .setLabel(label)
                .addConstraints(constraintValues)
                .setLocation(loc)
                .build();
          } catch (DuplicateConstraintException dce) {
            throw new EvalException(
                loc, String.format("Cannot create PlatformInfo: %s", dce.getMessage()));
          }
        }
      };

  /** Identifier used to retrieve this provider from rules which export it. */
  public static final SkylarkProviderIdentifier SKYLARK_IDENTIFIER =
      SkylarkProviderIdentifier.forKey(SKYLARK_CONSTRUCTOR.getKey());

  private final Label label;
  private final ImmutableList<ConstraintValueInfo> constraints;
  private final ImmutableMap<String, String> remoteExecutionProperties;

  private PlatformInfo(
      Label label,
      ImmutableList<ConstraintValueInfo> constraints,
      ImmutableMap<String, String> remoteExecutionProperties,
      Location location) {
    super(
        SKYLARK_CONSTRUCTOR,
        ImmutableMap.<String, Object>of(
            "label", label,
            "constraints", constraints),
        location);

    this.label = label;
    this.constraints = constraints;
    this.remoteExecutionProperties = remoteExecutionProperties;
  }

  @SkylarkCallable(
    name = "label",
    doc = "The label of the target that created this platform.",
    structField = true
  )
  public Label label() {
    return label;
  }

  @SkylarkCallable(
    name = "constraints",
    doc =
        "The <a href=\"ConstraintValueInfo.html\">ConstraintValueInfo</a> instances that define "
            + "this platform.",
    structField = true
  )
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

  /** Returns a new {@link Builder} for creating a fresh {@link PlatformInfo} instance. */
  public static Builder builder() {
    return new Builder();
  }

  /** Builder class to facilitate creating valid {@link PlatformInfo} instances. */
  public static class Builder {
    private Label label;
    private final List<ConstraintValueInfo> constraints = new ArrayList<>();
    private final Map<String, String> remoteExecutionProperties = new HashMap<>();
    private Location location = Location.BUILTIN;

    /**
     * Sets the {@link Label} for this {@link PlatformInfo}.
     *
     * @param label the label identifying this platform
     * @return the {@link Builder} instance for method chaining
     */
    public Builder setLabel(Label label) {
      this.label = label;
      return this;
    }

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
     * Sets the {@link Location} where this {@link PlatformInfo} was created.
     *
     * @param location the location where the instance was created
     * @return the {@link Builder} instance for method chaining
     */
    public Builder setLocation(Location location) {
      this.location = location;
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
      return new PlatformInfo(
          label, validatedConstraints, ImmutableMap.copyOf(remoteExecutionProperties), location);
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
