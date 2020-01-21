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

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.platform.ConstraintCollection.DuplicateConstraintException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkbuildapi.platform.PlatformInfoApi;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.StringUtilities;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import javax.annotation.Nullable;

/** Provider for a platform, which is a group of constraints and values. */
@Immutable
@AutoCodec
public class PlatformInfo extends NativeInfo
    implements PlatformInfoApi<ConstraintSettingInfo, ConstraintValueInfo> {

  /**
   * The literal key that will be used to copy the {@link #remoteExecutionProperties} from the
   * parent {@link PlatformInfo} into a new {@link PlatformInfo}'s {@link
   * #remoteExecutionProperties}.
   */
  public static final String PARENT_REMOTE_EXECUTION_KEY = "{PARENT_REMOTE_EXECUTION_PROPERTIES}";

  /** Name used in Skylark for accessing this provider. */
  public static final String SKYLARK_NAME = "PlatformInfo";

  /** Provider singleton constant. */
  public static final BuiltinProvider<PlatformInfo> PROVIDER = new Provider();

  /** Provider for {@link ToolchainInfo} objects. */
  private static class Provider extends BuiltinProvider<PlatformInfo>
      implements PlatformInfoApi.Provider<
          ConstraintSettingInfo, ConstraintValueInfo, PlatformInfo> {
    private Provider() {
      super(SKYLARK_NAME, PlatformInfo.class);
    }

    @Override
    public PlatformInfo platformInfo(
        Label label,
        Object parentUnchecked,
        Sequence<?> constraintValuesUnchecked,
        Object execPropertiesUnchecked,
        Location location)
        throws EvalException {
      PlatformInfo.Builder builder = PlatformInfo.builder();
      builder.setLabel(label);
      if (parentUnchecked != Starlark.NONE) {
        builder.setParent((PlatformInfo) parentUnchecked);
      }
      if (!constraintValuesUnchecked.isEmpty()) {
        builder.addConstraints(
            constraintValuesUnchecked.getContents(ConstraintValueInfo.class, "constraint_values"));
      }
      if (execPropertiesUnchecked != null) {
        Map<String, String> execProperties =
            Dict.castSkylarkDictOrNoneToDict(
                execPropertiesUnchecked, String.class, String.class, "exec_properties");
        builder.setExecProperties(ImmutableMap.copyOf(execProperties));
      }
      builder.setLocation(location);

      try {
        return builder.build();
      } catch (DuplicateConstraintException | ExecPropertiesException e) {
        throw new EvalException(location, e);
      }
    }
  }

  private final Label label;
  private final ConstraintCollection constraints;
  private final String remoteExecutionProperties;
  /** execProperties will deprecate and replace remoteExecutionProperties */
  private final ImmutableMap<String, String> execProperties;

  @AutoCodec.Instantiator
  @VisibleForSerialization
  PlatformInfo(
      Label label,
      ConstraintCollection constraints,
      String remoteExecutionProperties,
      ImmutableMap<String, String> execProperties,
      Location location) {
    super(PROVIDER, location);

    this.label = label;
    this.constraints = constraints;
    this.remoteExecutionProperties = Strings.nullToEmpty(remoteExecutionProperties);
    this.execProperties = execProperties;
  }

  @Override
  public Label label() {
    return label;
  }

  @Override
  public ConstraintCollection constraints() {
    return constraints;
  }

  @Override
  public String remoteExecutionProperties() {
    return remoteExecutionProperties;
  }

  @Override
  public ImmutableMap<String, String> execProperties() {
    return execProperties;
  }

  @Override
  public void repr(Printer printer) {
    printer.format("PlatformInfo(%s, constraints=%s)", label.toString(), constraints.toString());
  }

  /** Returns a new {@link Builder} for creating a fresh {@link PlatformInfo} instance. */
  public static Builder builder() {
    return new Builder();
  }

  /** Add this platform to the given fingerprint. */
  public void addTo(Fingerprint fp) {
    fp.addString(label.toString());
    fp.addNullableString(remoteExecutionProperties);
    fp.addStringMap(execProperties);
    constraints.addToFingerprint(fp);
  }

  /** Builder class to facilitate creating valid {@link PlatformInfo} instances. */
  public static class Builder {

    @Nullable private PlatformInfo parent = null;
    private Label label;
    private final ConstraintCollection.Builder constraints = ConstraintCollection.builder();
    private String remoteExecutionProperties = null;
    @Nullable private ImmutableMap<String, String> execProperties;
    private Location location = Location.BUILTIN;

    /**
     * Sets the parent {@link PlatformInfo} that this platform inherits from. Constraint values set
     * directly on this instance will be kept, but any other constraint settings will be found from
     * the parent, if set.
     *
     * @param parent the platform that is the parent of this platform
     * @return the {@link Builder} instance for method chaining
     */
    public Builder setParent(@Nullable PlatformInfo parent) {
      this.parent = parent;
      if (parent == null) {
        this.constraints.parent(null);
      } else {
        this.constraints.parent(parent.constraints);
      }
      return this;
    }

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
      this.constraints.addConstraints(constraint);
      return this;
    }

    /**
     * Adds the given constraint values to the constraints that define this {@link PlatformInfo}.
     *
     * @param constraints the constraints to add
     * @return the {@link Builder} instance for method chaining
     */
    public Builder addConstraints(Iterable<ConstraintValueInfo> constraints) {
      this.constraints.addConstraints(constraints);
      return this;
    }

    /** Returns the remote execution properties. */
    @Nullable
    public String getRemoteExecutionProperties() {
      return remoteExecutionProperties;
    }

    /** Returns the exec properties. */
    @Nullable
    public ImmutableMap<String, String> getExecProperties() {
      return execProperties;
    }

    /**
     * Sets the data being sent to a potential remote executor. If there is a parent {@link
     * PlatformInfo} set, the literal string "{PARENT_REMOTE_EXECUTION_PROPERTIES}" will be replaced
     * by the {@link #remoteExecutionProperties} from that parent. Also if the parent is set, and
     * this instance's {@link #remoteExecutionProperties} is blank or unset, the parent's will be
     * used directly.
     *
     * <p>Specific examples:
     *
     * <ul>
     *   <li>parent.remoteExecutionProperties is unset: use the child's value
     *   <li>parent.remoteExecutionProperties is set, child.remoteExecutionProperties is unset: use
     *       the parent's value
     *   <li>parent.remoteExecutionProperties is set, child.remoteExecutionProperties is set, and
     *       does not contain {PARENT_REMOTE_EXECUTION_PROPERTIES}: use the child's value
     *   <li>parent.remoteExecutionProperties is set, child.remoteExecutionProperties is set, and
     *       does contain {PARENT_REMOTE_EXECUTION_PROPERTIES}: use the child's value, but
     *       substitute the parent's value for {PARENT_REMOTE_EXECUTION_PROPERTIES}
     * </ul>
     *
     * @param properties the properties to be added
     * @return the {@link Builder} instance for method chaining
     */
    public Builder setRemoteExecutionProperties(String properties) {
      this.remoteExecutionProperties = properties;
      return this;
    }

    /**
     * Sets the execution properties.
     *
     * <p>If there is a parent {@link PlatformInfo} set, then all parent's properties will be
     * inherited. Any properties included in both will use the child's value. Use the value of empty
     * string to unset a property.
     */
    public Builder setExecProperties(@Nullable ImmutableMap<String, String> properties) {
      this.execProperties = properties;
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

    private void checkRemoteExecutionProperties() throws ExecPropertiesException {
      if (execProperties != null && !Strings.isNullOrEmpty(remoteExecutionProperties)) {
        throw new ExecPropertiesException(
            "Platform contains both remote_execution_properties and exec_properties. Prefer"
                + " exec_properties over the deprecated remote_execution_properties.");
      }
      if (execProperties != null
          && parent != null
          && !Strings.isNullOrEmpty(parent.remoteExecutionProperties())) {
        throw new ExecPropertiesException(
            "Platform specifies exec_properties but its parent "
                + parent.label()
                + " specifies remote_execution_properties. Prefer exec_properties over the"
                + " deprecated remote_execution_properties.");
      }
      if (!Strings.isNullOrEmpty(remoteExecutionProperties)
          && parent != null
          && !parent.execProperties().isEmpty()) {
        throw new ExecPropertiesException(
            "Platform specifies remote_execution_properties but its parent specifies"
                + " exec_properties. Prefer exec_properties over the deprecated"
                + " remote_execution_properties.");
      }
    }

    /**
     * Returns the new {@link PlatformInfo} instance.
     *
     * @throws DuplicateConstraintException if more than one constraint value exists for the same
     *     constraint setting
     */
    public PlatformInfo build() throws DuplicateConstraintException, ExecPropertiesException {
      checkRemoteExecutionProperties();

      // Merge the remote execution properties.
      String remoteExecutionProperties =
          mergeRemoteExecutionProperties(parent, this.remoteExecutionProperties);

      ImmutableMap<String, String> execProperties =
          mergeExecProperties(parent, this.execProperties);
      if (execProperties == null) {
        execProperties = ImmutableMap.of();
      }

      return new PlatformInfo(
          label, constraints.build(), remoteExecutionProperties, execProperties, location);
    }

    private static String mergeRemoteExecutionProperties(
        PlatformInfo parent, String remoteExecutionProperties) {
      String parentRemoteExecutionProperties = "";
      if (parent != null) {
        parentRemoteExecutionProperties = parent.remoteExecutionProperties();
      }

      if (remoteExecutionProperties == null) {
        return parentRemoteExecutionProperties;
      }

      return StringUtilities.replaceAllLiteral(
          remoteExecutionProperties, PARENT_REMOTE_EXECUTION_KEY, parentRemoteExecutionProperties);
    }

    @Nullable
    private static ImmutableMap<String, String> mergeExecProperties(
        PlatformInfo parent, Map<String, String> execProperties) {
      if ((parent == null || parent.execProperties() == null) && execProperties == null) {
        return null;
      }

      HashMap<String, String> result = new HashMap<>();
      if (parent != null && parent.execProperties() != null) {
        result.putAll(parent.execProperties());
      }

      if (execProperties != null) {
        for (Map.Entry<String, String> entry : execProperties.entrySet()) {
          if (Strings.isNullOrEmpty(entry.getValue())) {
            result.remove(entry.getKey());
          } else {
            result.put(entry.getKey(), entry.getValue());
          }
        }
      }

      return ImmutableMap.copyOf(result);
    }
  }

  @Override
  public boolean equals(Object o) {
    if (!(o instanceof PlatformInfo)) {
      return false;
    }
    PlatformInfo that = (PlatformInfo) o;
    return Objects.equals(label, that.label)
        && Objects.equals(constraints, that.constraints)
        && Objects.equals(remoteExecutionProperties, that.remoteExecutionProperties)
        && Objects.equals(execProperties, that.execProperties);
  }

  @Override
  public int hashCode() {
    return Objects.hash(label, constraints, remoteExecutionProperties);
  }

  /** Exception that indicates something is wrong in exec_properties configuration. */
  public static class ExecPropertiesException extends Exception {
    ExecPropertiesException(String message) {
      super(message);
    }
  }
}
