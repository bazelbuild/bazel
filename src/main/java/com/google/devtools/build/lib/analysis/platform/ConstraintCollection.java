// Copyright 2018 The Bazel Authors. All rights reserved.
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

import static java.util.stream.Collectors.joining;

import com.google.common.base.Functions;
import com.google.common.base.Objects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkbuildapi.platform.ConstraintCollectionApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.util.Fingerprint;
import java.util.function.Function;
import javax.annotation.Nullable;

/** A collection of constraint values. */
@Immutable
@AutoCodec
public final class ConstraintCollection
    implements ConstraintCollectionApi<ConstraintSettingInfo, ConstraintValueInfo> {
  @Nullable private final ConstraintCollection parent;
  private final ImmutableMap<ConstraintSettingInfo, ConstraintValueInfo> constraints;

  /** Creates a new {@link ConstraintCollection} with the given constraint values. */
  public ConstraintCollection(ImmutableList<ConstraintValueInfo> constraints) {
    this(null, constraints);
  }

  /**
   * Creates a new {@link ConstraintCollection} which inherits constraint values from {@code
   * parent}.
   *
   * <p>Any constraints passed in {@code constraints} will override values from {@code parent}.
   */
  public ConstraintCollection(
      @Nullable ConstraintCollection parent, ImmutableList<ConstraintValueInfo> constraints) {
    this(
        parent,
        constraints.stream()
            .collect(
                ImmutableMap.toImmutableMap(ConstraintValueInfo::constraint, Function.identity())));
  }

  /**
   * Returns {@code true} if this {@link ConstraintCollection} contains every {@link
   * ConstraintValueInfo} in {@code expected}, or if the expected constraint value is the default
   * for its setting.
   */
  public boolean containsAll(Iterable<ConstraintValueInfo> expected) {
    return findMissing(expected).isEmpty();
  }

  /**
   * Returns the set of {@link ConstraintValueInfo constraints} from {@code expected} that are not
   * present in this {@link ConstraintCollection}, either directly, or by being the default for
   * their {@link ConstraintSettingInfo}.
   */
  public ImmutableList<ConstraintValueInfo> findMissing(Iterable<ConstraintValueInfo> expected) {
    ImmutableList.Builder<ConstraintValueInfo> missing = new ImmutableList.Builder<>();
    // For every constraint check if it is (1) non-null and (2) set correctly.
    for (ConstraintValueInfo constraint : expected) {
      ConstraintSettingInfo setting = constraint.constraint();
      ConstraintValueInfo targetValue = get(setting);
      if (targetValue == null || !constraint.equals(targetValue)) {
        missing.add(constraint);
      }
    }
    return missing.build();
  }

  /**
   * Returns the set of {@link ConstraintSettingInfo settings} where this {@link
   * ConstraintCollection} and {@code other} have different {@link ConstraintValueInfo values}.
   */
  public ImmutableSet<ConstraintSettingInfo> diff(ConstraintCollection other) {
    ImmutableSet<ConstraintSettingInfo> constraintsToCheck =
        new ImmutableSet.Builder<ConstraintSettingInfo>()
            .addAll(this.constraintSettings())
            .addAll(other.constraintSettings())
            .build();
    ImmutableSet.Builder<ConstraintSettingInfo> mismatchSettings = new ImmutableSet.Builder<>();
    for (ConstraintSettingInfo constraintSetting : constraintsToCheck) {
      ConstraintValueInfo thisConstraint = this.get(constraintSetting);
      ConstraintValueInfo otherConstraint = other.get(constraintSetting);

      if (thisConstraint != null && !thisConstraint.equals(otherConstraint)) {
        mismatchSettings.add(constraintSetting);
      }
    }

    return mismatchSettings.build();
  }

  @AutoCodec.Instantiator
  @VisibleForSerialization
  ConstraintCollection(
      @Nullable ConstraintCollection parent,
      ImmutableMap<ConstraintSettingInfo, ConstraintValueInfo> constraints) {
    this.parent = parent;
    this.constraints = constraints;
  }

  private ConstraintSettingInfo convertKey(Object key, Location loc) throws EvalException {
    if (!(key instanceof ConstraintSettingInfo)) {
      throw new EvalException(
          loc,
          String.format(
              "Constraint names must be platform_common.ConstraintSettingInfo, got %s instead",
              EvalUtils.getDataTypeName(key)));
    }

    return (ConstraintSettingInfo) key;
  }

  @Override
  public boolean has(ConstraintSettingInfo constraint) {
    // First, check locally.
    if (constraints.containsKey(constraint)) {
      return true;
    }

    // Then, check the parent, directly to ignore defaults.
    if (parent != null) {
      return parent.has(constraint);
    }

    return constraint.hasDefaultConstraintValue();
  }

  /**
   * Returns the {@link ConstraintValueInfo} for the given {@link ConstraintSettingInfo}, or {@code
   * null} if none exists.
   */
  @Nullable
  @Override
  public ConstraintValueInfo get(ConstraintSettingInfo constraint) {
    // First, check locally.
    if (constraints.containsKey(constraint)) {
      return constraints.get(constraint);
    }

    // Then, check the parent, directly to ignore defaults.
    if (parent != null) {
      return parent.get(constraint);
    }

    // Finally, Since this constraint isn't set, fall back to the default.
    return constraint.defaultConstraintValue();
  }

  @Override
  public SkylarkList<ConstraintSettingInfo> constraintSettings() {
    return SkylarkList.createImmutable(constraints.keySet());
  }

  @Override
  public Object getIndex(Object key, Location loc) throws EvalException {
    ConstraintSettingInfo constraint = convertKey(key, loc);
    return get(constraint);
  }

  @Override
  public boolean containsKey(Object key, Location loc) throws EvalException {
    ConstraintSettingInfo constraint = convertKey(key, loc);
    return has(constraint);
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }

    if (!(other instanceof ConstraintCollection)) {
      return false;
    }
    ConstraintCollection pc = (ConstraintCollection) other;
    return Objects.equal(this.constraints, pc.constraints);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(constraints);
  }

  @Override
  public String toString() {
    return Printer.str(this);
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    printer.append("<");
    printer.append(
        constraints.values().stream()
            .map(ConstraintValueInfo::label)
            .map(Functions.toStringFunction())
            .collect(joining(", ")));
    printer.append(">");
  }

  /**
   * Adds information to the {@link Fingerprint} to uniquely identify this collection of
   * constraints.
   */
  public void addToFingerprint(Fingerprint fp) {
    fp.addInt(constraints.size());
    constraints.values().forEach(constraintValue -> constraintValue.addTo(fp));
  }
}
