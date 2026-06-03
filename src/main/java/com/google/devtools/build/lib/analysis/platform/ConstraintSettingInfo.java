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

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ConstraintSettingInfoApi;
import com.google.devtools.build.lib.util.Fingerprint;
import java.util.Objects;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import net.starlark.java.eval.Printer;

/** Provider for a platform constraint setting that is available to be fulfilled. */
@Immutable
public class ConstraintSettingInfo extends NativeInfo implements ConstraintSettingInfoApi {
  /** Name used in Starlark for accessing this provider. */
  public static final String STARLARK_NAME = "ConstraintSettingInfo";

  /** Provider singleton constant. */
  public static final BuiltinProvider<ConstraintSettingInfo> PROVIDER =
      new BuiltinProvider<ConstraintSettingInfo>(STARLARK_NAME, ConstraintSettingInfo.class) {};

  private final Label label;
  @Nullable private final Label defaultConstraintValueLabel;
  @Nullable private final ConstraintValueInfo refinedConstraintValue;

  private ConstraintSettingInfo(
      Label label,
      @Nullable Label defaultConstraintValueLabel,
      @Nullable ConstraintValueInfo refinedConstraintValue) {
    this.label = label;
    this.defaultConstraintValueLabel = defaultConstraintValueLabel;
    this.refinedConstraintValue = refinedConstraintValue;
  }

  @Override
  public BuiltinProvider<ConstraintSettingInfo> getProvider() {
    return PROVIDER;
  }

  @Override
  public Label label() {
    return label;
  }

  @Override
  public boolean hasDefaultConstraintValue() {
    return defaultConstraintValueLabel != null;
  }

  @Override
  @Nullable
  public ConstraintValueInfo defaultConstraintValue() {
    if (!hasDefaultConstraintValue()) {
      return null;
    }
    return ConstraintValueInfo.create(this, defaultConstraintValueLabel);
  }

  /** Returns the refined constraint value, or {@code null} if not set. */
  @Nullable
  public ConstraintValueInfo refinedConstraintValue() {
    return refinedConstraintValue;
  }

  /**
   * Iterates the constraint value labels that any {@code constraint_value} for this setting
   * recursively implies via the {@code refines_constraint_value} chain.
   */
  public Iterable<Label> refinementChain() {
    return () ->
        Stream.iterate(
                refinedConstraintValue,
                Objects::nonNull,
                cv -> cv.constraint().refinedConstraintValue)
            .map(ConstraintValueInfo::label)
            .iterator();
  }

  /** Add this constraint setting to the given fingerprint. */
  public void addTo(Fingerprint fp) {
    fp.addString(label.getCanonicalForm());
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof ConstraintSettingInfo otherConstraint)) {
      return false;
    }

    return Objects.equals(label, otherConstraint.label);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(label);
  }

  @Override
  public void repr(Printer printer) {
    printer.append("ConstraintSettingInfo(").append(label.toString());
    if (defaultConstraintValueLabel != null) {
      printer.append(", default_constraint_value=").append(defaultConstraintValueLabel.toString());
    }
    if (refinedConstraintValue != null) {
      printer.append(", refines_constraint_value=").str(refinedConstraintValue.label(), semantics);
    }
    printer.append(")");
  }

  /** Returns a new {@link ConstraintSettingInfo} with the given data. */
  public static ConstraintSettingInfo create(Label constraintSetting) {
    return create(
        constraintSetting, /* defaultConstraintValue= */ null, /* refinedConstraintValue= */ null);
  }

  /** Returns a new {@link ConstraintSettingInfo} with the given data. */
  public static ConstraintSettingInfo create(
      Label constraintSetting,
      @Nullable Label defaultConstraintValue,
      @Nullable ConstraintValueInfo refinedConstraintValue) {
    return new ConstraintSettingInfo(
        constraintSetting, defaultConstraintValue, refinedConstraintValue);
  }
}
