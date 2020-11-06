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

import com.google.common.base.Objects;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ConstraintSettingInfoApi;
import com.google.devtools.build.lib.util.Fingerprint;
import javax.annotation.Nullable;
import net.starlark.java.eval.Printer;

/** Provider for a platform constraint setting that is available to be fulfilled. */
@Immutable
@AutoCodec
public class ConstraintSettingInfo extends NativeInfo implements ConstraintSettingInfoApi {
  /** Name used in Starlark for accessing this provider. */
  public static final String STARLARK_NAME = "ConstraintSettingInfo";

  /** Provider singleton constant. */
  public static final BuiltinProvider<ConstraintSettingInfo> PROVIDER =
      new BuiltinProvider<ConstraintSettingInfo>(STARLARK_NAME, ConstraintSettingInfo.class) {};

  private final Label label;
  @Nullable private final Label defaultConstraintValueLabel;

  @VisibleForSerialization
  ConstraintSettingInfo(Label label, Label defaultConstraintValueLabel) {
    this.label = label;
    this.defaultConstraintValueLabel = defaultConstraintValueLabel;
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

  /** Add this constraint setting to the given fingerprint. */
  public void addTo(Fingerprint fp) {
    fp.addString(label.getCanonicalForm());
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof ConstraintSettingInfo)) {
      return false;
    }

    ConstraintSettingInfo otherConstraint = (ConstraintSettingInfo) other;
    return Objects.equal(label, otherConstraint.label);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(label);
  }

  @Override
  public void repr(Printer printer) {
    Printer.format(printer, "ConstraintSettingInfo(%s", label.toString());
    if (defaultConstraintValueLabel != null) {
      Printer.format(
          printer, ", default_constraint_value=%s", defaultConstraintValueLabel.toString());
    }
    printer.append(")");
  }

  /** Returns a new {@link ConstraintSettingInfo} with the given data. */
  public static ConstraintSettingInfo create(Label constraintSetting) {
    return create(constraintSetting, null);
  }

  /** Returns a new {@link ConstraintSettingInfo} with the given data. */
  public static ConstraintSettingInfo create(
      Label constraintSetting, Label defaultConstraintValue) {
    return new ConstraintSettingInfo(constraintSetting, defaultConstraintValue);
  }
}
