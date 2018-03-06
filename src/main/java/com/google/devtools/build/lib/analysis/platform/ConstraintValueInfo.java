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
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.util.Fingerprint;

/** Provider for a platform constraint value that fulfills a {@link ConstraintSettingInfo}. */
@SkylarkModule(
  name = "ConstraintValueInfo",
  doc = "A value for a constraint setting that can be used to define a platform.",
  category = SkylarkModuleCategory.PROVIDER
)
@Immutable
@AutoCodec
public class ConstraintValueInfo extends NativeInfo {
  /** Name used in Skylark for accessing this provider. */
  public static final String SKYLARK_NAME = "ConstraintValueInfo";

  /** Skylark constructor and identifier for this provider. */
  public static final NativeProvider<ConstraintValueInfo> SKYLARK_CONSTRUCTOR =
      new NativeProvider<ConstraintValueInfo>(ConstraintValueInfo.class, SKYLARK_NAME) {};

  private final ConstraintSettingInfo constraint;
  private final Label label;

  @VisibleForSerialization
  ConstraintValueInfo(ConstraintSettingInfo constraint, Label label, Location location) {
    super(
        SKYLARK_CONSTRUCTOR,
        location);

    this.constraint = constraint;
    this.label = label;
  }

  @SkylarkCallable(
    name = "constraint",
    doc =
        "The <a href=\"ConstraintSettingInfo.html\">ConstraintSettingInfo</a> this value can be "
            + "applied to.",
    structField = true
  )
  public ConstraintSettingInfo constraint() {
    return constraint;
  }

  @SkylarkCallable(
    name = "label",
    doc = "The label of the target that created this constraint value.",
    structField = true
  )
  public Label label() {
    return label;
  }

  /** Returns a new {@link ConstraintValueInfo} with the given data. */
  public static ConstraintValueInfo create(ConstraintSettingInfo constraint, Label value) {
    return create(constraint, value, Location.BUILTIN);
  }

  /** Returns a new {@link ConstraintValueInfo} with the given data. */
  public static ConstraintValueInfo create(
      ConstraintSettingInfo constraint, Label value, Location location) {
    return new ConstraintValueInfo(constraint, value, location);
  }

  /** Add this constraint value to the given fingerprint. */
  public void addTo(Fingerprint fp) {
    this.constraint.addTo(fp);
    fp.addString(label.getCanonicalForm());
  }
}
