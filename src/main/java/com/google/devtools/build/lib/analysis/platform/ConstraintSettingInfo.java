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

/** Provider for a platform constraint setting that is available to be fulfilled. */
@SkylarkModule(
  name = "ConstraintSettingInfo",
  doc = "A specific constraint setting that may be used to define a platform.",
  category = SkylarkModuleCategory.PROVIDER
)
@Immutable
@AutoCodec
public class ConstraintSettingInfo extends NativeInfo {
  /** Name used in Skylark for accessing this provider. */
  public static final String SKYLARK_NAME = "ConstraintSettingInfo";

  /** Skylark constructor and identifier for this provider. */
  public static final NativeProvider<ConstraintSettingInfo> PROVIDER =
      new NativeProvider<ConstraintSettingInfo>(ConstraintSettingInfo.class, SKYLARK_NAME) {};

  private final Label label;

  @VisibleForSerialization
  ConstraintSettingInfo(Label label, Location location) {
    super(PROVIDER, location);

    this.label = label;
  }

  @SkylarkCallable(
    name = "label",
    doc = "The label of the target that created this constraint.",
    structField = true
  )
  public Label label() {
    return label;
  }

  /** Returns a new {@link ConstraintSettingInfo} with the given data. */
  public static ConstraintSettingInfo create(Label constraintSetting) {
    return create(constraintSetting, Location.BUILTIN);
  }

  /** Returns a new {@link ConstraintSettingInfo} with the given data. */
  public static ConstraintSettingInfo create(Label constraintSetting, Location location) {
    return new ConstraintSettingInfo(constraintSetting, location);
  }

  /** Add this constraint setting to the given fingerprint. */
  public void addTo(Fingerprint fp) {
    fp.addString(label.getCanonicalForm());
  }
}
