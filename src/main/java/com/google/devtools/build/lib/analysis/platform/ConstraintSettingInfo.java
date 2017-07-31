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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.SkylarkType;

/** Provider for a platform constraint setting that is available to be fulfilled. */
@SkylarkModule(
  name = "ConstraintSettingInfo",
  doc = "A specific constraint setting that may be used to define a platform.",
  category = SkylarkModuleCategory.PROVIDER
)
@Immutable
public class ConstraintSettingInfo extends Info {

  /** Name used in Skylark for accessing this provider. */
  public static final String SKYLARK_NAME = "ConstraintSettingInfo";

  private static final FunctionSignature.WithValues<Object, SkylarkType> SIGNATURE =
      FunctionSignature.WithValues.create(
          FunctionSignature.of(
              /*numMandatoryPositionals=*/ 1,
              /*numOptionalPositionals=*/ 0,
              /*numMandatoryNamedOnly*/ 0,
              /*starArg=*/ false,
              /*kwArg=*/ false,
              /*names=*/ "label"),
          /*defaultValues=*/ null,
          /*types=*/ ImmutableList.<SkylarkType>of(SkylarkType.of(Label.class)));

  /** Skylark constructor and identifier for this provider. */
  public static final NativeProvider<ConstraintSettingInfo> SKYLARK_CONSTRUCTOR =
      new NativeProvider<ConstraintSettingInfo>(
          ConstraintSettingInfo.class, SKYLARK_NAME, SIGNATURE) {
        @Override
        protected ConstraintSettingInfo createInstanceFromSkylark(Object[] args, Location loc)
            throws EvalException {
          // Based on SIGNATURE above, the args are label.
          Label label = (Label) args[0];
          return ConstraintSettingInfo.create(label, loc);
        }
      };

  private final Label label;

  private ConstraintSettingInfo(Label label, Location location) {
    super(SKYLARK_CONSTRUCTOR, ImmutableMap.<String, Object>of("label", label), location);

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
}
