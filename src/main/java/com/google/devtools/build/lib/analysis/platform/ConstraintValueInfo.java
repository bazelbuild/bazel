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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.ClassObjectConstructor;
import com.google.devtools.build.lib.packages.NativeClassObjectConstructor;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.packages.SkylarkProviderIdentifier;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/** Provider for a platform constraint value that fulfills a {@link ConstraintSettingInfo}. */
@SkylarkModule(
  name = "ConstraintValueProvider",
  doc = "A value for a constraint setting that can be used to define a platform.",
  category = SkylarkModuleCategory.PROVIDER
)
@Immutable
public class ConstraintValueInfo extends SkylarkClassObject {

  /** Name used in Skylark for accessing this provider. */
  public static final String SKYLARK_NAME = "ConstraintValueInfo";

  /** Skylark constructor and identifier for this provider. */
  public static final ClassObjectConstructor SKYLARK_CONSTRUCTOR =
      new NativeClassObjectConstructor(SKYLARK_NAME) {};

  /** Identifier used to retrieve this provider from rules which export it. */
  public static final SkylarkProviderIdentifier SKYLARK_IDENTIFIER =
      SkylarkProviderIdentifier.forKey(SKYLARK_CONSTRUCTOR.getKey());

  private final ConstraintSettingInfo constraint;
  private final Label label;

  private ConstraintValueInfo(ConstraintSettingInfo constraint, Label label) {
    super(
        SKYLARK_CONSTRUCTOR,
        ImmutableMap.<String, Object>of(
            "constraint", constraint,
            "label", label));

    this.constraint = constraint;
    this.label = label;
  }

  public ConstraintSettingInfo constraint() {
    return constraint;
  }

  public Label label() {
    return label;
  }

  /** Returns a new {@link ConstraintValueInfo} with the given data. */
  public static ConstraintValueInfo create(ConstraintSettingInfo constraint, Label value) {
    return new ConstraintValueInfo(constraint, value);
  }
}
