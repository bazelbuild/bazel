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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.ClassObjectConstructor;
import com.google.devtools.build.lib.packages.NativeClassObjectConstructor;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.packages.SkylarkProviderIdentifier;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.util.Preconditions;

/** Provider for a platform constraint setting that is available to be fulfilled. */
@SkylarkModule(
  name = "ConstraintSettingInfo",
  doc = "A specific constraint setting that may be used to define a platform.",
  category = SkylarkModuleCategory.PROVIDER
)
@Immutable
public class ConstraintSettingInfo extends SkylarkClassObject {

  /** Name used in Skylark for accessing this provider. */
  static final String SKYLARK_NAME = "ConstraintSettingInfo";

  /** Skylark constructor and identifier for this provider. */
  static final ClassObjectConstructor SKYLARK_CONSTRUCTOR =
      new NativeClassObjectConstructor(SKYLARK_NAME) {};

  /** Identifier used to retrieve this provider from rules which export it. */
  public static final SkylarkProviderIdentifier SKYLARK_IDENTIFIER =
      SkylarkProviderIdentifier.forKey(SKYLARK_CONSTRUCTOR.getKey());

  private final Label label;

  private ConstraintSettingInfo(Label label) {
    super(SKYLARK_CONSTRUCTOR, ImmutableMap.<String, Object>of("label", label));

    this.label = label;
  }

  public Label label() {
    return label;
  }

  /** Retrieves and casts the provider from the given target. */
  public static ConstraintSettingInfo fromTarget(TransitiveInfoCollection target) {
    Object provider = target.get(SKYLARK_IDENTIFIER);
    if (provider == null) {
      return null;
    }
    Preconditions.checkState(provider instanceof ConstraintSettingInfo);
    return (ConstraintSettingInfo) provider;
  }

  /** Retrieves and casts the providers from the given targets. */
  public static Iterable<ConstraintSettingInfo> fromTargets(
      Iterable<? extends TransitiveInfoCollection> targets) {
    return Iterables.transform(
        targets,
        new Function<TransitiveInfoCollection, ConstraintSettingInfo>() {
          @Override
          public ConstraintSettingInfo apply(TransitiveInfoCollection target) {
            return fromTarget(target);
          }
        });
  }

  /** Returns a new {@link ConstraintSettingInfo} with the given data. */
  public static ConstraintSettingInfo create(Label constraintSetting) {
    return new ConstraintSettingInfo(constraintSetting);
  }
}
