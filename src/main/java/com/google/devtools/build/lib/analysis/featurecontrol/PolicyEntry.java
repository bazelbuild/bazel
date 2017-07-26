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

package com.google.devtools.build.lib.analysis.featurecontrol;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.cmdline.Label;

/**
 * Policy value object encoding the package group which can access a given feature.
 *
 * @deprecated This is deprecated because the dependency on the package group used to hold the
 *     whitelist is not accessible through blaze query. Use {@link Whitelist}.
 */
@Deprecated
@AutoValue
public abstract class PolicyEntry {
  /** Creates a new PolicyEntry for the given feature and package_group label. */
  public static PolicyEntry create(String feature, Label packageGroupLabel) {
    return new AutoValue_PolicyEntry(feature, packageGroupLabel);
  }

  /** Gets the feature identifier this policy is for. */
  public abstract String getFeature();

  /** Gets the label for the package group which controls access to this feature. */
  public abstract Label getPackageGroupLabel();
}
