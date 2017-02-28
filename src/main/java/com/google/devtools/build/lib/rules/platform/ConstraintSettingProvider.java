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

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/** Provider for a platform constraint setting that is available to be fulfilled. */
@AutoValue
@Immutable
public abstract class ConstraintSettingProvider implements TransitiveInfoProvider {
  public abstract Label constraintSetting();

  public static ConstraintSettingProvider create(Label constraintSetting) {
    return new AutoValue_ConstraintSettingProvider(constraintSetting);
  }
}
