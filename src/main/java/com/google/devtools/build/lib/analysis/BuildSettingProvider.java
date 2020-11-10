// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.BuildSetting;
import com.google.devtools.build.lib.packages.RequiredProviders;
import com.google.devtools.build.lib.packages.Type;

/**
 * A built-in provider to allow select()s to know the type, default value, and label when selecting
 * on build settings
 */
public class BuildSettingProvider implements TransitiveInfoProvider {

  public static final RequiredProviders REQUIRE_BUILD_SETTING_PROVIDER =
      RequiredProviders.acceptAnyBuilder()
          .addBuiltinSet(ImmutableSet.of(BuildSettingProvider.class))
          .build();

  private final BuildSetting buildSetting;
  private final Object defaultValue;
  private final Label label;

  public BuildSettingProvider(BuildSetting buildSetting, Object defaultValue, Label label) {
    this.buildSetting = buildSetting;
    this.defaultValue = defaultValue;
    this.label = label;
  }

  public Type<?> getType() {
    return buildSetting.getType();
  }

  public Object getDefaultValue() {
    return defaultValue;
  }

  public Label getLabel() {
    return label;
  }
}
