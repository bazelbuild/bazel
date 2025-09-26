// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.starlark;

import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.INTEGER;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.Types.STRING_LIST;
import static com.google.devtools.build.lib.packages.Types.STRING_SET;

import com.google.devtools.build.lib.analysis.config.ExecutionTransitionFactory;
import com.google.devtools.build.lib.analysis.config.transitions.NoConfigTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.packages.BuildSetting;
import com.google.devtools.build.lib.starlarkbuildapi.config.ConfigurationTransitionApi;
import com.google.devtools.build.lib.starlarkbuildapi.config.StarlarkConfigApi;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;

/** Starlark namespace for creating build settings. */
public class StarlarkConfig implements StarlarkConfigApi {

  @Override
  public BuildSetting intSetting(Boolean flag) {
    return BuildSetting.create(flag, INTEGER);
  }

  @Override
  public BuildSetting boolSetting(Boolean flag) {
    return BuildSetting.create(flag, BOOLEAN);
  }

  @Override
  public BuildSetting stringSetting(Boolean flag, Boolean allowMultiple) {
    return BuildSetting.create(flag, STRING, allowMultiple, false);
  }

  @Override
  public BuildSetting stringListSetting(Boolean flag, Boolean repeatable) throws EvalException {
    if (repeatable && !flag) {
      throw Starlark.errorf("'repeatable' can only be set for a setting with 'flag = True'");
    }
    return BuildSetting.create(flag, STRING_LIST, false, repeatable);
  }

  @Override
  public BuildSetting stringSetSetting(Boolean flag, Boolean repeatable) throws EvalException {
    if (repeatable && !flag) {
      throw Starlark.errorf("'repeatable' can only be set for a setting with 'flag = True'");
    }
    return BuildSetting.create(flag, STRING_SET, false, repeatable);
  }

  @Override
  public ExecutionTransitionFactory exec(Object execGroupUnchecked) {
    return execGroupUnchecked == Starlark.NONE
        ? ExecutionTransitionFactory.createFactory()
        : ExecutionTransitionFactory.createFactory((String) execGroupUnchecked);
  }

  @Override
  public ConfigurationTransitionApi target() {
    return (ConfigurationTransitionApi) NoTransition.getFactory();
  }

  @Override
  public ConfigurationTransitionApi none() {
    return (ConfigurationTransitionApi) NoConfigTransition.getFactory();
  }

  @Override
  public void repr(Printer printer) {
    printer.append("<config>");
  }
}
