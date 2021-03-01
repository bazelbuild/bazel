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
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;

import com.google.devtools.build.lib.analysis.config.ExecutionTransitionFactory;
import com.google.devtools.build.lib.packages.BuildSetting;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkConfigApi;
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
    return BuildSetting.create(flag, STRING, allowMultiple);
  }

  @Override
  public BuildSetting stringListSetting(Boolean flag) {
    return BuildSetting.create(flag, STRING_LIST);
  }

  @Override
  public ExecutionTransitionFactory exec(Object execGroupUnchecked) {
    return execGroupUnchecked == Starlark.NONE
        ? ExecutionTransitionFactory.create()
        : ExecutionTransitionFactory.create((String) execGroupUnchecked);
  }

  @Override
  public void repr(Printer printer) {
    printer.append("<config>");
  }
}
