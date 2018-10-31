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

package com.google.devtools.build.lib.analysis.skylark;

import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;
import static com.google.devtools.build.lib.syntax.Type.INTEGER;
import static com.google.devtools.build.lib.syntax.Type.STRING;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST;

import com.google.devtools.build.lib.packages.BuildSetting;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkConfigApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;

/**
 * Skylark namespace for creating build settings.
 * TODO(juliexxia): Consider adding more types of build settings, specifically other label types.
 */
public class SkylarkConfig implements SkylarkConfigApi {

  @Override
  public BuildSetting intSetting(Boolean flag) {
    return new BuildSetting(flag, INTEGER);
  }

  @Override
  public BuildSetting boolSetting(Boolean flag) {
    return new BuildSetting(flag, BOOLEAN);
  }

  @Override
  public BuildSetting stringSetting(Boolean flag) {
    return new BuildSetting(flag, STRING);
  }

  @Override
  public BuildSetting stringListSetting(Boolean flag) {
    return new BuildSetting(flag, STRING_LIST);
  }

  @Override
  public BuildSetting labelSetting(Boolean flag) {
    return new BuildSetting(flag, LABEL);
  }

  @Override
  public BuildSetting labelListSetting(Boolean flag) {
    return new BuildSetting(flag, LABEL_LIST);
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    printer.append("<config>");
  }
}
