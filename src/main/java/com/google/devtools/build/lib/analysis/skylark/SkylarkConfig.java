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

import com.google.devtools.build.lib.skylarkbuildapi.SkylarkConfigApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.Type;

/**
 * Skylark namespace for creating build setting descriptors.
 */
public class SkylarkConfig implements SkylarkConfigApi {

  @Override
  public BuildSettingDescriptor intSetting(Boolean flag) {
    return new BuildSettingDescriptor(flag, INTEGER);
  }

  @Override
  public BuildSettingDescriptor boolSetting(Boolean flag) {
    return new BuildSettingDescriptor(flag, BOOLEAN);
  }

  @Override
  public BuildSettingDescriptor stringSetting(Boolean flag) {
    return new BuildSettingDescriptor(flag, STRING);
  }

  @Override
  public BuildSettingDescriptor stringListSetting(Boolean flag) {
    return new BuildSettingDescriptor(flag, STRING_LIST);
  }

  @Override
  public BuildSettingDescriptor labelSetting(Boolean flag) {
    return new BuildSettingDescriptor(flag, LABEL);
  }

  @Override
  public BuildSettingDescriptor labelListSetting(Boolean flag) {
    return new BuildSettingDescriptor(flag, LABEL_LIST);
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    printer.append("<config>");
  }

  /**
   * An object that describes what time of build setting a skylark rule is (if any type).
   */
  public static final class BuildSettingDescriptor implements SkylarkConfigApi.BuildSettingApi {
    private boolean isFlag;
    private Type<?> type;

    BuildSettingDescriptor(boolean isFlag, Type<?> type) {
      this.isFlag = isFlag;
      this.type = type;
    }

    public Type<?> getType() {
      return type;
    }

    @Override
    public void repr(SkylarkPrinter printer) {
      printer.append("<build_setting." + type.toString() + ">");
    }
  }
}
