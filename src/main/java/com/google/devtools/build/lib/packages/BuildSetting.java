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
package com.google.devtools.build.lib.packages;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.packages.Type.LabelClass;
import com.google.devtools.build.lib.starlarkbuildapi.config.StarlarkConfigApi.BuildSettingApi;
import net.starlark.java.eval.Printer;

/**
 * Metadata of a build setting rule's properties. This describes the build setting's type (for
 * example, 'int' or 'string'), and whether the build setting corresponds to a command line flag.
 */
public class BuildSetting implements BuildSettingApi {
  private final boolean isFlag;
  private final Type<?> type;
  private final boolean allowMultiple;
  private final boolean repeatable;

  private BuildSetting(boolean isFlag, Type<?> type, boolean allowMultiple, boolean repeatable) {
    this.isFlag = isFlag;
    this.type = type;
    this.allowMultiple = allowMultiple;
    this.repeatable = repeatable;
  }

  public static BuildSetting create(
      boolean isFlag, Type<?> type, boolean allowMultiple, boolean repeatable) {
    return new BuildSetting(isFlag, type, allowMultiple, repeatable);
  }

  public static BuildSetting create(boolean isFlag, Type<?> type) {
    Preconditions.checkState(
        type.getLabelClass() != LabelClass.DEPENDENCY,
        "Build settings should not create a dependency with their default attribute");
    return new BuildSetting(isFlag, type, /* allowMultiple= */ false, false);
  }

  public Type<?> getType() {
    return type;
  }

  @VisibleForTesting
  public boolean isFlag() {
    return isFlag;
  }

  public boolean allowsMultiple() {
    return allowMultiple;
  }

  public boolean isRepeatableFlag() {
    return repeatable;
  }

  @Override
  public void repr(Printer printer) {
    printer.append("<build_setting." + type + ">");
  }
}
