// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;
import javax.annotation.Nullable;

/** A {@link SkyFunction} that loads the owned code paths from a project file. */
public class ProjectOwnedCodePathsFunction implements SkyFunction {

  private static final String TOP_LEVEL_VARIABLE_NAME = "owned_code_paths";

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    ProjectOwnedCodePathsValue.Key key = (ProjectOwnedCodePathsValue.Key) skyKey.argument();

    BzlLoadValue bzlLoadValue =
        (BzlLoadValue) env.getValue(BzlLoadValue.keyForBuild(key.getProjectFile()));
    if (bzlLoadValue == null) {
      return null;
    }

    Object ret = bzlLoadValue.getModule().getGlobal(TOP_LEVEL_VARIABLE_NAME);
    if (ret == null) {
      return new ProjectOwnedCodePathsValue(ImmutableSet.of());
    } else {
      @SuppressWarnings("unchecked")
      Collection<? extends String> dirs = (Collection<? extends String>) ret;
      return new ProjectOwnedCodePathsValue(ImmutableSet.copyOf(dirs));
    }
  }
}
