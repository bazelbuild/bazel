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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/** A SkyValue representing the code paths that are owned by a project. */
public final class ProjectOwnedCodePathsValue implements SkyValue {

  private final ImmutableSet<String> ownedCodePaths;

  public ProjectOwnedCodePathsValue(ImmutableSet<String> ownedCodePaths) {
    this.ownedCodePaths = ownedCodePaths;
  }

  public ImmutableSet<String> getOwnedCodePaths() {
    return ownedCodePaths;
  }

  /** The SkyKey. Uses the label of the project file as the input. */
  public static final class Key implements SkyKey {
    private final Label projectFile;

    public Key(Label projectFile) {
      this.projectFile = Preconditions.checkNotNull(projectFile);
    }

    public Label getProjectFile() {
      return projectFile;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.PROJECT_DIRECTORIES;
    }
  }
}
