// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.packages.ExternalPackage;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import javax.annotation.Nullable;

/**
 * Holds the contents of a WORKSPACE file as the //external package.
 */
public class WorkspaceFileValue implements SkyValue {

  private final String workspace;
  private final ExternalPackage pkg;

  public WorkspaceFileValue(String workspace, ExternalPackage pkg) {
    this.workspace = workspace;
    this.pkg = pkg;
  }

  /**
   * Returns the name of this workspace (or null for the default workspace).
   */
  @Nullable
  public String getWorkspace() {
    return workspace;
  }

  /**
   * Returns the //external package.
   */
  public ExternalPackage getPackage() {
    return pkg;
  }

  /**
   * Generates a SkyKey based on the path to the WORKSPACE file.
   */
  public static SkyKey key(RootedPath workspacePath) {
    return new SkyKey(SkyFunctions.WORKSPACE_FILE, workspacePath);
  }

}
