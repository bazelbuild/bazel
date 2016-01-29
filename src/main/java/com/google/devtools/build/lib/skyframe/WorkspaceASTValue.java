// Copyright 2016 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * A SkyValue that stores the parsed WORKSPACE file as an AST.
 */
public class WorkspaceASTValue implements SkyValue {

  private final BuildFileAST ast;

  public WorkspaceASTValue(BuildFileAST ast) {
    Preconditions.checkNotNull(ast);
    this.ast = ast;
  }

  public BuildFileAST getAST() {
    return ast;
  }

  public static SkyKey key(RootedPath path) {
    return new SkyKey(SkyFunctions.WORKSPACE_AST, path);
  }
}

