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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.LoadStatement;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.syntax.Statement;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.util.List;

/**
 * A SkyFunction to parse WORKSPACE files into a BuildFileAST.
 */
public class WorkspaceASTFunction implements SkyFunction {
  private final RuleClassProvider ruleClassProvider;

  public WorkspaceASTFunction(RuleClassProvider ruleClassProvider) {
    this.ruleClassProvider = ruleClassProvider;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, WorkspaceASTFunctionException {
    RootedPath workspaceRoot = (RootedPath) skyKey.argument();
    FileValue workspaceFileValue = (FileValue) env.getValue(FileValue.key(workspaceRoot));
    if (workspaceFileValue == null) {
      return null;
    }

    Path repoWorkspace = workspaceRoot.getRoot().getRelative(workspaceRoot.getRelativePath());
    PathFragment pathFragment = new PathFragment("/DEFAULT.WORKSPACE");
    try {
      BuildFileAST ast = BuildFileAST.parseBuildFile(
          ParserInputSource.create(ruleClassProvider.getDefaultWorkspaceFile(), pathFragment),
          env.getListener(), false);
      if (ast.containsErrors()) {
        throw new WorkspaceASTFunctionException(
            new IOException("Failed to parse default WORKSPACE file"), Transience.PERSISTENT);
      }
      if (workspaceFileValue.exists()) {
        ast = BuildFileAST.parseBuildFile(
            ParserInputSource.create(repoWorkspace), ast.getStatements(), env.getListener(), false);
        if (ast.containsErrors()) {
          throw new WorkspaceASTFunctionException(
              new IOException("Failed to parse WORKSPACE file"), Transience.PERSISTENT);
        }
      }
      return new WorkspaceASTValue(splitAST(ast));
    } catch (IOException ex) {
      throw new WorkspaceASTFunctionException(ex, Transience.TRANSIENT);
    }
  }

  /**
   * Cut {@code ast} into a list of AST separated by load statements. We cut right before each load
   * statement series.
   */
  private static ImmutableList<BuildFileAST> splitAST(BuildFileAST ast) {
    ImmutableList.Builder<BuildFileAST> asts = ImmutableList.builder();
    int prevIdx = 0;
    boolean lastIsLoad = true; // don't cut if the first statement is a load.
    List<Statement> statements = ast.getStatements();
    for (int idx = 0; idx < statements.size(); idx++) {
      Statement st = statements.get(idx);
      if (st instanceof LoadStatement) {
        if (!lastIsLoad) {
          asts.add(ast.subTree(prevIdx, idx));
          prevIdx = idx;
        }
        lastIsLoad = true;
      } else {
        lastIsLoad = false;
      }
    }
    if (!statements.isEmpty()) {
      asts.add(ast.subTree(prevIdx, statements.size()));
    }
    return asts.build();
  }

  private static final class WorkspaceASTFunctionException extends SkyFunctionException {
    public WorkspaceASTFunctionException(Exception e, Transience transience) {
      super(e, transience);
    }
  }
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
