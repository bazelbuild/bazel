// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Package.LegacyBuilder;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.WorkspaceFactory;
import com.google.devtools.build.lib.skyframe.PackageFunction.PackageFunctionException;
import com.google.devtools.build.lib.skyframe.WorkspaceFileValue.WorkspaceFileKey;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Environment.Extension;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * A SkyFunction to parse WORKSPACE files.
 */
public class WorkspaceFileFunction implements SkyFunction {

  private final PackageFactory packageFactory;
  private final BlazeDirectories directories;
  private final RuleClassProvider ruleClassProvider;

  public WorkspaceFileFunction(
      RuleClassProvider ruleClassProvider,
      PackageFactory packageFactory,
      BlazeDirectories directories) {
    this.packageFactory = packageFactory;
    this.directories = directories;
    this.ruleClassProvider = ruleClassProvider;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws WorkspaceFileFunctionException,
      InterruptedException {

    WorkspaceFileKey key = (WorkspaceFileKey) skyKey.argument();
    RootedPath workspaceRoot = key.getPath();
    WorkspaceASTValue workspaceASTValue =
        (WorkspaceASTValue) env.getValue(WorkspaceASTValue.key(workspaceRoot));
    if (workspaceASTValue == null) {
      return null;
    }

    Path repoWorkspace = workspaceRoot.getRoot().getRelative(workspaceRoot.getRelativePath());
    LegacyBuilder builder =
        Package.newExternalPackageBuilder(repoWorkspace, ruleClassProvider.getRunfilesPrefix());

    if (workspaceASTValue.getASTs().isEmpty()) {
      return new WorkspaceFileValue(
          builder.build(), // resulting package
          ImmutableMap.<String, Extension>of(), // list of imports
          ImmutableMap.<String, Object>of(), // list of symbol bindings
          workspaceRoot, // Workspace root
          0, // first fragment, idx = 0
          false); // last fragment
    }
    WorkspaceFactory parser;
    try (Mutability mutability = Mutability.create("workspace %s", repoWorkspace)) {
      parser =
          new WorkspaceFactory(
              builder,
              ruleClassProvider,
              packageFactory.getEnvironmentExtensions(),
              mutability,
              key.getIndex() == 0,
              directories.getEmbeddedBinariesRoot(),
              directories.getWorkspace());
      if (key.getIndex() > 0) {
        WorkspaceFileValue prevValue =
            (WorkspaceFileValue)
                env.getValue(WorkspaceFileValue.key(key.getPath(), key.getIndex() - 1));
        if (prevValue == null) {
          return null;
        }
        if (prevValue.next() == null) {
          return prevValue;
        }
        parser.setParent(prevValue.getPackage(), prevValue.getImportMap(), prevValue.getBindings());
      }
      BuildFileAST ast = workspaceASTValue.getASTs().get(key.getIndex());
      PackageFunction.SkylarkImportResult importResult =
          PackageFunction.fetchImportsFromBuildFile(
              repoWorkspace, Label.EXTERNAL_PACKAGE_IDENTIFIER, ast, env, null);
      if (importResult == null) {
        return null;
      }
      parser.execute(ast, importResult.importMap);
    } catch (PackageFunctionException | NameConflictException e) {
      throw new WorkspaceFileFunctionException(e, Transience.PERSISTENT);
    }

    return new WorkspaceFileValue(
        builder.build(),
        parser.getImportMap(),
        parser.getVariableBindings(),
        workspaceRoot,
        key.getIndex(),
        key.getIndex() < workspaceASTValue.getASTs().size() - 1);
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  private static final class WorkspaceFileFunctionException extends SkyFunctionException {
    public WorkspaceFileFunctionException(Exception e, Transience transience) {
      super(e, transience);
    }

    public WorkspaceFileFunctionException(Exception e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
