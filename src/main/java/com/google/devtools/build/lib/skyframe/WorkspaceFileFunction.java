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

import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Package.LegacyBuilder;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.WorkspaceFactory;
import com.google.devtools.build.lib.skyframe.PackageFunction.PackageFunctionException;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;

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
    final Environment skyEnvironment = env;

    RootedPath workspaceRoot = (RootedPath) skyKey.argument();
    FileValue workspaceFileValue = (FileValue) env.getValue(FileValue.key(workspaceRoot));
    if (workspaceFileValue == null) {
      return null;
    }

    Path repoWorkspace = workspaceRoot.getRoot().getRelative(workspaceRoot.getRelativePath());
    LegacyBuilder builder =
        com.google.devtools.build.lib.packages.Package.newExternalPackageBuilder(
            repoWorkspace, ruleClassProvider.getRunfilesPrefix());
    try (Mutability mutability = Mutability.create("workspace %s", repoWorkspace)) {
      WorkspaceFactory parser =
          new WorkspaceFactory(
              builder,
              ruleClassProvider,
              packageFactory.getEnvironmentExtensions(),
              mutability,
              directories.getEmbeddedBinariesRoot(),
              directories.getWorkspace());
      try {
        PathFragment pathFragment = new PathFragment("/DEFAULT.WORKSPACE");
        if (!parse(
                ParserInputSource.create(ruleClassProvider.getDefaultWorkspaceFile(), pathFragment),
                repoWorkspace, parser, skyEnvironment)) {
          return null;
        }
        if (!workspaceFileValue.exists()) {
          return new PackageValue(builder.build());
        }

        if (!parse(
                ParserInputSource.create(repoWorkspace), repoWorkspace, parser, skyEnvironment)) {
          return null;
        }
      } catch (PackageFunctionException e) {
        throw new WorkspaceFileFunctionException(e, Transience.PERSISTENT);
      } catch (IOException e) {
        for (Event event : parser.getEvents()) {
          env.getListener().handle(event);
        }
        throw new WorkspaceFileFunctionException(e, Transience.TRANSIENT);
      }
    }

    return new PackageValue(builder.build());
  }

  private boolean loadSkylarkImports(Path repoWorkspace, WorkspaceFactory parser,
      Environment skyEnvironment) throws PackageFunctionException, InterruptedException {
    // Load skylark imports
    PackageFunction.SkylarkImportResult importResult;
    importResult = PackageFunction.fetchImportsFromBuildFile(repoWorkspace,
            Label.EXTERNAL_PACKAGE_IDENTIFIER,
            parser.getBuildFileAST(),
            skyEnvironment,
            null);
    if (importResult == null) {
      return false;
    }
    parser.setImportedExtensions(importResult.importMap);
    return true;
  }

  private boolean parse(ParserInputSource source, Path repoWorkspace, WorkspaceFactory parser,
      Environment skyEnvironment)
      throws PackageFunctionException, InterruptedException, IOException {
    parser.parseWorkspaceFile(source);
    if (!loadSkylarkImports(repoWorkspace, parser, skyEnvironment)) {
      return false;
    }
    parser.execute();
    return true;
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
