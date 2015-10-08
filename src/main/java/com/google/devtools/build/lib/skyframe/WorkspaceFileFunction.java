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
import com.google.devtools.build.lib.packages.Package.Builder;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.WorkspaceFactory;
import com.google.devtools.build.lib.syntax.EvalException;
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
    RootedPath workspaceRoot = (RootedPath) skyKey.argument();
    FileValue workspaceFileValue = (FileValue) env.getValue(FileValue.key(workspaceRoot));
    if (workspaceFileValue == null) {
      return null;
    }

    Path repoWorkspace = workspaceRoot.getRoot().getRelative(workspaceRoot.getRelativePath());
    Builder builder =
        com.google.devtools.build.lib.packages.Package.newExternalPackageBuilder(
            repoWorkspace, packageFactory.getRuleClassProvider().getRunfilesPrefix());
    try (Mutability mutability = Mutability.create("workspace %s", repoWorkspace)) {
      WorkspaceFactory parser =
          new WorkspaceFactory(
              builder,
              packageFactory.getRuleClassProvider(),
              packageFactory.getEnvironmentExtensions(),
              mutability,
              directories.getEmbeddedBinariesRoot(),
              directories.getWorkspace());
      parser.parse(
          ParserInputSource.create(
              ruleClassProvider.getDefaultWorkspaceFile(), new PathFragment("DEFAULT.WORKSPACE")));
      if (!workspaceFileValue.exists()) {
        return new PackageValue(builder.build());
      }

      try {
        parser.parse(ParserInputSource.create(repoWorkspace, workspaceFileValue.getSize()));
      } catch (IOException e) {
        throw new WorkspaceFileFunctionException(e, Transience.TRANSIENT);
      }
    }

    return new PackageValue(builder.build());
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  private static final class WorkspaceFileFunctionException extends SkyFunctionException {
    public WorkspaceFileFunctionException(IOException e, Transience transience) {
      super(e, transience);
    }

    public WorkspaceFileFunctionException(EvalException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
