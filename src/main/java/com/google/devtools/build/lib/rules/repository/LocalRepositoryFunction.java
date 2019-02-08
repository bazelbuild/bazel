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

package com.google.devtools.build.lib.rules.repository;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.events.ExtendedEventHandler.ResolvedEvent;
import com.google.devtools.build.lib.packages.BuildFileName;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Access a repository on the local filesystem.
 */
public class LocalRepositoryFunction extends RepositoryFunction {

  @Override
  public boolean isLocal(Environment env, FileSystem fileSystem,
      Rule rule) {
    return true;
  }

  @Override
  public RepositoryDirectoryValue.Builder fetch(Rule rule, Path outputDirectory,
      BlazeDirectories directories, Environment env, Map<String, String> markerData)
      throws InterruptedException, RepositoryFunctionException {
    PathFragment pathFragment = RepositoryFunction.getTargetPath(rule, directories.getWorkspace());
    RepositoryDirectoryValue.Builder result =
        LocalRepositoryFunction.symlink(outputDirectory, pathFragment, env);
    if (result != null) {
      env.getListener().post(resolve(rule, directories));
    }
    return result;
  }

  public static RepositoryDirectoryValue.Builder symlink(
      Path source, PathFragment destination, Environment env)
      throws RepositoryFunctionException, InterruptedException {
    try {
      source.createSymbolicLink(destination);
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException("Could not create symlink to repository " + destination + ": "
              + e.getMessage(), e), Transience.TRANSIENT);
    }
    FileValue repositoryValue = getRepositoryDirectory(source, env);
    if (repositoryValue == null) {
      // TODO(bazel-team): If this returns null, we unnecessarily recreate the symlink above on the
      // second execution.
      return null;
    }

    if (!repositoryValue.isDirectory()) {
      throw new RepositoryFunctionException(
          new IOException(source + " must be an existing directory"), Transience.PERSISTENT);
    }

    // Check that the repository contains a WORKSPACE file.
    // It's important to check the real path, otherwise this looks under the "external/[repo]" path
    // and cause a Skyframe cycle in the lookup.
    FileValue workspaceFileValue = getWorkspaceFile(repositoryValue.realRootedPath(), env);
    if (workspaceFileValue == null) {
      return null;
    }

    if (!workspaceFileValue.exists()) {
      throw new RepositoryFunctionException(
          new IOException("No WORKSPACE file found in " + source), Transience.PERSISTENT);
    }

    return RepositoryDirectoryValue.builder().setPath(source);
  }

  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return LocalRepositoryRule.class;
  }

  @Nullable
  protected static FileValue getWorkspaceFile(RootedPath directory, Environment env)
      throws RepositoryFunctionException, InterruptedException {
    RootedPath workspaceRootedFile =
        RootedPath.toRootedPath(
            directory.getRoot(),
            directory
                .getRootRelativePath()
                .getRelative(BuildFileName.WORKSPACE.getFilenameFragment()));

    SkyKey workspaceFileKey = FileValue.key(workspaceRootedFile);
    FileValue value;
    try {
      value = (FileValue) env.getValueOrThrow(workspaceFileKey, IOException.class);
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException("Could not access " + workspaceRootedFile + ": " + e.getMessage()),
          Transience.PERSISTENT);
    }
    return value;
  }

  private static ResolvedEvent resolve(Rule rule, BlazeDirectories directories) {
    String name = rule.getName();
    Object pathObj = rule.getAttributeContainer().getAttr("path");
    String path;
    if (pathObj instanceof String) {
      path = (String) pathObj;
    } else {
      path = "";
    }
    // Find a descrption of the path; there is a case, where we do not(!) want to hard-code
    // the argument we obtained: if the path is under the embedded binaries root.
    String pathArg;
    PathFragment pathFragment = PathFragment.create(path);
    PathFragment embeddedDir = directories.getEmbeddedBinariesRoot().asFragment();
    if (pathFragment.isAbsolute() && pathFragment.startsWith(embeddedDir)) {
      pathArg =
          "__embedded_dir__ + \"/\" + "
              + Printer.getPrinter().repr(pathFragment.relativeTo(embeddedDir).toString());
    } else {
      pathArg = Printer.getPrinter().repr(path).toString();
    }
    String repr =
        "local_repository(name = " + Printer.getPrinter().repr(name) + ", path = " + pathArg + ")";
    return new ResolvedEvent() {
      @Override
      public String getName() {
        return name;
      }

      @Override
      public Object getResolvedInformation() {
        return ImmutableMap.<String, Object>builder()
            .put(ResolvedHashesFunction.ORIGINAL_RULE_CLASS, "local_repository")
            .put(
                ResolvedHashesFunction.ORIGINAL_ATTRIBUTES,
                ImmutableMap.<String, Object>builder().put("name", name).put("path", path).build())
            .put(ResolvedHashesFunction.NATIVE, repr)
            .build();
      }
    };
  }
}
