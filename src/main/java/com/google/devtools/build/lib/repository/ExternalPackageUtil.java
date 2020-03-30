// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.repository;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.BuildFileName;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import javax.annotation.Nullable;

/** Utility class to centralize looking up data from the external package. */
public class ExternalPackageUtil {
  /**
   * Returns directories, that should not be symlinked under the execroot.
   *
   * <p>Searches for toplevel_output_directories calls in the WORKSPACE file, and gathers values of
   * all "paths" attributes.
   */
  public static ImmutableSortedSet<String> getNotSymlinkedInExecrootDirectories(Environment env)
      throws InterruptedException {
    ImmutableSortedSet.Builder<String> builder = ImmutableSortedSet.naturalOrder();
    WorkspaceFileValueProcessor gatherer =
        workspaceFileValue -> {
          ImmutableSortedSet<String> paths = workspaceFileValue.getDoNotSymlinkInExecrootPaths();
          if (paths != null) {
            builder.addAll(paths);
          }
          // Continue to read all the fragments.
          return true;
        };
    if (!iterateWorkspaceFragments(env, gatherer)) {
      return null;
    }
    return builder.build();
  }

  /** Uses a rule name to fetch the corresponding Rule from the external package. */
  @Nullable
  public static Rule getRuleByName(final String ruleName, Environment env)
      throws ExternalPackageException, InterruptedException {

    ExternalPackageRuleExtractor extractor = new ExternalPackageRuleExtractor(env, ruleName);
    if (!iterateWorkspaceFragments(env, extractor)) {
      // Values missing
      return null;
    }

    return extractor.getRule();
  }

  private static RootedPath checkWorkspaceFile(
      Environment env, Root root, PathFragment workspaceFile) throws InterruptedException {
    RootedPath candidate = RootedPath.toRootedPath(root, workspaceFile);
    FileValue fileValue = (FileValue) env.getValue(FileValue.key(candidate));
    if (env.valuesMissing()) {
      return null;
    }

    return fileValue.isFile() ? candidate : null;
  }

  /**
   * Returns the path of the main WORKSPACE file or null when a Skyframe restart is required.
   *
   * <p>Should also return null when the WORKSPACE file is not present, but then some tests break,
   * so then it lies and returns the RootedPath corresponding to the last package path entry.
   */
  @Nullable
  public static RootedPath findWorkspaceFile(Environment env) throws InterruptedException {
    PathPackageLocator packageLocator = PrecomputedValue.PATH_PACKAGE_LOCATOR.get(env);
    ImmutableList<Root> packagePath = packageLocator.getPathEntries();
    for (Root candidateRoot : packagePath) {
      RootedPath path =
          checkWorkspaceFile(
              env, candidateRoot, BuildFileName.WORKSPACE_DOT_BAZEL.getFilenameFragment());
      if (env.valuesMissing()) {
        return null;
      }

      if (path != null) {
        return path;
      }

      path = checkWorkspaceFile(env, candidateRoot, BuildFileName.WORKSPACE.getFilenameFragment());
      if (env.valuesMissing()) {
        return null;
      }

      if (path != null) {
        return path;
      }
    }

    // TODO(lberki): Technically, this means that the WORKSPACE file was not found. I'd love to not
    // have this here, but a lot of tests break without it because they rely on Bazel kinda working
    // even if the WORKSPACE file is not present.
    return RootedPath.toRootedPath(
        Iterables.getLast(packagePath), BuildFileName.WORKSPACE.getFilenameFragment());
  }

  /** Returns false if some SkyValues were missing. */
  private static boolean iterateWorkspaceFragments(
      Environment env, WorkspaceFileValueProcessor processor) throws InterruptedException {
    RootedPath workspacePath = findWorkspaceFile(env);
    if (env.valuesMissing()) {
      return false;
    }

    SkyKey workspaceKey = WorkspaceFileValue.key(workspacePath);
    WorkspaceFileValue value;
    do {
      value = (WorkspaceFileValue) env.getValue(workspaceKey);
      if (value == null) {
        return false;
      }
    } while (processor.processAndShouldContinue(value) && (workspaceKey = value.next()) != null);
    return true;
  }

  private static class ExternalPackageRuleExtractor implements WorkspaceFileValueProcessor {
    private final Environment env;
    private final String ruleName;
    private ExternalPackageException exception;
    private Rule rule;

    private ExternalPackageRuleExtractor(Environment env, String ruleName) {
      this.env = env;
      this.ruleName = ruleName;
    }

    @Override
    public boolean processAndShouldContinue(WorkspaceFileValue workspaceFileValue) {
      Package externalPackage = workspaceFileValue.getPackage();
      if (externalPackage.containsErrors()) {
        Event.replayEventsOn(env.getListener(), externalPackage.getEvents());
        exception =
            new ExternalPackageException(
                new BuildFileContainsErrorsException(
                    LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER,
                    "Could not load //external package"),
                Transience.PERSISTENT);
        // Stop iteration when encountered errors.
        return false;
      }
      rule = externalPackage.getRule(ruleName);
      // Stop if the rule is found = continue while it is null.
      return rule == null;
    }

    public Rule getRule() throws ExternalPackageException {
      if (exception != null) {
        throw exception;
      }
      if (rule == null) {
        throw new ExternalRuleNotFoundException(ruleName);
      }
      return rule;
    }
  }

  private interface WorkspaceFileValueProcessor {
    boolean processAndShouldContinue(WorkspaceFileValue value);
  }
}
