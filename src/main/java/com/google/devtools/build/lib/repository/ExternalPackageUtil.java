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

import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.skyframe.PackageLookupValue;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/** Utility class to centralize looking up data from the external package. */
public class ExternalPackageUtil {
  /**
   * Returns directories, that should not be symlinked under the execroot.
   *
   * Searches for dont_symlink_directories_in_execroot calls in the WORKSPACE file,
   * and gathers values of all "paths" attributes. */
  public static ImmutableSortedSet<String> getNotSymlinkedInExecrootDirectories(
      SkyValueComputer env) throws InterruptedException {
    ImmutableSortedSet.Builder<String> builder = ImmutableSortedSet.naturalOrder();
    Predicate<WorkspaceFileValue> gatherer = workspaceFileValue -> {
      ImmutableSortedSet<String> paths = workspaceFileValue.getDoNotSymlinkInExecrootPaths();
      if (paths != null) {
        builder.addAll(paths);
      }
      // Continue to read all the fragments.
      return false;
    };
    if (! iterateWorkspaceFragments(env, gatherer)) {
      return null;
    }
    return builder.build();
  }

  /** Uses a rule name to fetch the corresponding Rule from the external package. */
  @Nullable
  public static Rule getRuleByName(final String ruleName, Environment env)
      throws ExternalPackageException, InterruptedException {

    ExternalPackageRuleExtractor extractor = new ExternalPackageRuleExtractor(env, ruleName);
    if (!iterateWorkspaceFragments(env::getValue, extractor)) {
      // Values missing
      return null;
    }

    return extractor.getRule();
  }

  private static boolean iterateWorkspaceFragments(
      SkyValueComputer env,
      Predicate<WorkspaceFileValue> consumer) throws InterruptedException {
    SkyKey packageLookupKey = PackageLookupValue.key(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER);
    PackageLookupValue packageLookupValue = (PackageLookupValue) env.getValue(packageLookupKey);
    if (packageLookupValue == null) {
      return false;
    }
    RootedPath workspacePath =
        packageLookupValue.getRootedPath(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER);

    SkyKey workspaceKey = WorkspaceFileValue.key(workspacePath);
    do {
      WorkspaceFileValue value = (WorkspaceFileValue) env.getValue(workspaceKey);
      if (value == null) {
        return false;
      }
      if (!consumer.test(value)) {
        return true;
      }
      workspaceKey = value.next();
    } while (workspaceKey != null);
    return true;
  }

  private static class ExternalPackageRuleExtractor implements Predicate<WorkspaceFileValue> {
    private final Environment env;
    private final String ruleName;
    private ExternalPackageException exception;
    private Rule rule;

    private ExternalPackageRuleExtractor(Environment env, String ruleName) {
      this.env = env;
      this.ruleName = ruleName;
    }

    @Override
    public boolean test(WorkspaceFileValue workspaceFileValue) {
      Package externalPackage = workspaceFileValue.getPackage();
      if (externalPackage.containsErrors()) {
        Event.replayEventsOn(env.getListener(), externalPackage.getEvents());
        exception = new ExternalPackageException(
            new BuildFileContainsErrorsException(
                LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER,
                "Could not load //external package"),
            Transience.PERSISTENT);
      }
      rule = externalPackage.getRule(ruleName);
      return rule != null;
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

  public interface SkyValueComputer {
    @Nullable SkyValue getValue(SkyKey key) throws InterruptedException;
  }
}
