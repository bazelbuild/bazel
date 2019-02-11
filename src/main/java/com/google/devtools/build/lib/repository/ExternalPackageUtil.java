// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skyframe.PackageLookupValue;
import com.google.devtools.build.lib.skyframe.WorkspaceFileValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** Utility class to centralize looking up rules from the external package. */
public class ExternalPackageUtil {

  /**
   * Loads the external package and then calls the selector to find matching rules.
   *
   * @param env the environment to use for lookups
   * @param returnFirst whether to return only the first rule found
   * @param selector the function to call to load rules
   */
  @Nullable
  private static List<Rule> getRules(
      Environment env, boolean returnFirst, Function<Package, List<Rule>> selector)
      throws ExternalPackageException, InterruptedException {
    RootedPath workspacePath = getWorkspacePath(env);
    if (env.valuesMissing()) {
      return null;
    }

    List<Rule> rules = returnFirst ? ImmutableList.of() : Lists.newArrayList();
    SkyKey workspaceKey = WorkspaceFileValue.key(workspacePath);
    do {
      WorkspaceFileValue value = (WorkspaceFileValue) env.getValue(workspaceKey);
      if (value == null) {
        return null;
      }
      Package externalPackage = value.getPackage();
      if (externalPackage.containsErrors()) {
        Event.replayEventsOn(env.getListener(), externalPackage.getEvents());
        throw new ExternalPackageException(
            new BuildFileContainsErrorsException(
                LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER, "Could not load //external package"),
            Transience.PERSISTENT);
      }
      List<Rule> results = selector.apply(externalPackage);
      if (results != null && !results.isEmpty()) {
        if (returnFirst) {
          // assert expected non null value explicitly for possible future callers
          return ImmutableList.of(Preconditions.checkNotNull(results.get(0)));
        }
        rules.addAll(results);
      }
      workspaceKey = value.next();
    } while (workspaceKey != null);

    return rules;
  }

  @Nullable
  public static RootedPath getWorkspacePath(final Environment env) throws InterruptedException {
    SkyKey packageLookupKey = PackageLookupValue.key(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER);
    PackageLookupValue packageLookupValue = (PackageLookupValue) env.getValue(packageLookupKey);
    if (packageLookupValue == null) {
      return null;
    }
    return packageLookupValue.getRootedPath(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER);
  }

  /** Uses a rule name to fetch the corresponding Rule from the external package. */
  @Nullable
  public static Rule getRuleByName(final String ruleName, Environment env)
      throws ExternalPackageException, InterruptedException {

    List<Rule> rules =
        getRules(
            env,
            true,
            new Function<Package, List<Rule>>() {
              @Nullable
              @Override
              public List<Rule> apply(Package externalPackage) {
                Rule rule = externalPackage.getRule(ruleName);
                if (rule == null) {
                  return null;
                }
                return ImmutableList.of(rule);
              }
            });

    if (env.valuesMissing()) {
      return null;
    }
    if (rules == null || rules.isEmpty()) {
      throw new ExternalRuleNotFoundException(ruleName);
    }
    return Iterables.getFirst(rules, null);
  }

}
