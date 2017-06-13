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

package com.google.devtools.build.lib.rules;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skyframe.PackageLookupValue;
import com.google.devtools.build.lib.skyframe.WorkspaceFileValue;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import javax.annotation.Nullable;

/** Utility class to centralize looking up rules from the external package. */
public class ExternalPackageUtil {

  /** Uses a rule name to fetch the corresponding Rule from the external package. */
  @Nullable
  public static Rule getRule(String ruleName, Environment env)
      throws ExternalPackageException, InterruptedException {

    SkyKey packageLookupKey = PackageLookupValue.key(Label.EXTERNAL_PACKAGE_IDENTIFIER);
    PackageLookupValue packageLookupValue = (PackageLookupValue) env.getValue(packageLookupKey);
    if (packageLookupValue == null) {
      return null;
    }
    RootedPath workspacePath = packageLookupValue.getRootedPath(Label.EXTERNAL_PACKAGE_IDENTIFIER);

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
                Label.EXTERNAL_PACKAGE_IDENTIFIER, "Could not load //external package"),
            Transience.PERSISTENT);
      }
      Rule rule = externalPackage.getRule(ruleName);
      if (rule != null) {
        return rule;
      }
      workspaceKey = value.next();
    } while (workspaceKey != null);
    throw new ExternalRuleNotFoundException(ruleName);
  }

  @Nullable
  public static Rule getRule(String ruleName, @Nullable String ruleClassName, Environment env)
      throws ExternalPackageException, InterruptedException {
    try {
      return getRule(RepositoryName.create("@" + ruleName), ruleClassName, env);
    } catch (LabelSyntaxException e) {
      throw new ExternalPackageException(
          new IOException("Invalid rule name " + ruleName), Transience.PERSISTENT);
    }
  }

  /**
   * Uses a remote repository name to fetch the corresponding Rule describing how to get it. This
   * should be called from {@link SkyFunction#compute} functions, which should return null if this
   * returns null. If {@code ruleClassName} is set, the rule found must have a matching rule class
   * name.
   */
  @Nullable
  public static Rule getRule(
      RepositoryName repositoryName, @Nullable String ruleClassName, Environment env)
      throws ExternalPackageException, InterruptedException {
    Rule rule = getRule(repositoryName.strippedName(), env);
    Preconditions.checkState(
        rule == null || ruleClassName == null || rule.getRuleClass().equals(ruleClassName),
        "Got %s, was expecting a %s",
        rule,
        ruleClassName);
    return rule;
  }

  /** Exception thrown when something goes wrong accessing a rule. */
  public static class ExternalPackageException extends SkyFunctionException {
    public ExternalPackageException(NoSuchPackageException cause, Transience transience) {
      super(cause, transience);
    }

    /** Error reading or writing to the filesystem. */
    public ExternalPackageException(IOException cause, Transience transience) {
      super(cause, transience);
    }

    /** For errors in WORKSPACE file rules (e.g., malformed paths or URLs). */
    public ExternalPackageException(EvalException cause, Transience transience) {
      super(cause, transience);
    }
  }

  /** Exception thrown when a rule cannot be found. */
  public static final class ExternalRuleNotFoundException extends ExternalPackageException {
    public ExternalRuleNotFoundException(String ruleName) {
      super(
          new BuildFileContainsErrorsException(
              Label.EXTERNAL_PACKAGE_IDENTIFIER,
              "The rule named '" + ruleName + "' could not be resolved"),
          Transience.PERSISTENT);
    }
  }
}
