// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleFactory;
import com.google.devtools.build.lib.packages.RuleFactory.BuildLangTypedAttributeValuesMap;
import com.google.devtools.build.lib.packages.RuleFactory.InvalidRuleException;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import java.util.Map;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkThread.CallStackEntry;
import net.starlark.java.syntax.Location;

/**
 * Creates a repo rule instance for Bzlmod. This class contrasts with the WORKSPACE repo rule
 * creation mechanism in that it creates an "external" package that contains only 1 rule.
 */
public final class BzlmodRepoRuleCreator {
  private BzlmodRepoRuleCreator() {}

  /** Creates a repo rule instance from the given parameters. */
  public static Rule createRule(
      PackageIdentifier basePackageId,
      RepositoryMapping repoMapping,
      BlazeDirectories directories,
      StarlarkSemantics semantics,
      ExtendedEventHandler eventHandler,
      String callStackEntry,
      RuleClass ruleClass,
      Map<String, Object> attributes)
      throws InterruptedException, InvalidRuleException, NoSuchPackageException, EvalException {
    // TODO(bazel-team): Don't use the {@link Rule} class for repository rule.
    // Currently, the repository rule is represented with the {@link Rule} class that's designed
    // for build rules. Therefore, we have to create a package instance for it, which doesn't make
    // sense. We should migrate away from this implementation so that we don't refer to any build
    // rule specific things in repository rule.
    Package.Builder packageBuilder =
        Package.newExternalPackageBuilderForBzlmod(
            RootedPath.toRootedPath(
                Root.fromPath(directories.getWorkspace()),
                LabelConstants.MODULE_DOT_BAZEL_FILE_NAME),
            semantics,
            basePackageId,
            repoMapping);
    BuildLangTypedAttributeValuesMap attributeValues =
        new BuildLangTypedAttributeValuesMap(attributes);
    ImmutableList<CallStackEntry> callStack =
        ImmutableList.of(StarlarkThread.callStackEntry(callStackEntry, Location.BUILTIN));
    Rule rule;
    try {
      rule =
          RuleFactory.createAndAddRule(
              packageBuilder, ruleClass, attributeValues, eventHandler, callStack);
    } catch (NameConflictException e) {
      // This literally cannot happen -- we just created the package!
      throw new IllegalStateException(e);
    }
    if (rule.containsErrors()) {
      throw Starlark.errorf(
          "failed to instantiate '%s' from this module extension", ruleClass.getName());
    }
    packageBuilder.build();
    return rule;
  }
}
