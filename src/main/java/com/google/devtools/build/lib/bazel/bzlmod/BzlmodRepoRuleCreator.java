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

import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleFactory.InvalidRuleException;
import java.util.Map;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * An interface for {@link RepositoryRuleFunction} to create a repository rule instance with given
 * parameters.
 */
public interface BzlmodRepoRuleCreator {
  Rule createAndAddRule(
      Package.Builder packageBuilder,
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
        ImmutableList.of(new CallStackEntry(callStackEntry, Location.BUILTIN));
    Rule rule;
    try {
      rule =
          RuleFactory.createAndAddRule(
              packageBuilder, ruleClass, attributeValues, eventHandler, semantics, callStack);
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
