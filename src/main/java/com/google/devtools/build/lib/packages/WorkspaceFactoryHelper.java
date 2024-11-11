// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.base.Verify;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.RuleFactory.BuildLangTypedAttributeValuesMap;
import com.google.devtools.build.lib.packages.TargetRecorder.NameConflictException;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Map;
import java.util.stream.Collectors;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkThread;

/** A helper for the {@link WorkspaceFactory} to create repository rules */
public final class WorkspaceFactoryHelper {

  public static final String DEFAULT_WORKSPACE_SUFFIX_FILE = "/DEFAULT.WORKSPACE.SUFFIX";

  public static boolean originatesInWorkspaceSuffix(
      ImmutableList<StarlarkThread.CallStackEntry> callstack) {
    return callstack.get(0).location.file().equals(DEFAULT_WORKSPACE_SUFFIX_FILE);
  }

  @CanIgnoreReturnValue
  public static Rule createAndAddRepositoryRule(
      Package.Builder pkgBuilder,
      RuleClass ruleClass,
      Map<String, Object> kwargs,
      ImmutableList<StarlarkThread.CallStackEntry> callstack)
      throws RuleFactory.InvalidRuleException,
          NameConflictException,
          LabelSyntaxException,
          InterruptedException {
    BuildLangTypedAttributeValuesMap attributeValues = new BuildLangTypedAttributeValuesMap(kwargs);
    Rule rule = RuleFactory.createRule(pkgBuilder, ruleClass, attributeValues, true, callstack);
    overwriteRule(pkgBuilder, rule);
    return rule;
  }

  /**
   * Updates the map of attributes specified by the user to match the set of attributes declared in
   * the rule definition.
   */
  public static Map<String, Object> getFinalKwargs(Map<String, Object> kwargs) {
    // 'repo_mapping' is not an explicit attribute of any rule and so it would
    // result in a rule error if propagated to the rule factory.
    return kwargs.entrySet().stream()
        .filter(x -> !x.getKey().equals("repo_mapping"))
        .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
  }

  /**
   * Adds an entry in the repo map for {@code externalRepoName} from the main repo name to
   * {@code @}.
   *
   * <p>This is so that labels that refer to the main workspace name (either from the main workspace
   * or from an external repository) in different forms all resolve to the same label.
   *
   * <p>For example, consider a main workspace with the name {@code foo}. The labels
   * {@code @foo//:bar} and {@code //:bar} from the main workspace should resolve to the same thing.
   * Additionally, the labels {@code @foo//:bar} and {@code @//:bar} from an external repository
   * should also evaluate to the same thing.
   */
  public static void addMainRepoEntry(Package.Builder builder, String externalRepoName)
      throws LabelSyntaxException {
    if (!Strings.isNullOrEmpty(builder.getWorkspaceName())) {
      // Create repository names with validation, LabelSyntaxException is thrown is the name
      // is not valid.
      builder.addRepositoryMappingEntry(
          RepositoryName.create(externalRepoName), builder.getWorkspaceName(), RepositoryName.MAIN);
    }
  }

  /**
   * Processes {@code repo_mapping} attribute and populates the package builder with the mappings.
   *
   * @throws EvalException if {@code repo_mapping} is present in kwargs but is not a
   *     string-to-string dict.
   */
  public static void addRepoMappings(
      Package.Builder builder, Map<String, Object> kwargs, String externalRepoName)
      throws EvalException, LabelSyntaxException {
    Object repoMapping = kwargs.get("repo_mapping");
    if (repoMapping != null) {
      for (Map.Entry<String, String> e :
          Dict.cast(repoMapping, String.class, String.class, "repo_mapping").entrySet()) {
        // Create repository names with validation; may throw LabelSyntaxException.
        // For legacy reasons, the repository names given to the repo_mapping attribute need to be
        // prefixed with an @.
        if (!e.getKey().startsWith("@")) {
          throw new LabelSyntaxException(
              "invalid repository name '"
                  + e.getKey()
                  + "': repo names used in the repo_mapping attribute must start with '@'");
        }
        if (!e.getValue().startsWith("@")) {
          throw new LabelSyntaxException(
              "invalid repository name '"
                  + e.getValue()
                  + "': repo names used in the repo_mapping attribute must start with '@'");
        }
        RepositoryName.validateUserProvidedRepoName(e.getKey().substring(1));
        builder.addRepositoryMappingEntry(
            RepositoryName.create(externalRepoName),
            e.getKey().substring(1),
            RepositoryName.create(e.getValue().substring(1)));
      }
    }
  }

  static void addBindRule(
      Package.Builder pkg,
      RuleClass bindRuleClass,
      Label virtual,
      Label actual,
      ImmutableList<StarlarkThread.CallStackEntry> callstack)
      throws RuleFactory.InvalidRuleException, NameConflictException, InterruptedException {

    Map<String, Object> attributes = Maps.newHashMap();
    // Bound rules don't have a name field, but this works because we don't want more than one
    // with the same virtual name.
    attributes.put("name", virtual.getName());
    if (actual != null) {
      attributes.put("actual", actual);
    }
    BuildLangTypedAttributeValuesMap attributeValues =
        new BuildLangTypedAttributeValuesMap(attributes);
    Rule rule = RuleFactory.createRule(pkg, bindRuleClass, attributeValues, true, callstack);
    overwriteRule(pkg, rule);
  }

  private static void overwriteRule(Package.Builder pkg, Rule rule) throws NameConflictException {
    Preconditions.checkArgument(rule.getOutputFiles().isEmpty());
    Target old = pkg.getTarget(rule.getName());
    if (old != null) {
      if (old instanceof Rule) {
        Verify.verify(((Rule) old).getOutputFiles().isEmpty());
      }
      pkg.replaceTarget(rule);
    } else {
      pkg.addRule(rule);
    }
  }

  private WorkspaceFactoryHelper() {}
}
