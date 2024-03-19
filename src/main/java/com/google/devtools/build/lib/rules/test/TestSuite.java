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
package com.google.devtools.build.lib.rules.test;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.test.TestTagsProvider;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.TestTargetUtils;
import com.google.devtools.build.lib.packages.Types;
import com.google.devtools.build.lib.util.Pair;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Implementation for the "test_suite" rule.
 */
public class TestSuite implements RuleConfiguredTargetFactory {

  @Override
  @Nullable
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    checkTestsAndSuites(ruleContext, "tests");
    if (ruleContext.hasErrors()) {
      return null;
    }

    //
    //  CAUTION!  Keep this logic consistent with lib.query2.TestsFunction!
    //

    List<String> tagsAttribute =
        new ArrayList<>(ruleContext.attributes().get("tags", Types.STRING_LIST));
    // TODO(ulfjack): This is inconsistent with the other places that do test_suite expansion.
    tagsAttribute.remove("manual");
    Pair<Collection<String>, Collection<String>> requiredExcluded =
        TestTargetUtils.sortTagsBySense(tagsAttribute);

    List<TransitiveInfoCollection> directTestsAndSuitesBuilder = new ArrayList<>();

    // The set of implicit tests is determined in
    // {@link com.google.devtools.build.lib.packages.Package}.
    // Manual tests are already filtered out there. That is what $implicit_tests is about.
    for (TransitiveInfoCollection dep :
          Iterables.concat(
              getPrerequisites(ruleContext, "tests"),
              getPrerequisites(ruleContext, "$implicit_tests"))) {
      if (dep.getProvider(TestTagsProvider.class) != null) {
        // getTestTags maps to Rule.getRuleTags.
        List<String> tags = dep.getProvider(TestTagsProvider.class).getTestTags();
        if (!TestTargetUtils.testMatchesFilters(
            tags, requiredExcluded.first, requiredExcluded.second)) {
          // This test does not match our filter. Ignore it.
          continue;
        }
      }
      directTestsAndSuitesBuilder.add(dep);
    }

    Runfiles runfiles =
        new Runfiles.Builder(
                ruleContext.getWorkspaceName(),
                ruleContext.getConfiguration().legacyExternalRunfiles())
            .addTargets(
                directTestsAndSuitesBuilder,
                RunfilesProvider.DATA_RUNFILES,
                ruleContext.getConfiguration().alwaysIncludeFilesToBuildInData())
            .build();

    return new RuleConfiguredTargetBuilder(ruleContext)
        .add(RunfilesProvider.class,
            RunfilesProvider.withData(Runfiles.EMPTY, runfiles))
        .add(TransitiveTestsProvider.class, new TransitiveTestsProvider())
        .build();
  }

  private Iterable<? extends TransitiveInfoCollection> getPrerequisites(
      RuleContext ruleContext, String attributeName) {
    if (ruleContext.attributes().has(attributeName, BuildType.LABEL_LIST)) {
      return ruleContext.getPrerequisites(attributeName);
    } else {
      return ImmutableList.<TransitiveInfoCollection>of();
    }
  }

  private void checkTestsAndSuites(RuleContext ruleContext, String attributeName) {
    if (!ruleContext.attributes().has(attributeName, BuildType.LABEL_LIST)) {
      return;
    }
    for (TransitiveInfoCollection dep : ruleContext.getPrerequisites(attributeName)) {
      // TODO(bazel-team): Maybe convert the TransitiveTestsProvider into an inner interface.
      TransitiveTestsProvider provider = dep.getProvider(TransitiveTestsProvider.class);
      TestTagsProvider tagsProvider = dep.getProvider(TestTagsProvider.class);
      if (provider == null && tagsProvider == null) {
        ruleContext.attributeError(attributeName,
            "expecting a test or a test_suite rule but '" + dep.getLabel() + "' is not one");
      }
    }
  }
}
