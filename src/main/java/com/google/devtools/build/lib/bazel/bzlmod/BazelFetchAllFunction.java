// Copyright 2023 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.bazel.repository.starlark.StarlarkRepositoryFunction;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Void function designed to gather and fetch all the repositories without returning any specific
 * result (empty value is returned).
 */
public class BazelFetchAllFunction implements SkyFunction {

  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {

    // Collect all the repos we want to fetch here
    List<RepositoryName> reposToFetch = new ArrayList<>();

    // 1. Run resolution and collect the dependency graph repos except for main
    BazelDepGraphValue depGraphValue = (BazelDepGraphValue) env.getValue(BazelDepGraphValue.KEY);
    if (depGraphValue == null) {
      return null;
    }
    reposToFetch.addAll(
        depGraphValue.getCanonicalRepoNameLookup().keySet().stream()
            .filter(repo -> !repo.isMain())
            .collect(toImmutableList()));

    // 2. Run every extension found in the modules
    ImmutableSet<ModuleExtensionId> extensionIds =
        depGraphValue.getExtensionUsagesTable().rowKeySet();
    ImmutableSet<SkyKey> singleEvalKeys =
        extensionIds.stream().map(SingleExtensionEvalValue::key).collect(toImmutableSet());
    SkyframeLookupResult singleEvalValues = env.getValuesAndExceptions(singleEvalKeys);

    // 3. For each extension, collect its generated repos
    for (SkyKey singleEvalKey : singleEvalKeys) {
      SingleExtensionEvalValue singleEvalValue =
          (SingleExtensionEvalValue) singleEvalValues.get(singleEvalKey);
      if (singleEvalValue == null) {
        return null;
      }
      reposToFetch.addAll(singleEvalValue.getCanonicalRepoNameToInternalNames().keySet());
    }

    // 4. If this is fetch configure, get repo rules and only collect repos marked as configure
    Boolean fetchConfigure = (Boolean) skyKey.argument();
    if (fetchConfigure) {
      ImmutableSet<SkyKey> repoRuleKeys =
          reposToFetch.stream().map(BzlmodRepoRuleValue::key).collect(toImmutableSet());
      reposToFetch.clear(); // empty this list to only add configured repos
      SkyframeLookupResult repoRuleValues = env.getValuesAndExceptions(repoRuleKeys);
      for (SkyKey repoRuleKey : repoRuleKeys) {
        BzlmodRepoRuleValue repoRuleValue = (BzlmodRepoRuleValue) repoRuleValues.get(repoRuleKey);
        if (repoRuleValue == null) {
          return null;
        }
        if (StarlarkRepositoryFunction.isConfigureRule(repoRuleValue.getRule())) {
          reposToFetch.add(RepositoryName.createUnvalidated(repoRuleValue.getRule().getName()));
        }
      }
    }

    // 5. Fetch all the collected repos
    ImmutableSet<SkyKey> repoDelegatorKeys =
        reposToFetch.stream().map(RepositoryDirectoryValue::key).collect(toImmutableSet());
    SkyframeLookupResult repoDirValues = env.getValuesAndExceptions(repoDelegatorKeys);
    for (SkyKey repoDelegatorKey : repoDelegatorKeys) {
      RepositoryDirectoryValue repoDirValue =
          (RepositoryDirectoryValue) repoDirValues.get(repoDelegatorKey);
      if (repoDirValue == null) {
        return null;
      }
    }

    return BazelFetchAllValue.create();
  }

}
