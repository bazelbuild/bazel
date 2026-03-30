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

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static java.util.stream.Collectors.toCollection;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.bazel.repository.RepoDefinition;
import com.google.devtools.build.lib.bazel.repository.RepoDefinitionValue;
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
 * Gather and fetch all the repositories from MODULE.bazel resolution and extensions evaluation. If
 * this is fetch configure, only configure repos will be fetched and returned
 */
public class BazelFetchAllFunction implements SkyFunction {

  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {

    // Collect all the repos we want to fetch here
    List<RepositoryName> reposToFetch;

    // 1. Run resolution and collect the dependency graph repos except for main
    BazelDepGraphValue depGraphValue = (BazelDepGraphValue) env.getValue(BazelDepGraphValue.KEY);
    if (depGraphValue == null) {
      return null;
    }
    reposToFetch =
        depGraphValue.getCanonicalRepoNameLookup().keySet().stream()
            .filter(repo -> !repo.isMain())
            .collect(toCollection(ArrayList::new));

    // 2. Run every extension found in the modules & collect its generated repos
    ImmutableSet<ModuleExtensionId> extensionIds =
        depGraphValue.getExtensionUsagesTable().rowKeySet();
    ImmutableSet<SkyKey> singleExtensionKeys =
        extensionIds.stream().map(SingleExtensionValue::key).collect(toImmutableSet());
    SkyframeLookupResult singleExtensionValues = env.getValuesAndExceptions(singleExtensionKeys);
    for (SkyKey singleExtensionKey : singleExtensionKeys) {
      SingleExtensionValue singleExtensionValue =
          (SingleExtensionValue) singleExtensionValues.get(singleExtensionKey);
      if (singleExtensionValue == null) {
        return null;
      }
      reposToFetch.addAll(singleExtensionValue.canonicalRepoNameToInternalNames().keySet());
    }

    // 3. If this is fetch configure, get repo rules and only collect repos marked as configure
    Boolean fetchConfigure = (Boolean) skyKey.argument();
    if (fetchConfigure) {
      ImmutableSet<SkyKey> repoDefinitionKeys =
          reposToFetch.stream().map(RepoDefinitionValue::key).collect(toImmutableSet());
      reposToFetch.clear(); // empty this list to only add configured repos
      SkyframeLookupResult repoDefinitionValues = env.getValuesAndExceptions(repoDefinitionKeys);
      for (SkyKey repoRuleKey : repoDefinitionKeys) {
        RepoDefinitionValue repoDefinitionValue =
            (RepoDefinitionValue) repoDefinitionValues.get(repoRuleKey);
        if (repoDefinitionValue == null) {
          return null;
        }
        if (repoDefinitionValue instanceof RepoDefinitionValue.Found(RepoDefinition repoDefinition)
            && repoDefinition.repoRule().configure()) {
          reposToFetch.add((RepositoryName) repoRuleKey.argument());
        }
      }
    }

    // 4. Fetch all the collected repos
    List<RepositoryName> shouldVendor = new ArrayList<>();
    ImmutableSet<SkyKey> repoDelegatorKeys =
        reposToFetch.stream().map(RepositoryDirectoryValue::key).collect(toImmutableSet());
    SkyframeLookupResult repoDirValues = env.getValuesAndExceptions(repoDelegatorKeys);
    for (SkyKey repoDelegatorKey : repoDelegatorKeys) {
      RepositoryDirectoryValue repoDirValue =
          (RepositoryDirectoryValue) repoDirValues.get(repoDelegatorKey);
      if (repoDirValue == null) {
        return null;
      }
      if (repoDirValue instanceof RepositoryDirectoryValue.Success s && !s.excludeFromVendoring()) {
        shouldVendor.add((RepositoryName) repoDelegatorKey.argument());
      }
    }

    return BazelFetchAllValue.create(ImmutableList.copyOf(shouldVendor));
  }
}
