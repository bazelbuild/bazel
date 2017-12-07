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
package com.google.devtools.build.lib.skyframe.packages;

import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.rules.BazelRuleClassProvider;
import com.google.devtools.build.lib.packages.BuildFileName;
import com.google.devtools.build.lib.packages.PackageFactory.EnvironmentExtension;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryFunction;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryRule;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryLoaderFunction;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.skyframe.LocalRepositoryLookupFunction;
import com.google.devtools.build.lib.skyframe.PackageFunction.ActionOnIOExceptionReadingBuildFile;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction.CrossRepositoryLabelViolationStrategy;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Concrete implementation of {@link PackageLoader} that uses skyframe under the covers, but with
 * no caching or incrementality.
 */
public class BazelPackageLoader extends AbstractPackageLoader {
  /** Returns a fresh {@link Builder} instance. */
  public static Builder builder(Path workspaceDir) {
    Builder builder = new Builder(workspaceDir);

    // Set up SkyFunctions and PrecomputedValues needed to make local repositories work correctly.
    ImmutableMap<String, RepositoryFunction> repositoryHandlers =
        ImmutableMap.of(
            LocalRepositoryRule.NAME, (RepositoryFunction) new LocalRepositoryFunction());

    builder.addExtraSkyFunctions(
        ImmutableMap.<SkyFunctionName, SkyFunction>of(
            SkyFunctions.LOCAL_REPOSITORY_LOOKUP,
            new LocalRepositoryLookupFunction(),
            SkyFunctions.REPOSITORY_DIRECTORY,
            new RepositoryDelegatorFunction(
                repositoryHandlers,
                null,
                new AtomicBoolean(true),
                ImmutableMap::of,
                builder.directories),
            SkyFunctions.REPOSITORY,
            new RepositoryLoaderFunction()));

    // Set extra precomputed values.
    builder.addExtraPrecomputedValues(
        PrecomputedValue.injected(
            RepositoryDelegatorFunction.REPOSITORY_OVERRIDES,
            Suppliers.ofInstance(ImmutableMap.of())));

    return builder;
  }

  /** Builder for {@link BazelPackageLoader} instances. */
  public static class Builder extends AbstractPackageLoader.Builder {
    private Builder(Path workspaceDir) {
      super(workspaceDir);
    }

    @Override
    public BazelPackageLoader buildImpl() {
      return new BazelPackageLoader(this);
    }

    @Override
    protected RuleClassProvider getDefaultRuleClassProvider() {
      return BazelRuleClassProvider.create();
    }

    @Override
    protected String getDefaultDefaultPackageContents() {
      return BazelRuleClassProvider.create().getDefaultsPackageContent(
          InvocationPolicy.getDefaultInstance());
    }
  }

  private BazelPackageLoader(Builder builder) {
    super(builder);
  }

  @Override
  protected String getName() {
    return "BazelPackageLoader";
  }

  @Override
  protected ImmutableList<EnvironmentExtension> getEnvironmentExtensions() {
    return ImmutableList.of();
  }

  @Override
  protected CrossRepositoryLabelViolationStrategy getCrossRepositoryLabelViolationStrategy() {
    return BazelSkyframeExecutorConstants.CROSS_REPOSITORY_LABEL_VIOLATION_STRATEGY;
  }

  @Override
  protected ImmutableList<BuildFileName> getBuildFilesByPriority() {
    return BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY;
  }

  @Override
  protected ActionOnIOExceptionReadingBuildFile getActionOnIOExceptionReadingBuildFile() {
    return BazelSkyframeExecutorConstants.ACTION_ON_IO_EXCEPTION_READING_BUILD_FILE;
  }
}
