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
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.bazel.BazelRepositoryModule;
import com.google.devtools.build.lib.bazel.repository.MavenDownloader;
import com.google.devtools.build.lib.bazel.repository.MavenServerFunction;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache;
import com.google.devtools.build.lib.bazel.repository.downloader.HttpDownloader;
import com.google.devtools.build.lib.bazel.repository.skylark.SkylarkRepositoryFunction;
import com.google.devtools.build.lib.bazel.rules.BazelRulesModule;
import com.google.devtools.build.lib.packages.BuildFileName;
import com.google.devtools.build.lib.packages.PackageFactory.EnvironmentExtension;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryLoaderFunction;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.skyframe.ActionEnvironmentFunction;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.skyframe.ClientEnvironmentFunction;
import com.google.devtools.build.lib.skyframe.DirectoryListingFunction;
import com.google.devtools.build.lib.skyframe.DirectoryListingStateFunction;
import com.google.devtools.build.lib.skyframe.LocalRepositoryLookupFunction;
import com.google.devtools.build.lib.skyframe.PackageFunction.ActionOnIOExceptionReadingBuildFile;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction.CrossRepositoryLabelViolationStrategy;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Concrete implementation of {@link PackageLoader} that uses skyframe under the covers, but with no
 * caching or incrementality.
 */
public class BazelPackageLoader extends AbstractPackageLoader {

  /**
   * Version is the string BazelPackageLoader reports in native.bazel_version to be used by Skylark.
   */
  private final String version;

  /** Returns a fresh {@link Builder} instance. */
  public static Builder builder(Path workspaceDir, Path installBase, Path outputBase) {
    // Prevent PackageLoader from fetching any remote repositories; these should only be fetched by
    // Bazel before calling PackageLoader.
    AtomicBoolean isFetch = new AtomicBoolean(false);

    Builder builder = new Builder(workspaceDir, installBase, outputBase, isFetch);

    RepositoryCache repositoryCache = new RepositoryCache();
    HttpDownloader httpDownloader = new HttpDownloader(repositoryCache);

    // Set up SkyFunctions and PrecomputedValues needed to make local repositories work correctly.
    ImmutableMap<String, RepositoryFunction> repositoryHandlers =
        BazelRepositoryModule.repositoryRules(httpDownloader, new MavenDownloader(repositoryCache));

    builder.addExtraSkyFunctions(
        ImmutableMap.<SkyFunctionName, SkyFunction>builder()
            .put(
                SkyFunctions.CLIENT_ENVIRONMENT_VARIABLE,
                new ClientEnvironmentFunction(new AtomicReference<>(ImmutableMap.of())))
            .put(
                SkyFunctions.DIRECTORY_LISTING_STATE,
                new DirectoryListingStateFunction(builder.externalFilesHelper))
            .put(SkyFunctions.ACTION_ENVIRONMENT_VARIABLE, new ActionEnvironmentFunction())
            .put(SkyFunctions.DIRECTORY_LISTING, new DirectoryListingFunction())
            .put(SkyFunctions.LOCAL_REPOSITORY_LOOKUP, new LocalRepositoryLookupFunction())
            .put(
                SkyFunctions.REPOSITORY_DIRECTORY,
                new RepositoryDelegatorFunction(
                    repositoryHandlers,
                    new SkylarkRepositoryFunction(httpDownloader),
                    isFetch,
                    ImmutableMap::of,
                    builder.directories))
            .put(SkyFunctions.REPOSITORY, new RepositoryLoaderFunction())
            .put(MavenServerFunction.NAME, new MavenServerFunction(builder.directories))
            .build());

    // Set extra precomputed values.
    builder.addExtraPrecomputedValues(
        PrecomputedValue.injected(PrecomputedValue.ACTION_ENV, ImmutableMap.of()),
        PrecomputedValue.injected(
            RepositoryDelegatorFunction.REPOSITORY_OVERRIDES,
            Suppliers.ofInstance(ImmutableMap.of())));

    return builder;
  }

  /** Builder for {@link BazelPackageLoader} instances. */
  public static class Builder extends AbstractPackageLoader.Builder {
    private static final ConfiguredRuleClassProvider DEFAULT_RULE_CLASS_PROVIDER =
        createRuleClassProvider();

    private final AtomicBoolean isFetch;

    private String version = "";

    private static ConfiguredRuleClassProvider createRuleClassProvider() {
      ConfiguredRuleClassProvider.Builder classProvider = new ConfiguredRuleClassProvider.Builder();
      new BazelRepositoryModule().initializeRuleClasses(classProvider);
      new BazelRulesModule().initializeRuleClasses(classProvider);
      return classProvider.build();
    }

    private Builder(Path workspaceDir, Path installBase, Path outputBase, AtomicBoolean isFetch) {
      super(workspaceDir, installBase, outputBase);
      this.isFetch = isFetch;
    }

    @Override
    public BazelPackageLoader buildImpl() {
      return new BazelPackageLoader(this, version);
    }

    @Override
    protected RuleClassProvider getDefaultRuleClassProvider() {
      return DEFAULT_RULE_CLASS_PROVIDER;
    }

    @Override
    protected String getDefaultDefaultPackageContents() {
      return DEFAULT_RULE_CLASS_PROVIDER.getDefaultsPackageContent(
          InvocationPolicy.getDefaultInstance());
    }

    /**
     * Version is the string BazelPackageLoader reports in native.bazel_version to be used by
     * Skylark.
     */
    public Builder setVersion(String version) {
      this.version = version;
      return this;
    }

    Builder setFetchForTesting() {
      this.isFetch.set(true);
      return this;
    }
  }

  private BazelPackageLoader(Builder builder, String version) {
    super(builder);
    this.version = version;
  }

  @Override
  protected String getVersion() {
    return version;
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
