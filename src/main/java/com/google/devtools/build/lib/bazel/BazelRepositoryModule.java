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

package com.google.devtools.build.lib.bazel;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.bazel.commands.FetchCommand;
import com.google.devtools.build.lib.bazel.repository.GitRepositoryFunction;
import com.google.devtools.build.lib.bazel.repository.HttpArchiveFunction;
import com.google.devtools.build.lib.bazel.repository.HttpFileFunction;
import com.google.devtools.build.lib.bazel.repository.HttpJarFunction;
import com.google.devtools.build.lib.bazel.repository.MavenJarFunction;
import com.google.devtools.build.lib.bazel.repository.MavenServerFunction;
import com.google.devtools.build.lib.bazel.repository.MavenServerRepositoryFunction;
import com.google.devtools.build.lib.bazel.repository.NewGitRepositoryFunction;
import com.google.devtools.build.lib.bazel.repository.NewHttpArchiveFunction;
import com.google.devtools.build.lib.bazel.rules.android.AndroidNdkRepositoryFunction;
import com.google.devtools.build.lib.bazel.rules.android.AndroidNdkRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.android.AndroidSdkRepositoryFunction;
import com.google.devtools.build.lib.bazel.rules.android.AndroidSdkRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.workspace.GitRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.workspace.HttpArchiveRule;
import com.google.devtools.build.lib.bazel.rules.workspace.HttpFileRule;
import com.google.devtools.build.lib.bazel.rules.workspace.HttpJarRule;
import com.google.devtools.build.lib.bazel.rules.workspace.MavenJarRule;
import com.google.devtools.build.lib.bazel.rules.workspace.MavenServerRule;
import com.google.devtools.build.lib.bazel.rules.workspace.NewGitRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.workspace.NewHttpArchiveRule;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryFunction;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryRule;
import com.google.devtools.build.lib.rules.repository.NewLocalRepositoryFunction;
import com.google.devtools.build.lib.rules.repository.NewLocalRepositoryRule;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryLoaderFunction;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.SkyValueDirtinessChecker;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.common.options.OptionsProvider;

import java.util.Map.Entry;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;

import javax.annotation.Nullable;

/**
 * Adds support for fetching external code.
 */
public class BazelRepositoryModule extends BlazeModule {

  // A map of repository handlers that can be looked up by rule class name.
  private final ImmutableMap<String, RepositoryFunction> repositoryHandlers;
  private final AtomicBoolean isFetch = new AtomicBoolean(false);

  public BazelRepositoryModule() {
    repositoryHandlers =
        ImmutableMap.<String, RepositoryFunction>builder()
            .put(LocalRepositoryRule.NAME, new LocalRepositoryFunction())
            .put(HttpArchiveRule.NAME, new HttpArchiveFunction())
            .put(GitRepositoryRule.NAME, new GitRepositoryFunction())
            .put(HttpJarRule.NAME, new HttpJarFunction())
            .put(HttpFileRule.NAME, new HttpFileFunction())
            .put(MavenJarRule.NAME, new MavenJarFunction())
            .put(NewHttpArchiveRule.NAME, new NewHttpArchiveFunction())
            .put(NewGitRepositoryRule.NAME, new NewGitRepositoryFunction())
            .put(NewLocalRepositoryRule.NAME, new NewLocalRepositoryFunction())
            .put(AndroidSdkRepositoryRule.NAME, new AndroidSdkRepositoryFunction())
            .put(AndroidNdkRepositoryRule.NAME, new AndroidNdkRepositoryFunction())
            .put(MavenServerRule.NAME, new MavenServerRepositoryFunction())
            .build();
  }

  @Override
  public void blazeStartup(OptionsProvider startupOptions,
      BlazeVersionInfo versionInfo, UUID instanceId, BlazeDirectories directories,
      Clock clock) {
    for (RepositoryFunction handler : repositoryHandlers.values()) {
      handler.setDirectories(directories);
    }
  }

  /**
   * A dirtiness checker that always dirties {@link RepositoryDirectoryValue}s so that if they were
   * produced in a {@code --nofetch} build, they are re-created no subsequent {@code --fetch}
   * builds.
   *
   * <p>The alternative solution would be to reify the value of the flag as a Skyframe value.
   */
  private static final SkyValueDirtinessChecker REPOSITORY_VALUE_CHECKER =
      new SkyValueDirtinessChecker() {
        @Override
        public boolean applies(SkyKey skyKey) {
          return skyKey.functionName().equals(SkyFunctions.REPOSITORY_DIRECTORY);
        }

        @Override
        public SkyValue createNewValue(SkyKey key, @Nullable TimestampGranularityMonitor tsgm) {
          throw new UnsupportedOperationException();
        }

        @Override
        public DirtyResult check(
            SkyKey skyKey, SkyValue skyValue, @Nullable TimestampGranularityMonitor tsgm) {
          RepositoryDirectoryValue repositoryValue = (RepositoryDirectoryValue) skyValue;
          return repositoryValue.isFetchingDelayed()
              ? DirtyResult.dirty(skyValue)
              : DirtyResult.notDirty(skyValue);
        }
      };

  @Override
  public Iterable<SkyValueDirtinessChecker> getCustomDirtinessCheckers() {
    return ImmutableList.of(REPOSITORY_VALUE_CHECKER);
  }

  @Override
  public void initializeRuleClasses(ConfiguredRuleClassProvider.Builder builder) {
    for (Entry<String, RepositoryFunction> handler : repositoryHandlers.entrySet()) {
      // TODO(bazel-team): Migrate away from Class<?>
      RuleDefinition ruleDefinition;
      try {
        ruleDefinition = handler.getValue().getRuleDefinition().newInstance();
      } catch (IllegalAccessException | InstantiationException e) {
        throw new IllegalStateException(e);
      }
      builder.addRuleDefinition(ruleDefinition);
    }
  }

  @Override
  public Iterable<? extends BlazeCommand> getCommands() {
    return ImmutableList.of(new FetchCommand());
  }

  @Override
  public void handleOptions(OptionsProvider optionsProvider) {
    PackageCacheOptions pkgOptions = optionsProvider.getOptions(PackageCacheOptions.class);
    isFetch.set(pkgOptions != null && pkgOptions.fetch);
  }

  @Override
  public ImmutableMap<SkyFunctionName, SkyFunction> getSkyFunctions(BlazeDirectories directories) {
    ImmutableMap.Builder<SkyFunctionName, SkyFunction> builder = ImmutableMap.builder();

    // Create the repository function everything flows through.
    builder.put(SkyFunctions.REPOSITORY, new RepositoryLoaderFunction());

    // Helper SkyFunctions.
    builder.put(SkyFunctions.REPOSITORY_DIRECTORY,
        new RepositoryDelegatorFunction(directories, repositoryHandlers, isFetch));
    builder.put(MavenServerFunction.NAME, new MavenServerFunction(directories));
    return builder.build();
  }
}
