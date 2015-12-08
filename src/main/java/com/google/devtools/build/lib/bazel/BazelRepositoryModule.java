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
import com.google.devtools.build.lib.bazel.repository.FileFunction;
import com.google.devtools.build.lib.bazel.repository.GitCloneFunction;
import com.google.devtools.build.lib.bazel.repository.GitRepositoryFunction;
import com.google.devtools.build.lib.bazel.repository.HttpArchiveFunction;
import com.google.devtools.build.lib.bazel.repository.HttpDownloadFunction;
import com.google.devtools.build.lib.bazel.repository.HttpFileFunction;
import com.google.devtools.build.lib.bazel.repository.HttpJarFunction;
import com.google.devtools.build.lib.bazel.repository.JarFunction;
import com.google.devtools.build.lib.bazel.repository.MavenJarFunction;
import com.google.devtools.build.lib.bazel.repository.MavenServerFunction;
import com.google.devtools.build.lib.bazel.repository.NewGitRepositoryFunction;
import com.google.devtools.build.lib.bazel.repository.NewHttpArchiveFunction;
import com.google.devtools.build.lib.bazel.repository.TarGzFunction;
import com.google.devtools.build.lib.bazel.repository.ZipFunction;
import com.google.devtools.build.lib.bazel.rules.android.AndroidNdkRepositoryFunction;
import com.google.devtools.build.lib.bazel.rules.android.AndroidNdkRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.android.AndroidSdkRepositoryFunction;
import com.google.devtools.build.lib.bazel.rules.android.AndroidSdkRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.workspace.GitRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.workspace.HttpArchiveRule;
import com.google.devtools.build.lib.bazel.rules.workspace.HttpFileRule;
import com.google.devtools.build.lib.bazel.rules.workspace.HttpJarRule;
import com.google.devtools.build.lib.bazel.rules.workspace.MavenJarRule;
import com.google.devtools.build.lib.bazel.rules.workspace.NewGitRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.workspace.NewHttpArchiveRule;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryFunction;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryRule;
import com.google.devtools.build.lib.rules.repository.NewLocalRepositoryFunction;
import com.google.devtools.build.lib.rules.repository.NewLocalRepositoryRule;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.common.options.OptionsProvider;

import java.util.Map.Entry;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Adds support for fetching external code.
 */
public class BazelRepositoryModule extends BlazeModule {

  // A map of repository handlers that can be looked up by rule class name.
  private final ImmutableMap<String, RepositoryFunction> repositoryHandlers;
  private final AtomicBoolean isFetch = new AtomicBoolean(false);
  private HttpDownloadFunction downloadFunction;
  private GitCloneFunction gitCloneFunction;

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
            .build();
  }

  @Override
  public void beforeCommand(Command command, CommandEnvironment env) {
    downloadFunction.setReporter(env.getReporter());
    gitCloneFunction.setReporter(env.getReporter());
  }

  @Override
  public void blazeStartup(OptionsProvider startupOptions,
      BlazeVersionInfo versionInfo, UUID instanceId, BlazeDirectories directories,
      Clock clock) {
    for (RepositoryFunction handler : repositoryHandlers.values()) {
      handler.setDirectories(directories);
    }
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

    // Bazel-specific repository downloaders.
    for (RepositoryFunction handler : repositoryHandlers.values()) {
      builder.put(handler.getSkyFunctionName(), handler);
    }

    // Create the delegator everything flows through.
    builder.put(SkyFunctions.REPOSITORY,
        new RepositoryDelegatorFunction(directories, repositoryHandlers, isFetch));

    // Helper SkyFunctions.
    downloadFunction = new HttpDownloadFunction();
    builder.put(HttpDownloadFunction.NAME, downloadFunction);
    gitCloneFunction = new GitCloneFunction();
    builder.put(SkyFunctionName.create(GitCloneFunction.NAME), gitCloneFunction);
    builder.put(JarFunction.NAME, new JarFunction());
    builder.put(ZipFunction.NAME, new ZipFunction());
    builder.put(TarGzFunction.NAME, new TarGzFunction());
    builder.put(FileFunction.NAME, new FileFunction());
    builder.put(MavenServerFunction.NAME, new MavenServerFunction(directories));
    return builder.build();
  }
}
