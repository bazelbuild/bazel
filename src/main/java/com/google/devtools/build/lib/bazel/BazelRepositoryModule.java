// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.bazel.commands.FetchCommand;
import com.google.devtools.build.lib.bazel.repository.HttpArchiveFunction;
import com.google.devtools.build.lib.bazel.repository.HttpDownloadFunction;
import com.google.devtools.build.lib.bazel.repository.HttpJarFunction;
import com.google.devtools.build.lib.bazel.repository.JarFunction;
import com.google.devtools.build.lib.bazel.repository.LocalRepositoryFunction;
import com.google.devtools.build.lib.bazel.repository.MavenJarFunction;
import com.google.devtools.build.lib.bazel.repository.NewHttpArchiveFunction;
import com.google.devtools.build.lib.bazel.repository.NewLocalRepositoryFunction;
import com.google.devtools.build.lib.bazel.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.bazel.repository.RepositoryFunction;
import com.google.devtools.build.lib.bazel.repository.ZipFunction;
import com.google.devtools.build.lib.bazel.rules.android.AndroidNdkRepositoryFunction;
import com.google.devtools.build.lib.bazel.rules.android.AndroidNdkRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.android.AndroidSdkRepositoryFunction;
import com.google.devtools.build.lib.bazel.rules.android.AndroidSdkRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.workspace.HttpArchiveRule;
import com.google.devtools.build.lib.bazel.rules.workspace.HttpJarRule;
import com.google.devtools.build.lib.bazel.rules.workspace.LocalRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.workspace.MavenJarRule;
import com.google.devtools.build.lib.bazel.rules.workspace.NewHttpArchiveRule;
import com.google.devtools.build.lib.bazel.rules.workspace.NewLocalRepositoryRule;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.common.options.OptionsProvider;

import java.util.Map.Entry;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Adds support for fetching external code.
 */
public class BazelRepositoryModule extends BlazeModule {

  private BlazeDirectories directories;
  // A map of repository handlers that can be looked up by rule class name.
  private final ImmutableMap<String, RepositoryFunction> repositoryHandlers;
  private final AtomicBoolean isFetch = new AtomicBoolean(false);

  public BazelRepositoryModule() {
    repositoryHandlers = ImmutableMap.<String, RepositoryFunction>builder()
        .put(LocalRepositoryRule.NAME, new LocalRepositoryFunction())
        .put(HttpArchiveRule.NAME, new HttpArchiveFunction())
        .put(HttpJarRule.NAME, new HttpJarFunction())
        .put(MavenJarRule.NAME, new MavenJarFunction())
        .put(NewHttpArchiveRule.NAME, new NewHttpArchiveFunction())
        .put(NewLocalRepositoryRule.NAME, new NewLocalRepositoryFunction())
        .put(AndroidSdkRepositoryRule.NAME, new AndroidSdkRepositoryFunction())
        .put(AndroidNdkRepositoryRule.NAME, new AndroidNdkRepositoryFunction())
        .build();
  }

  @Override
  public void blazeStartup(OptionsProvider startupOptions,
      BlazeVersionInfo versionInfo, UUID instanceId, BlazeDirectories directories,
      Clock clock) {
    this.directories = directories;
    for (RepositoryFunction handler : repositoryHandlers.values()) {
      handler.setDirectories(directories);
    }
  }

  @Override
  public Set<Path> getImmutableDirectories() {
    return ImmutableSet.of(RepositoryFunction.getExternalRepositoryDirectory(directories));
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
    builder.put(SkyFunctionName.create(HttpDownloadFunction.NAME), new HttpDownloadFunction());
    builder.put(JarFunction.NAME, new JarFunction());
    builder.put(ZipFunction.NAME, new ZipFunction());
    return builder.build();
  }
}
