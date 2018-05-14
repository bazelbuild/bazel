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

import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.bazel.commands.FetchCommand;
import com.google.devtools.build.lib.bazel.repository.GitRepositoryFunction;
import com.google.devtools.build.lib.bazel.repository.HttpArchiveFunction;
import com.google.devtools.build.lib.bazel.repository.HttpFileFunction;
import com.google.devtools.build.lib.bazel.repository.HttpJarFunction;
import com.google.devtools.build.lib.bazel.repository.MavenDownloader;
import com.google.devtools.build.lib.bazel.repository.MavenJarFunction;
import com.google.devtools.build.lib.bazel.repository.MavenServerFunction;
import com.google.devtools.build.lib.bazel.repository.MavenServerRepositoryFunction;
import com.google.devtools.build.lib.bazel.repository.NewGitRepositoryFunction;
import com.google.devtools.build.lib.bazel.repository.NewHttpArchiveFunction;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.RepositoryOverride;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache;
import com.google.devtools.build.lib.bazel.repository.downloader.HttpDownloader;
import com.google.devtools.build.lib.bazel.repository.skylark.SkylarkRepositoryFunction;
import com.google.devtools.build.lib.bazel.repository.skylark.SkylarkRepositoryModule;
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
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryFunction;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryRule;
import com.google.devtools.build.lib.rules.repository.NewLocalRepositoryFunction;
import com.google.devtools.build.lib.rules.repository.NewLocalRepositoryRule;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryLoaderFunction;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.ServerBuilder;
import com.google.devtools.build.lib.runtime.WorkspaceBuilder;
import com.google.devtools.build.lib.runtime.commands.InfoItem;
import com.google.devtools.build.lib.skyframe.MutableSupplier;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue.Injected;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.SkyValueDirtinessChecker;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsProvider;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** Adds support for fetching external code. */
public class BazelRepositoryModule extends BlazeModule {

  // Default location (relative to output user root) of the repository cache.
  public static final String DEFAULT_CACHE_LOCATION = "cache/repos/v1";

  // A map of repository handlers that can be looked up by rule class name.
  private final ImmutableMap<String, RepositoryFunction> repositoryHandlers;
  private final AtomicBoolean isFetch = new AtomicBoolean(false);
  private final SkylarkRepositoryFunction skylarkRepositoryFunction;
  private final RepositoryCache repositoryCache = new RepositoryCache();
  private final HttpDownloader httpDownloader = new HttpDownloader(repositoryCache);
  private final MavenDownloader mavenDownloader = new MavenDownloader(repositoryCache);
  private final MutableSupplier<Map<String, String>> clientEnvironmentSupplier =
      new MutableSupplier<>();
  private ImmutableMap<RepositoryName, PathFragment> overrides = ImmutableMap.of();
  private FileSystem filesystem;

  public BazelRepositoryModule() {
    this.skylarkRepositoryFunction = new SkylarkRepositoryFunction(httpDownloader);
    this.repositoryHandlers = repositoryRules(httpDownloader, mavenDownloader);
  }

  public static ImmutableMap<String, RepositoryFunction> repositoryRules(
      HttpDownloader httpDownloader, MavenDownloader mavenDownloader) {
    return ImmutableMap.<String, RepositoryFunction>builder()
        .put(LocalRepositoryRule.NAME, new LocalRepositoryFunction())
        .put(HttpArchiveRule.NAME, new HttpArchiveFunction(httpDownloader))
        .put(GitRepositoryRule.NAME, new GitRepositoryFunction(httpDownloader))
        .put(HttpJarRule.NAME, new HttpJarFunction(httpDownloader))
        .put(HttpFileRule.NAME, new HttpFileFunction(httpDownloader))
        .put(MavenJarRule.NAME, new MavenJarFunction(mavenDownloader))
        .put(NewHttpArchiveRule.NAME, new NewHttpArchiveFunction(httpDownloader))
        .put(NewGitRepositoryRule.NAME, new NewGitRepositoryFunction(httpDownloader))
        .put(NewLocalRepositoryRule.NAME, new NewLocalRepositoryFunction())
        .put(AndroidSdkRepositoryRule.NAME, new AndroidSdkRepositoryFunction())
        .put(AndroidNdkRepositoryRule.NAME, new AndroidNdkRepositoryFunction())
        .put(MavenServerRule.NAME, new MavenServerRepositoryFunction())
        .build();
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
          return repositoryValue.repositoryExists() && repositoryValue.isFetchingDelayed()
              ? DirtyResult.dirty(skyValue)
              : DirtyResult.notDirty(skyValue);
        }
      };

  private static class RepositoryCacheInfoItem extends InfoItem {
    private final RepositoryCache repositoryCache;

    RepositoryCacheInfoItem(RepositoryCache repositoryCache) {
      super("repository_cache", "The location of the repository download cache used");
      this.repositoryCache = repositoryCache;
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException, InterruptedException {
      return this.repositoryCache.getRootPath().toString().getBytes(StandardCharsets.UTF_8);
    }
  }

  @Override
  public void serverInit(OptionsProvider startupOptions, ServerBuilder builder) {
    builder.addCommands(new FetchCommand());
    builder.addInfoItems(new RepositoryCacheInfoItem(repositoryCache));
  }

  @Override
  public void workspaceInit(
      BlazeRuntime runtime, BlazeDirectories directories, WorkspaceBuilder builder) {
    builder.addCustomDirtinessChecker(REPOSITORY_VALUE_CHECKER);
    // Create the repository function everything flows through.
    builder.addSkyFunction(SkyFunctions.REPOSITORY, new RepositoryLoaderFunction());
    builder.addSkyFunction(
        SkyFunctions.REPOSITORY_DIRECTORY,
        new RepositoryDelegatorFunction(
            repositoryHandlers,
            skylarkRepositoryFunction,
            isFetch,
            clientEnvironmentSupplier,
            directories));
    builder.addSkyFunction(MavenServerFunction.NAME, new MavenServerFunction(directories));
    filesystem = runtime.getFileSystem();
  }

  @Override
  public void initializeRuleClasses(ConfiguredRuleClassProvider.Builder builder) {
    for (Map.Entry<String, RepositoryFunction> handler : repositoryHandlers.entrySet()) {
      RuleDefinition ruleDefinition;
      try {
        ruleDefinition =
            handler.getValue().getRuleDefinition().getDeclaredConstructor().newInstance();
      } catch (IllegalAccessException
          | InstantiationException
          | NoSuchMethodException
          | InvocationTargetException e) {
        throw new IllegalStateException(e);
      }
      builder.addRuleDefinition(ruleDefinition);
    }
    builder.addSkylarkModule(SkylarkRepositoryModule.class);
  }

  @Override
  public void beforeCommand(CommandEnvironment env) {
    clientEnvironmentSupplier.set(env.getActionClientEnv());
    PackageCacheOptions pkgOptions = env.getOptions().getOptions(PackageCacheOptions.class);
    isFetch.set(pkgOptions != null && pkgOptions.fetch);

    RepositoryOptions repoOptions = env.getOptions().getOptions(RepositoryOptions.class);
    if (repoOptions != null) {
      if (repoOptions.experimentalRepositoryCache != null) {
        Path repositoryCachePath;
        if (repoOptions.experimentalRepositoryCache.isAbsolute()) {
          repositoryCachePath = filesystem.getPath(repoOptions.experimentalRepositoryCache);
        } else {
          repositoryCachePath =
              env.getBlazeWorkspace()
                  .getWorkspace()
                  .getRelative(repoOptions.experimentalRepositoryCache);
        }
        repositoryCache.setRepositoryCachePath(repositoryCachePath);
      } else {
        Path repositoryCachePath =
            env.getDirectories()
                .getServerDirectories()
                .getOutputUserRoot()
                .getRelative(DEFAULT_CACHE_LOCATION);
        try {
          FileSystemUtils.createDirectoryAndParents(repositoryCachePath);
          repositoryCache.setRepositoryCachePath(repositoryCachePath);
        } catch (IOException e) {
          env.getReporter()
              .handle(
                  Event.warn(
                      "Failed to set up cache at "
                          + repositoryCachePath.toString()
                          + ": "
                          + e.getMessage()));
        }
      }

      if (repoOptions.experimentalDistdir != null) {
        httpDownloader.setDistdir(
            repoOptions
                .experimentalDistdir
                .stream()
                .map(
                    path ->
                        path.isAbsolute()
                            ? filesystem.getPath(path)
                            : env.getBlazeWorkspace().getWorkspace().getRelative(path))
                .collect(Collectors.toList()));
      }

      if (repoOptions.repositoryOverrides != null) {
        ImmutableMap.Builder<RepositoryName, PathFragment> builder = ImmutableMap.builder();
        for (RepositoryOverride override : repoOptions.repositoryOverrides) {
          builder.put(override.repositoryName(), override.path());
        }
        ImmutableMap<RepositoryName, PathFragment> newOverrides = builder.build();
        if (!Maps.difference(overrides, newOverrides).areEqual()) {
          overrides = newOverrides;
        }
      } else {
        overrides = ImmutableMap.of();
      }
    }
  }

  @Override
  public ImmutableList<Injected> getPrecomputedValues() {
    return ImmutableList.of(
        PrecomputedValue.injected(RepositoryDelegatorFunction.REPOSITORY_OVERRIDES, overrides));
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return ImmutableSet.of("fetch", "build", "query").contains(command.name())
        ? ImmutableList.<Class<? extends OptionsBase>>of(RepositoryOptions.class)
        : ImmutableList.<Class<? extends OptionsBase>>of();
  }
}
