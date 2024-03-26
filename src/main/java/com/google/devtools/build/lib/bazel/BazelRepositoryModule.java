// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.authandtls.GoogleAuthUtils;
import com.google.devtools.build.lib.authandtls.StaticCredentials;
import com.google.devtools.build.lib.authandtls.credentialhelper.CredentialHelperCredentials;
import com.google.devtools.build.lib.authandtls.credentialhelper.CredentialHelperEnvironment;
import com.google.devtools.build.lib.authandtls.credentialhelper.CredentialHelperProvider;
import com.google.devtools.build.lib.authandtls.credentialhelper.CredentialModule;
import com.google.devtools.build.lib.bazel.bzlmod.BazelDepGraphFunction;
import com.google.devtools.build.lib.bazel.bzlmod.BazelFetchAllFunction;
import com.google.devtools.build.lib.bazel.bzlmod.BazelLockFileFunction;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModTidyFunction;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorFunction;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleResolutionFunction;
import com.google.devtools.build.lib.bazel.bzlmod.LocalPathOverride;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtensionRepoMappingEntriesFunction;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileFunction;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleOverride;
import com.google.devtools.build.lib.bazel.bzlmod.NonRegistryOverride;
import com.google.devtools.build.lib.bazel.bzlmod.RegistryFactory;
import com.google.devtools.build.lib.bazel.bzlmod.RegistryFactoryImpl;
import com.google.devtools.build.lib.bazel.bzlmod.RepoSpecFunction;
import com.google.devtools.build.lib.bazel.bzlmod.SingleExtensionEvalFunction;
import com.google.devtools.build.lib.bazel.bzlmod.SingleExtensionUsagesFunction;
import com.google.devtools.build.lib.bazel.bzlmod.YankedVersionsUtil;
import com.google.devtools.build.lib.bazel.commands.FetchCommand;
import com.google.devtools.build.lib.bazel.commands.ModCommand;
import com.google.devtools.build.lib.bazel.commands.SyncCommand;
import com.google.devtools.build.lib.bazel.commands.VendorCommand;
import com.google.devtools.build.lib.bazel.repository.LocalConfigPlatformFunction;
import com.google.devtools.build.lib.bazel.repository.LocalConfigPlatformRule;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.BazelCompatibilityMode;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.CheckDirectDepsMode;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.LockfileMode;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.RepositoryOverride;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache;
import com.google.devtools.build.lib.bazel.repository.downloader.DelegatingDownloader;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import com.google.devtools.build.lib.bazel.repository.downloader.HttpDownloader;
import com.google.devtools.build.lib.bazel.repository.downloader.UrlRewriter;
import com.google.devtools.build.lib.bazel.repository.downloader.UrlRewriterParseException;
import com.google.devtools.build.lib.bazel.repository.starlark.StarlarkRepositoryFunction;
import com.google.devtools.build.lib.bazel.repository.starlark.StarlarkRepositoryModule;
import com.google.devtools.build.lib.bazel.rules.android.AndroidNdkRepositoryFunction;
import com.google.devtools.build.lib.bazel.rules.android.AndroidNdkRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.android.AndroidSdkRepositoryFunction;
import com.google.devtools.build.lib.bazel.rules.android.AndroidSdkRepositoryRule;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryFunction;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryRule;
import com.google.devtools.build.lib.rules.repository.NewLocalRepositoryFunction;
import com.google.devtools.build.lib.rules.repository.NewLocalRepositoryRule;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryDirtinessChecker;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.InfoItem;
import com.google.devtools.build.lib.runtime.ProcessWrapper;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutor;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutorFactory;
import com.google.devtools.build.lib.runtime.ServerBuilder;
import com.google.devtools.build.lib.runtime.WorkspaceBuilder;
import com.google.devtools.build.lib.server.FailureDetails.ExternalRepository;
import com.google.devtools.build.lib.server.FailureDetails.ExternalRepository.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.skyframe.MutableSupplier;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue.Injected;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.SkyframeExecutorRepositoryHelpersHolder;
import com.google.devtools.build.lib.starlarkbuildapi.repository.RepositoryBootstrap;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** Adds support for fetching external code. */
public class BazelRepositoryModule extends BlazeModule {
  // Default location (relative to output user root) of the repository cache.
  public static final String DEFAULT_CACHE_LOCATION = "cache/repos/v1";

  // Default list of registries.
  public static final ImmutableList<String> DEFAULT_REGISTRIES =
      ImmutableList.of("https://bcr.bazel.build/");

  // A map of repository handlers that can be looked up by rule class name.
  private final ImmutableMap<String, RepositoryFunction> repositoryHandlers;
  private final AtomicBoolean isFetch = new AtomicBoolean(false);
  private final StarlarkRepositoryFunction starlarkRepositoryFunction;
  private final RepositoryCache repositoryCache = new RepositoryCache();
  private final HttpDownloader httpDownloader = new HttpDownloader();
  private final DelegatingDownloader delegatingDownloader =
      new DelegatingDownloader(httpDownloader);
  private final DownloadManager downloadManager =
      new DownloadManager(repositoryCache, delegatingDownloader);
  private final MutableSupplier<Map<String, String>> clientEnvironmentSupplier =
      new MutableSupplier<>();
  private ImmutableMap<RepositoryName, PathFragment> overrides = ImmutableMap.of();
  private ImmutableMap<String, ModuleOverride> moduleOverrides = ImmutableMap.of();
  private Optional<RootedPath> resolvedFileReplacingWorkspace = Optional.empty();
  private FileSystem filesystem;
  private List<String> registries;
  private final AtomicBoolean ignoreDevDeps = new AtomicBoolean(false);
  private CheckDirectDepsMode checkDirectDepsMode = CheckDirectDepsMode.WARNING;
  private BazelCompatibilityMode bazelCompatibilityMode = BazelCompatibilityMode.ERROR;
  private LockfileMode bazelLockfileMode = LockfileMode.UPDATE;

  private Optional<Path> vendorDirectory;
  private List<String> allowedYankedVersions = ImmutableList.of();
  private SingleExtensionEvalFunction singleExtensionEvalFunction;
  private final ExecutorService repoFetchingWorkerThreadPool =
      Executors.newFixedThreadPool(
          100, new ThreadFactoryBuilder().setNameFormat("repo-fetching-worker-%d").build());

  @Nullable private CredentialModule credentialModule;

  private ImmutableMap<String, NonRegistryOverride> builtinModules = null;

  public BazelRepositoryModule() {
    this.starlarkRepositoryFunction = new StarlarkRepositoryFunction(downloadManager);
    this.repositoryHandlers = repositoryRules();
  }

  @VisibleForTesting
  public BazelRepositoryModule(ImmutableMap<String, NonRegistryOverride> builtinModules) {
    this();
    this.builtinModules = builtinModules;
  }

  private static DetailedExitCode detailedExitCode(String message, ExternalRepository.Code code) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setExternalRepository(ExternalRepository.newBuilder().setCode(code))
            .build());
  }

  public static ImmutableMap<String, RepositoryFunction> repositoryRules() {
    return ImmutableMap.<String, RepositoryFunction>builder()
        .put(LocalRepositoryRule.NAME, new LocalRepositoryFunction())
        .put(NewLocalRepositoryRule.NAME, new NewLocalRepositoryFunction())
        .put(AndroidSdkRepositoryRule.NAME, new AndroidSdkRepositoryFunction())
        .put(AndroidNdkRepositoryRule.NAME, new AndroidNdkRepositoryFunction())
        .put(LocalConfigPlatformRule.NAME, new LocalConfigPlatformFunction())
        .buildOrThrow();
  }

  private static class RepositoryCacheInfoItem extends InfoItem {
    private final RepositoryCache repositoryCache;

    RepositoryCacheInfoItem(RepositoryCache repositoryCache) {
      super("repository_cache", "The location of the repository download cache used");
      this.repositoryCache = repositoryCache;
    }

    @Override
    public byte[] get(
        Supplier<BuildConfigurationValue> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException, InterruptedException {
      return print(repositoryCache.getRootPath());
    }
  }

  @Override
  public void serverInit(OptionsParsingResult startupOptions, ServerBuilder builder) {
    builder.addCommands(new FetchCommand());
    builder.addCommands(new ModCommand());
    builder.addCommands(new SyncCommand());
    builder.addCommands(new VendorCommand());
    builder.addInfoItems(new RepositoryCacheInfoItem(repositoryCache));
  }

  @Override
  public void workspaceInit(
      BlazeRuntime runtime, BlazeDirectories directories, WorkspaceBuilder builder) {
    // TODO(b/27143724): Remove this guard when Google-internal flavor no longer uses repositories.
    if ("bazel".equals(runtime.getProductName())) {
      builder.setSkyframeExecutorRepositoryHelpersHolder(
          SkyframeExecutorRepositoryHelpersHolder.create(
              new RepositoryDirectoryDirtinessChecker()));
    }

    // Create the repository function everything flows through.
    RepositoryDelegatorFunction repositoryDelegatorFunction =
        new RepositoryDelegatorFunction(
            repositoryHandlers,
            starlarkRepositoryFunction,
            isFetch,
            clientEnvironmentSupplier,
            directories,
            BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER);
    RegistryFactory registryFactory =
        new RegistryFactoryImpl(
            directories.getWorkspace(), downloadManager, clientEnvironmentSupplier);
    singleExtensionEvalFunction =
        new SingleExtensionEvalFunction(directories, clientEnvironmentSupplier, downloadManager);

    if (builtinModules == null) {
      builtinModules = ModuleFileFunction.getBuiltinModules(directories.getEmbeddedBinariesRoot());
    }

    builder
        .addSkyFunction(SkyFunctions.REPOSITORY_DIRECTORY, repositoryDelegatorFunction)
        .addSkyFunction(
            SkyFunctions.MODULE_FILE,
            new ModuleFileFunction(
                runtime.getRuleClassProvider().getBazelStarlarkEnvironment(),
                registryFactory,
                directories.getWorkspace(),
                builtinModules))
        .addSkyFunction(SkyFunctions.BAZEL_DEP_GRAPH, new BazelDepGraphFunction())
        .addSkyFunction(
            SkyFunctions.BAZEL_LOCK_FILE, new BazelLockFileFunction(directories.getWorkspace()))
        .addSkyFunction(SkyFunctions.BAZEL_FETCH_ALL, new BazelFetchAllFunction())
        .addSkyFunction(SkyFunctions.BAZEL_MOD_TIDY, new BazelModTidyFunction())
        .addSkyFunction(SkyFunctions.BAZEL_MODULE_INSPECTION, new BazelModuleInspectorFunction())
        .addSkyFunction(SkyFunctions.BAZEL_MODULE_RESOLUTION, new BazelModuleResolutionFunction())
        .addSkyFunction(SkyFunctions.SINGLE_EXTENSION_EVAL, singleExtensionEvalFunction)
        .addSkyFunction(SkyFunctions.SINGLE_EXTENSION_USAGES, new SingleExtensionUsagesFunction())
        .addSkyFunction(SkyFunctions.REPO_SPEC, new RepoSpecFunction(registryFactory))
        .addSkyFunction(
            SkyFunctions.MODULE_EXTENSION_REPO_MAPPING_ENTRIES,
            new ModuleExtensionRepoMappingEntriesFunction());
    filesystem = runtime.getFileSystem();

    credentialModule = Preconditions.checkNotNull(runtime.getBlazeModule(CredentialModule.class));
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
    builder.addStarlarkBootstrap(new RepositoryBootstrap(new StarlarkRepositoryModule()));
  }

  @Override
  public void beforeCommand(CommandEnvironment env) throws AbruptExitException {
    clientEnvironmentSupplier.set(env.getRepoEnv());
    PackageOptions pkgOptions = env.getOptions().getOptions(PackageOptions.class);
    isFetch.set(pkgOptions != null && pkgOptions.fetch);
    resolvedFileReplacingWorkspace = Optional.empty();

    ProcessWrapper processWrapper = ProcessWrapper.fromCommandEnvironment(env);
    starlarkRepositoryFunction.setProcessWrapper(processWrapper);
    starlarkRepositoryFunction.setSyscallCache(env.getSyscallCache());
    singleExtensionEvalFunction.setProcessWrapper(processWrapper);

    RepositoryOptions repoOptions = env.getOptions().getOptions(RepositoryOptions.class);
    if (repoOptions != null) {
      switch (repoOptions.workerForRepoFetching) {
        case OFF:
          starlarkRepositoryFunction.setWorkerExecutorService(null);
          break;
        case PLATFORM:
          starlarkRepositoryFunction.setWorkerExecutorService(repoFetchingWorkerThreadPool);
          break;
        case VIRTUAL:
        case AUTO:
          try {
            // Since Google hasn't migrated to JDK 21 yet, we can't directly call
            // Executors.newVirtualThreadPerTaskExecutor here. But a bit of reflection never hurt
            // anyone... right? (OSS Bazel already ships with a bundled JDK 21)
            starlarkRepositoryFunction.setWorkerExecutorService(
                (ExecutorService)
                    Executors.class
                        .getDeclaredMethod("newVirtualThreadPerTaskExecutor")
                        .invoke(null));
          } catch (ReflectiveOperationException e) {
            if (repoOptions.workerForRepoFetching == RepositoryOptions.WorkerForRepoFetching.AUTO) {
              starlarkRepositoryFunction.setWorkerExecutorService(null);
            } else {
              throw new AbruptExitException(
                  detailedExitCode(
                      "couldn't create virtual worker thread executor for repo fetching",
                      Code.BAD_DOWNLOADER_CONFIG),
                  e);
            }
          }
      }
      downloadManager.setDisableDownload(repoOptions.disableDownload);
      if (repoOptions.repositoryDownloaderRetries >= 0) {
        downloadManager.setRetries(repoOptions.repositoryDownloaderRetries);
      }

      repositoryCache.setHardlink(repoOptions.useHardlinks);
      if (repoOptions.experimentalScaleTimeouts > 0.0) {
        starlarkRepositoryFunction.setTimeoutScaling(repoOptions.experimentalScaleTimeouts);
        singleExtensionEvalFunction.setTimeoutScaling(repoOptions.experimentalScaleTimeouts);
      } else {
        env.getReporter()
            .handle(
                Event.warn(
                    "Ignoring request to scale timeouts for repositories by a non-positive"
                        + " factor"));
        starlarkRepositoryFunction.setTimeoutScaling(1.0);
        singleExtensionEvalFunction.setTimeoutScaling(1.0);
      }
      if (repoOptions.experimentalRepositoryCache != null) {
        Path repositoryCachePath;
        if (repoOptions.experimentalRepositoryCache.isEmpty()) {
          // A set but empty path indicates a request to disable the repository cache.
          repositoryCachePath = null;
        } else if (repoOptions.experimentalRepositoryCache.isAbsolute()) {
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
          repositoryCachePath.createDirectoryAndParents();
          repositoryCache.setRepositoryCachePath(repositoryCachePath);
        } catch (IOException e) {
          env.getReporter()
              .handle(
                  Event.warn(
                      "Failed to set up cache at " + repositoryCachePath + ": " + e.getMessage()));
        }
      }

      try {
        downloadManager.setNetrcCreds(
            UrlRewriter.newCredentialsFromNetrc(
                env.getClientEnv(), env.getDirectories().getWorkingDirectory()));
      } catch (UrlRewriterParseException e) {
        // If the credentials extraction failed, we're letting bazel try without credentials.
        env.getReporter()
            .handle(
                Event.warn(String.format("Error parsing the .netrc file: %s.", e.getMessage())));
      }
      try {
        UrlRewriter rewriter =
            UrlRewriter.getDownloaderUrlRewriter(repoOptions.downloaderConfig, env.getReporter());
        downloadManager.setUrlRewriter(rewriter);
      } catch (UrlRewriterParseException e) {
        // It's important that the build stops ASAP, because this config file may be required for
        // security purposes, and the build must not proceed ignoring it.
        throw new AbruptExitException(
            detailedExitCode(
                String.format(
                    "Failed to parse downloader config%s: %s",
                    e.getLocation() != null ? String.format(" at %s", e.getLocation()) : "",
                    e.getMessage()),
                Code.BAD_DOWNLOADER_CONFIG));
      }

      try {
        AuthAndTLSOptions authAndTlsOptions = env.getOptions().getOptions(AuthAndTLSOptions.class);
        var credentialHelperEnvironment =
            CredentialHelperEnvironment.newBuilder()
                .setEventReporter(env.getReporter())
                .setWorkspacePath(env.getWorkspace())
                .setClientEnvironment(env.getClientEnv())
                .setHelperExecutionTimeout(authAndTlsOptions.credentialHelperTimeout)
                .build();
        CredentialHelperProvider credentialHelperProvider =
            GoogleAuthUtils.newCredentialHelperProvider(
                credentialHelperEnvironment,
                env.getCommandLinePathFactory(),
                authAndTlsOptions.credentialHelpers);

        downloadManager.setCredentialFactory(
            headers -> {
              Preconditions.checkNotNull(headers);

              return new CredentialHelperCredentials(
                  credentialHelperProvider,
                  credentialHelperEnvironment,
                  credentialModule.getCredentialCache(),
                  Optional.of(new StaticCredentials(headers)));
            });
      } catch (IOException e) {
        env.getReporter().handle(Event.error(e.getMessage()));
        env.getBlazeModuleEnvironment()
            .exit(
                new AbruptExitException(
                    detailedExitCode(
                        "Error initializing credential helper", Code.CREDENTIALS_INIT_FAILURE)));
        return;
      }

      if (repoOptions.experimentalDistdir != null) {
        downloadManager.setDistdir(
            repoOptions.experimentalDistdir.stream()
                .map(
                    path ->
                        path.isAbsolute()
                            ? filesystem.getPath(path)
                            : env.getBlazeWorkspace().getWorkspace().getRelative(path))
                .collect(Collectors.toList()));
      } else {
        downloadManager.setDistdir(ImmutableList.of());
      }

      if (repoOptions.httpTimeoutScaling > 0) {
        httpDownloader.setTimeoutScaling((float) repoOptions.httpTimeoutScaling);
      } else {
        env.getReporter()
            .handle(Event.warn("Ignoring request to scale http timeouts by a non-positive factor"));
        httpDownloader.setTimeoutScaling(1.0f);
      }
      httpDownloader.setMaxAttempts(repoOptions.httpConnectorAttempts);
      httpDownloader.setMaxRetryTimeout(repoOptions.httpConnectorRetryMaxTimeout);

      if (repoOptions.repositoryOverrides != null) {
        // To get the usual latest-wins semantics, we need a mutable map, as the builder
        // of an immutable map does not allow redefining the values of existing keys.
        // We use a LinkedHashMap to preserve the iteration order.
        Map<RepositoryName, PathFragment> overrideMap = new LinkedHashMap<>();
        for (RepositoryOverride override : repoOptions.repositoryOverrides) {
          String repoPath = getAbsolutePath(override.path(), env);
          overrideMap.put(override.repositoryName(), PathFragment.create(repoPath));
        }
        ImmutableMap<RepositoryName, PathFragment> newOverrides = ImmutableMap.copyOf(overrideMap);
        if (!Maps.difference(overrides, newOverrides).areEqual()) {
          overrides = newOverrides;
        }
      } else {
        overrides = ImmutableMap.of();
      }

      if (repoOptions.moduleOverrides != null) {
        Map<String, ModuleOverride> moduleOverrideMap = new LinkedHashMap<>();
        for (RepositoryOptions.ModuleOverride override : repoOptions.moduleOverrides) {
          String modulePath = getAbsolutePath(override.path(), env);
          moduleOverrideMap.put(override.moduleName(), LocalPathOverride.create(modulePath));
        }
        ImmutableMap<String, ModuleOverride> newModOverrides =
            ImmutableMap.copyOf(moduleOverrideMap);
        if (!Maps.difference(moduleOverrides, newModOverrides).areEqual()) {
          moduleOverrides = newModOverrides;
        }
      } else {
        moduleOverrides = ImmutableMap.of();
      }

      ignoreDevDeps.set(repoOptions.ignoreDevDependency);
      checkDirectDepsMode = repoOptions.checkDirectDependencies;
      bazelCompatibilityMode = repoOptions.bazelCompatibilityMode;
      bazelLockfileMode = repoOptions.lockfileMode;
      allowedYankedVersions = repoOptions.allowedYankedVersions;

      if (repoOptions.vendorDirectory != null) {
        vendorDirectory =
            Optional.of(
                repoOptions.vendorDirectory.isAbsolute()
                    ? filesystem.getPath(repoOptions.vendorDirectory)
                    : env.getWorkspace().getRelative(repoOptions.vendorDirectory));
      } else {
        vendorDirectory = Optional.empty();
      }

      if (repoOptions.registries != null && !repoOptions.registries.isEmpty()) {
        registries = repoOptions.registries;
      } else {
        registries = DEFAULT_REGISTRIES;
      }

      if (!Strings.isNullOrEmpty(repoOptions.experimentalResolvedFileInsteadOfWorkspace)) {
        Path resolvedFile;
        if (env.getWorkspace() != null) {
          resolvedFile =
              env.getWorkspace()
                  .getRelative(repoOptions.experimentalResolvedFileInsteadOfWorkspace);
        } else {
          resolvedFile = filesystem.getPath(repoOptions.experimentalResolvedFileInsteadOfWorkspace);
        }
        resolvedFileReplacingWorkspace =
            Optional.of(RootedPath.toRootedPath(Root.absoluteRoot(filesystem), resolvedFile));
      }

      RepositoryRemoteExecutorFactory remoteExecutorFactory =
          env.getRuntime().getRepositoryRemoteExecutorFactory();
      RepositoryRemoteExecutor remoteExecutor = null;
      if (remoteExecutorFactory != null) {
        remoteExecutor = remoteExecutorFactory.create();
      }
      starlarkRepositoryFunction.setRepositoryRemoteExecutor(remoteExecutor);
      singleExtensionEvalFunction.setRepositoryRemoteExecutor(remoteExecutor);
      delegatingDownloader.setDelegate(env.getRuntime().getDownloaderSupplier().get());
    }
  }

  /**
   * If the given path is absolute path, leave it as it is. If the given path is a relative path, it
   * is relative to the current working directory. If the given path starts with '%workspace%, it is
   * relative to the workspace root, which is the output of `bazel info workspace`.
   *
   * @return Absolute Path
   */
  private String getAbsolutePath(String path, CommandEnvironment env) {
    if (env.getWorkspace() != null) {
      path = path.replace("%workspace%", env.getWorkspace().getPathString());
    }
    if (!PathFragment.isAbsolute(path)) {
      path = env.getWorkingDirectory().getRelative(path).getPathString();
    }
    return path;
  }

  @Override
  public ImmutableList<Injected> getPrecomputedValues() {
    return ImmutableList.of(
        PrecomputedValue.injected(RepositoryDelegatorFunction.REPOSITORY_OVERRIDES, overrides),
        PrecomputedValue.injected(ModuleFileFunction.MODULE_OVERRIDES, moduleOverrides),
        PrecomputedValue.injected(
            RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE,
            resolvedFileReplacingWorkspace),
        // That key will be reinjected by the sync command with a universally unique identifier.
        // Nevertheless, we need to provide a default value for other commands.
        PrecomputedValue.injected(
            RepositoryDelegatorFunction.FORCE_FETCH,
            RepositoryDelegatorFunction.FORCE_FETCH_DISABLED),
        PrecomputedValue.injected(
            RepositoryDelegatorFunction.FORCE_FETCH_CONFIGURE,
            RepositoryDelegatorFunction.FORCE_FETCH_DISABLED),
        PrecomputedValue.injected(ModuleFileFunction.REGISTRIES, registries),
        PrecomputedValue.injected(ModuleFileFunction.IGNORE_DEV_DEPS, ignoreDevDeps.get()),
        PrecomputedValue.injected(
            BazelModuleResolutionFunction.CHECK_DIRECT_DEPENDENCIES, checkDirectDepsMode),
        PrecomputedValue.injected(
            BazelModuleResolutionFunction.BAZEL_COMPATIBILITY_MODE, bazelCompatibilityMode),
        PrecomputedValue.injected(BazelLockFileFunction.LOCKFILE_MODE, bazelLockfileMode),
        PrecomputedValue.injected(RepositoryDelegatorFunction.IS_VENDOR_COMMAND, false),
        PrecomputedValue.injected(RepositoryDelegatorFunction.VENDOR_DIRECTORY, vendorDirectory),
        PrecomputedValue.injected(
            YankedVersionsUtil.ALLOWED_YANKED_VERSIONS, allowedYankedVersions));
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return ImmutableList.of(RepositoryOptions.class);
  }
}
