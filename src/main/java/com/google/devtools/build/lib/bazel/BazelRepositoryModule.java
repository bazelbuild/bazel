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

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.CharMatcher;
import com.google.common.base.Preconditions;
import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
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
import com.google.devtools.build.lib.bazel.bzlmod.LocalPathRepoSpecs;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtensionRepoMappingEntriesFunction;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileFunction;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleOverride;
import com.google.devtools.build.lib.bazel.bzlmod.NonRegistryOverride;
import com.google.devtools.build.lib.bazel.bzlmod.RegistryFactoryImpl;
import com.google.devtools.build.lib.bazel.bzlmod.RegistryFunction;
import com.google.devtools.build.lib.bazel.bzlmod.RepoSpecFunction;
import com.google.devtools.build.lib.bazel.bzlmod.SingleExtensionEvalFunction;
import com.google.devtools.build.lib.bazel.bzlmod.SingleExtensionFunction;
import com.google.devtools.build.lib.bazel.bzlmod.SingleExtensionUsagesFunction;
import com.google.devtools.build.lib.bazel.bzlmod.VendorFileFunction;
import com.google.devtools.build.lib.bazel.bzlmod.VendorManager;
import com.google.devtools.build.lib.bazel.bzlmod.YankedVersionsFunction;
import com.google.devtools.build.lib.bazel.bzlmod.YankedVersionsUtil;
import com.google.devtools.build.lib.bazel.commands.FetchCommand;
import com.google.devtools.build.lib.bazel.commands.ModCommand;
import com.google.devtools.build.lib.bazel.commands.VendorCommand;
import com.google.devtools.build.lib.bazel.repository.RepositoryFetchFunction;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.BazelCompatibilityMode;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.CheckDirectDepsMode;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.LockfileMode;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.RepositoryOverride;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import com.google.devtools.build.lib.bazel.repository.downloader.UrlRewriter;
import com.google.devtools.build.lib.bazel.repository.downloader.UrlRewriterParseException;
import com.google.devtools.build.lib.bazel.repository.starlark.StarlarkRepositoryModule;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryDirtinessChecker;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
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
import com.google.devtools.build.lib.skyframe.MutableSupplier;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue.Injected;
import com.google.devtools.build.lib.skyframe.RepositoryMappingFunction;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.SkyframeExecutorRepositoryHelpersHolder;
import com.google.devtools.build.lib.starlarkbuildapi.repository.RepositoryBootstrap;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** Adds support for fetching external code. */
public class BazelRepositoryModule extends BlazeModule {
  // Default location (relative to output user root) of the repository cache.
  public static final String DEFAULT_CACHE_LOCATION = "cache/repos/v1";

  // Default list of registries.
  public static final ImmutableSet<String> DEFAULT_REGISTRIES =
      ImmutableSet.of("https://bcr.bazel.build/");

  private final AtomicBoolean isFetch = new AtomicBoolean(false);
  private final RepositoryCache repositoryCache = new RepositoryCache();
  private final MutableSupplier<Map<String, String>> clientEnvironmentSupplier =
      new MutableSupplier<>();
  private ImmutableMap<RepositoryName, PathFragment> overrides = ImmutableMap.of();
  private ImmutableMap<String, PathFragment> injections = ImmutableMap.of();
  private ImmutableMap<String, ModuleOverride> moduleOverrides = ImmutableMap.of();
  private FileSystem filesystem;
  private ImmutableSet<String> registries;
  private final AtomicBoolean ignoreDevDeps = new AtomicBoolean(false);
  private CheckDirectDepsMode checkDirectDepsMode = CheckDirectDepsMode.WARNING;
  private BazelCompatibilityMode bazelCompatibilityMode = BazelCompatibilityMode.ERROR;
  private LockfileMode bazelLockfileMode = LockfileMode.UPDATE;
  private Clock clock;
  private Instant lastRegistryInvalidation = Instant.EPOCH;

  private Optional<Path> vendorDirectory = Optional.empty();
  private List<String> allowedYankedVersions = ImmutableList.of();
  private RepositoryFetchFunction repositoryFetchFunction;
  private SingleExtensionEvalFunction singleExtensionEvalFunction;
  private ModuleFileFunction moduleFileFunction;
  private RepoSpecFunction repoSpecFunction;
  private YankedVersionsFunction yankedVersionsFunction;

  private final VendorCommand vendorCommand = new VendorCommand(clientEnvironmentSupplier);
  private final RegistryFactoryImpl registryFactory =
      new RegistryFactoryImpl(clientEnvironmentSupplier);

  @Nullable private CredentialModule credentialModule;

  private ImmutableMap<String, NonRegistryOverride> builtinModules = null;

  public BazelRepositoryModule() {}

  @VisibleForTesting
  public BazelRepositoryModule(ImmutableMap<String, NonRegistryOverride> builtinModules) {
    this.builtinModules = builtinModules;
  }

  private static DetailedExitCode detailedExitCode(String message, ExternalRepository.Code code) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setExternalRepository(ExternalRepository.newBuilder().setCode(code))
            .build());
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
      return print(repositoryCache.getPath());
    }
  }

  @Override
  public void serverInit(OptionsParsingResult startupOptions, ServerBuilder builder) {
    builder.addCommands(new FetchCommand());
    builder.addCommands(new ModCommand());
    builder.addCommands(vendorCommand);
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

    repositoryFetchFunction =
        new RepositoryFetchFunction(
            clientEnvironmentSupplier,
            isFetch,
            directories,
            repositoryCache.getRepoContentsCache());
    singleExtensionEvalFunction =
        new SingleExtensionEvalFunction(directories, clientEnvironmentSupplier);

    if (builtinModules == null) {
      builtinModules = ModuleFileFunction.getBuiltinModules();
    }

    moduleFileFunction =
        new ModuleFileFunction(
            runtime.getRuleClassProvider().getBazelStarlarkEnvironment(),
            directories.getWorkspace(),
            builtinModules);
    repoSpecFunction = new RepoSpecFunction();
    yankedVersionsFunction = new YankedVersionsFunction();

    builder
        .addSkyFunction(SkyFunctions.REPOSITORY_DIRECTORY, repositoryFetchFunction)
        .addSkyFunction(SkyFunctions.MODULE_FILE, moduleFileFunction)
        .addSkyFunction(SkyFunctions.BAZEL_DEP_GRAPH, new BazelDepGraphFunction())
        .addSkyFunction(
            SkyFunctions.BAZEL_LOCK_FILE,
            new BazelLockFileFunction(directories.getWorkspace(), directories.getOutputBase()))
        .addSkyFunction(SkyFunctions.BAZEL_FETCH_ALL, new BazelFetchAllFunction())
        .addSkyFunction(SkyFunctions.BAZEL_MOD_TIDY, new BazelModTidyFunction())
        .addSkyFunction(SkyFunctions.BAZEL_MODULE_INSPECTION, new BazelModuleInspectorFunction())
        .addSkyFunction(SkyFunctions.BAZEL_MODULE_RESOLUTION, new BazelModuleResolutionFunction())
        .addSkyFunction(SkyFunctions.SINGLE_EXTENSION, new SingleExtensionFunction())
        .addSkyFunction(SkyFunctions.SINGLE_EXTENSION_EVAL, singleExtensionEvalFunction)
        .addSkyFunction(SkyFunctions.SINGLE_EXTENSION_USAGES, new SingleExtensionUsagesFunction())
        .addSkyFunction(
            SkyFunctions.REGISTRY,
            new RegistryFunction(registryFactory, directories.getWorkspace()))
        .addSkyFunction(SkyFunctions.REPO_SPEC, repoSpecFunction)
        .addSkyFunction(SkyFunctions.YANKED_VERSIONS, yankedVersionsFunction)
        .addSkyFunction(
            SkyFunctions.VENDOR_FILE,
            new VendorFileFunction(runtime.getRuleClassProvider().getBazelStarlarkEnvironment()))
        .addSkyFunction(
            SkyFunctions.MODULE_EXTENSION_REPO_MAPPING_ENTRIES,
            new ModuleExtensionRepoMappingEntriesFunction());
    filesystem = runtime.getFileSystem();

    credentialModule = Preconditions.checkNotNull(runtime.getBlazeModule(CredentialModule.class));
  }

  @Override
  public void initializeRuleClasses(ConfiguredRuleClassProvider.Builder builder) {
    builder.addStarlarkBootstrap(new RepositoryBootstrap(new StarlarkRepositoryModule()));
  }

  @Override
  public void beforeCommand(CommandEnvironment env) throws AbruptExitException {
    DownloadManager downloadManager =
        new DownloadManager(
            repositoryCache.getDownloadCache(),
            env.getDownloaderDelegate(),
            env.getHttpDownloader());
    this.repositoryFetchFunction.setDownloadManager(downloadManager);
    this.moduleFileFunction.setDownloadManager(downloadManager);
    this.repoSpecFunction.setDownloadManager(downloadManager);
    this.yankedVersionsFunction.setDownloadManager(downloadManager);
    this.vendorCommand.setDownloadManager(downloadManager);

    clientEnvironmentSupplier.set(env.getRepoEnv());
    PackageOptions pkgOptions = env.getOptions().getOptions(PackageOptions.class);
    isFetch.set(pkgOptions != null && pkgOptions.fetch);

    ProcessWrapper processWrapper = ProcessWrapper.fromCommandEnvironment(env);
    repositoryFetchFunction.setProcessWrapper(processWrapper);
    repositoryFetchFunction.setSyscallCache(env.getSyscallCache());
    singleExtensionEvalFunction.setProcessWrapper(processWrapper);
    singleExtensionEvalFunction.setDownloadManager(downloadManager);

    RepositoryOptions repoOptions = env.getOptions().getOptions(RepositoryOptions.class);
    if (repoOptions != null) {
      downloadManager.setDisableDownload(repoOptions.disableDownload);
      if (repoOptions.repositoryDownloaderRetries >= 0) {
        downloadManager.setRetries(repoOptions.repositoryDownloaderRetries);
      }

      repositoryCache.getDownloadCache().setHardlink(repoOptions.useHardlinks);
      if (repoOptions.experimentalScaleTimeouts > 0.0) {
        repositoryFetchFunction.setTimeoutScaling(repoOptions.experimentalScaleTimeouts);
        singleExtensionEvalFunction.setTimeoutScaling(repoOptions.experimentalScaleTimeouts);
      } else {
        env.getReporter()
            .handle(
                Event.warn(
                    "Ignoring request to scale timeouts for repositories by a non-positive"
                        + " factor"));
        repositoryFetchFunction.setTimeoutScaling(1.0);
        singleExtensionEvalFunction.setTimeoutScaling(1.0);
      }
      if (repoOptions.repositoryCache != null) {
        repositoryCache.setPath(toPath(repoOptions.repositoryCache, env));
      } else {
        repositoryCache.setPath(
            env.getDirectories()
                .getServerDirectories()
                .getOutputUserRoot()
                .getRelative(DEFAULT_CACHE_LOCATION));
      }
      // Note that the repo contents cache stuff has to happen _after_ the repo cache stuff, because
      // the specific settings about the repo contents cache might overwrite the repo cache
      // settings. In particular, if `--repo_contents_cache` is not set (it's null), we use whatever
      // default set by `repositoryCache.setPath(...)`.
      if (repoOptions.repoContentsCache != null) {
        repositoryCache.getRepoContentsCache().setPath(toPath(repoOptions.repoContentsCache, env));
      }
      Path repoContentsCachePath = repositoryCache.getRepoContentsCache().getPath();
      if (repoContentsCachePath != null) {
        // Check that the repo contents cache directory, which is managed by a garbage collecting
        // idle task, does not contain the output base. Since the specified output base path may be
        // a symlink, we resolve it fully. Intermediate symlinks do not have to be checked as the
        // garbage collector ignores symlinks. We also resolve the repo contents cache directory,
        // where intermediate symlinks also don't matter since deletion only occurs under the fully
        // resolved path.
        Path resolvedOutputBase = env.getOutputBase();
        try {
          resolvedOutputBase = resolvedOutputBase.resolveSymbolicLinks();
        } catch (FileNotFoundException ignored) {
          // Will be created later.
        } catch (IOException e) {
          throw new AbruptExitException(
              detailedExitCode(
                  "could not resolve output base: %s".formatted(e.getMessage()),
                  Code.BAD_REPO_CONTENTS_CACHE),
              e);
        }
        Path resolvedRepoContentsCache = repoContentsCachePath;
        try {
          resolvedRepoContentsCache = resolvedRepoContentsCache.resolveSymbolicLinks();
        } catch (FileNotFoundException ignored) {
          // Will be created later.
        } catch (IOException e) {
          throw new AbruptExitException(
              detailedExitCode(
                  "could not resolve repo contents cache path: %s".formatted(e.getMessage()),
                  Code.BAD_REPO_CONTENTS_CACHE),
              e);
        }
        if (resolvedOutputBase.startsWith(resolvedRepoContentsCache)) {
          // This is dangerous as the repo contents cache GC may delete files in the output base.
          throw new AbruptExitException(
              detailedExitCode(
                  """
                  The output base [%s] is inside the repo contents cache [%s]. This can cause \
                  spurious failures. Disable the repo contents cache with `--repo_contents_cache=`, \
                  or specify `--repo_contents_cache=<path that doesn't contain the output base>`.
                  """
                      .formatted(resolvedOutputBase, resolvedRepoContentsCache),
                  Code.BAD_REPO_CONTENTS_CACHE));
        }
      }
      if (repoContentsCachePath != null
          && env.getWorkspace() != null
          && repoContentsCachePath.startsWith(env.getWorkspace())) {
        // Having the repo contents cache inside the main repo is very dangerous. During the
        // lifetime of a Bazel invocation, we treat files inside the main repo as immutable. This
        // can cause mysterious failures if we write files inside the main repo during the
        // invocation, as is often the case with the repo contents cache.
        // TODO: wyv@ - This is a crude check that disables some use cases (such as when the output
        //   base itself is inside the main repo). Investigate a better check.
        repositoryCache.getRepoContentsCache().setPath(null);
        throw new AbruptExitException(
            detailedExitCode(
                """
                The repo contents cache [%s] is inside the main repo [%s]. This can cause spurious \
                failures. Disable the repo contents cache with `--repo_contents_cache=`, or \
                specify `--repo_contents_cache=<path outside the main repo>`.
                """
                    .formatted(repoContentsCachePath, env.getWorkspace()),
                Code.BAD_REPO_CONTENTS_CACHE));
      }
      if (repositoryCache.getRepoContentsCache().isEnabled()) {
        try (SilentCloseable c =
            Profiler.instance()
                .profile(ProfilerTask.REPO_CACHE_GC_WAIT, "waiting to acquire repo cache lock")) {
          repositoryCache.getRepoContentsCache().acquireSharedLock();
        } catch (IOException e) {
          throw new AbruptExitException(
              detailedExitCode(
                  "could not acquire lock on repo contents cache", Code.BAD_REPO_CONTENTS_CACHE),
              e);
        }
        env.addIdleTask(
            repositoryCache
                .getRepoContentsCache()
                .createGcIdleTask(
                    repoOptions.repoContentsCacheGcMaxAge,
                    repoOptions.repoContentsCacheGcIdleDelay));
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
            UrlRewriter.getDownloaderUrlRewriter(
                env.getWorkspace(), repoOptions.downloaderConfig, env.getReporter());
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

      if (repoOptions.repositoryOverrides != null) {
        // To get the usual latest-wins semantics, we need a mutable map, as the builder
        // of an immutable map does not allow redefining the values of existing keys.
        // We use a LinkedHashMap to preserve the iteration order.
        Map<RepositoryName, PathFragment> overrideMap = new LinkedHashMap<>();
        for (RepositoryOverride override : repoOptions.repositoryOverrides) {
          if (override.path().isEmpty()) {
            overrideMap.remove(override.repositoryName());
            continue;
          }
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

      if (repoOptions.repositoryInjections != null) {
        Map<String, PathFragment> injectionMap = new LinkedHashMap<>();
        for (RepositoryOptions.RepositoryInjection injection : repoOptions.repositoryInjections) {
          if (injection.path().isEmpty()) {
            injectionMap.remove(injection.apparentName());
            continue;
          }
          String repoPath = getAbsolutePath(injection.path(), env);
          injectionMap.put(injection.apparentName(), PathFragment.create(repoPath));
        }
        ImmutableMap<String, PathFragment> newInjections = ImmutableMap.copyOf(injectionMap);
        if (!Maps.difference(injections, newInjections).areEqual()) {
          injections = newInjections;
        }
      } else {
        injections = ImmutableMap.of();
      }

      if (repoOptions.moduleOverrides != null) {
        Map<String, ModuleOverride> moduleOverrideMap = new LinkedHashMap<>();
        for (RepositoryOptions.ModuleOverride override : repoOptions.moduleOverrides) {
          if (override.path().isEmpty()) {
            moduleOverrideMap.remove(override.moduleName());
            continue;
          }
          String modulePath = getAbsolutePath(override.path(), env);
          moduleOverrideMap.put(
              override.moduleName(),
              new NonRegistryOverride(LocalPathRepoSpecs.create(modulePath)));
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
      if (env.getWorkspace() != null) {
        vendorDirectory =
            Optional.ofNullable(repoOptions.vendorDirectory)
                .map(vendorDirectory -> env.getWorkspace().getRelative(vendorDirectory));

        if (vendorDirectory.isPresent()) {
          try {
            Path externalRoot =
                env.getOutputBase().getRelative(LabelConstants.EXTERNAL_PATH_PREFIX);
            FileSystemUtils.ensureSymbolicLink(
                vendorDirectory.get().getChild(VendorManager.EXTERNAL_ROOT_SYMLINK_NAME),
                externalRoot);
            if (OS.getCurrent() == OS.WINDOWS) {
              // On Windows, symlinks are resolved differently.
              // Given <external>/repo_foo/link,
              // where <external>/repo_foo points to <vendor dir>/repo_foo in vendor mode
              // and repo_foo/link points to a relative path ../bazel-external/repo_bar/data.
              // Windows won't resolve `repo_foo` before resolving `link`, which causes
              // <external>/repo_foo/link to be resolved to <external>/bazel-external/repo_bar/data
              // To work around this, we create a symlink <external>/bazel-external -> <external>.
              FileSystemUtils.ensureSymbolicLink(
                  externalRoot.getChild(VendorManager.EXTERNAL_ROOT_SYMLINK_NAME), externalRoot);
            }
          } catch (IOException e) {
            env.getReporter()
                .handle(
                    Event.error(
                        "Failed to create symlink to external repo root under vendor directory: "
                            + e.getMessage()));
          }
        }
      }

      if (repoOptions.registries != null && !repoOptions.registries.isEmpty()) {
        registries = normalizeRegistries(repoOptions.registries);
      } else {
        registries = DEFAULT_REGISTRIES;
      }

      RepositoryRemoteExecutorFactory remoteExecutorFactory =
          env.getRuntime().getRepositoryRemoteExecutorFactory();
      RepositoryRemoteExecutor remoteExecutor = null;
      if (remoteExecutorFactory != null) {
        remoteExecutor = remoteExecutorFactory.create();
      }
      repositoryFetchFunction.setRepositoryRemoteExecutor(remoteExecutor);
      singleExtensionEvalFunction.setRepositoryRemoteExecutor(remoteExecutor);

      clock = env.getClock();
      try {
        var lastRegistryInvalidationValue =
            (PrecomputedValue)
                env.getSkyframeExecutor()
                    .getEvaluator()
                    .getExistingValue(RegistryFunction.LAST_INVALIDATION.getKey());
        if (lastRegistryInvalidationValue != null) {
          lastRegistryInvalidation = (Instant) lastRegistryInvalidationValue.get();
        }
      } catch (InterruptedException e) {
        // Not thrown in Bazel.
        throw new IllegalStateException(e);
      }
    }
  }

  private static ImmutableSet<String> normalizeRegistries(List<String> registries) {
    // Ensure that registries aren't duplicated even after `/modules/...` paths are appended to
    // them.
    return registries.stream()
        .map(url -> CharMatcher.is('/').trimTrailingFrom(url))
        .collect(toImmutableSet());
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

  /**
   * An empty path fragment is turned into {@code null}; otherwise, it's treated as relative to the
   * workspace root.
   */
  @Nullable
  private Path toPath(PathFragment path, CommandEnvironment env) {
    if (path.isEmpty() || env.getBlazeWorkspace().getWorkspace() == null) {
      return null;
    }
    return env.getBlazeWorkspace().getWorkspace().getRelative(path);
  }

  @Override
  public void afterCommand() throws AbruptExitException {
    if (repositoryCache.getRepoContentsCache().isEnabled()) {
      try {
        repositoryCache.getRepoContentsCache().releaseSharedLock();
      } catch (IOException e) {
        throw new AbruptExitException(
            detailedExitCode(
                "could not release lock on repo contents cache", Code.BAD_REPO_CONTENTS_CACHE),
            e);
      }
    }
  }

  @Override
  public ImmutableList<Injected> getPrecomputedValues() {
    Instant now = clock.now();
    if (now.isAfter(lastRegistryInvalidation.plus(RegistryFunction.INVALIDATION_INTERVAL))) {
      lastRegistryInvalidation = now;
    }
    return ImmutableList.of(
        PrecomputedValue.injected(RepositoryMappingFunction.REPOSITORY_OVERRIDES, overrides),
        PrecomputedValue.injected(ModuleFileFunction.INJECTED_REPOSITORIES, injections),
        PrecomputedValue.injected(ModuleFileFunction.MODULE_OVERRIDES, moduleOverrides),
        // That key will be reinjected by the sync command with a universally unique identifier.
        // Nevertheless, we need to provide a default value for other commands.
        PrecomputedValue.injected(
            RepositoryDirectoryValue.FORCE_FETCH, RepositoryDirectoryValue.FORCE_FETCH_DISABLED),
        PrecomputedValue.injected(
            RepositoryDirectoryValue.FORCE_FETCH_CONFIGURE,
            RepositoryDirectoryValue.FORCE_FETCH_DISABLED),
        PrecomputedValue.injected(ModuleFileFunction.REGISTRIES, registries),
        PrecomputedValue.injected(ModuleFileFunction.IGNORE_DEV_DEPS, ignoreDevDeps.get()),
        PrecomputedValue.injected(
            BazelModuleResolutionFunction.CHECK_DIRECT_DEPENDENCIES, checkDirectDepsMode),
        PrecomputedValue.injected(
            BazelModuleResolutionFunction.BAZEL_COMPATIBILITY_MODE, bazelCompatibilityMode),
        PrecomputedValue.injected(BazelLockFileFunction.LOCKFILE_MODE, bazelLockfileMode),
        PrecomputedValue.injected(RepositoryDirectoryValue.IS_VENDOR_COMMAND, false),
        PrecomputedValue.injected(RepositoryDirectoryValue.VENDOR_DIRECTORY, vendorDirectory),
        PrecomputedValue.injected(
            YankedVersionsUtil.ALLOWED_YANKED_VERSIONS, allowedYankedVersions),
        PrecomputedValue.injected(RegistryFunction.LAST_INVALIDATION, lastRegistryInvalidation));
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommonCommandOptions() {
    return ImmutableList.of(RepositoryOptions.class);
  }
}
