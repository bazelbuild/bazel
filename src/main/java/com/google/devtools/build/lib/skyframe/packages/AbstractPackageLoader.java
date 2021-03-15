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

import static com.google.common.base.Preconditions.checkState;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Suppliers;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.EventBus;
import com.google.common.hash.HashFunction;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.concurrent.NamedForkJoinPool;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.BuildFileName;
import com.google.devtools.build.lib.packages.CachingPackageLocator;
import com.google.devtools.build.lib.packages.ConstantRuleVisibility;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Package.Builder.DefaultPackageSettings;
import com.google.devtools.build.lib.packages.Package.ConfigSettingVisibilityPolicy;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageFactory.EnvironmentExtension;
import com.google.devtools.build.lib.packages.PackageLoadingListener;
import com.google.devtools.build.lib.packages.PackageValidator;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.repository.ExternalPackageHelper;
import com.google.devtools.build.lib.skyframe.BzlCompileFunction;
import com.google.devtools.build.lib.skyframe.BzlLoadFunction;
import com.google.devtools.build.lib.skyframe.ContainingPackageLookupFunction;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.ExternalPackageFunction;
import com.google.devtools.build.lib.skyframe.FileFunction;
import com.google.devtools.build.lib.skyframe.FileStateFunction;
import com.google.devtools.build.lib.skyframe.FileSymlinkCycleUniquenessFunction;
import com.google.devtools.build.lib.skyframe.FileSymlinkInfiniteExpansionUniquenessFunction;
import com.google.devtools.build.lib.skyframe.IgnoredPackagePrefixesFunction;
import com.google.devtools.build.lib.skyframe.ManagedDirectoriesKnowledge;
import com.google.devtools.build.lib.skyframe.PackageFunction;
import com.google.devtools.build.lib.skyframe.PackageFunction.ActionOnIOExceptionReadingBuildFile;
import com.google.devtools.build.lib.skyframe.PackageFunction.IncrementalityIntent;
import com.google.devtools.build.lib.skyframe.PackageFunction.LoadedPackageCacheEntry;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction.CrossRepositoryLabelViolationStrategy;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.PerBuildSyscallCache;
import com.google.devtools.build.lib.skyframe.PrecomputedFunction;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.RepositoryMappingFunction;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.StarlarkBuiltinsFunction;
import com.google.devtools.build.lib.skyframe.WorkspaceFileFunction;
import com.google.devtools.build.lib.skyframe.WorkspaceNameFunction;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.UnixGlob.FilesystemCalls;
import com.google.devtools.build.skyframe.BuildDriver;
import com.google.devtools.build.skyframe.Differencer;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.GraphInconsistencyReceiver;
import com.google.devtools.build.skyframe.ImmutableDiff;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.Injectable;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.SequentialBuildDriver;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.Version;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * Abstract base class of a {@link PackageLoader} implementation that has no incrementality or
 * caching.
 */
public abstract class AbstractPackageLoader implements PackageLoader {

  // See {@link PackageFactory.setMaxDirectoriesToEagerlyVisitInGlobbing}.
  private static final int MAX_DIRECTORIES_TO_EAGERLY_VISIT_IN_GLOBBING = 3000;

  private final ImmutableDiff preinjectedDiff;
  private final Differencer preinjectedDifferencer =
      new Differencer() {
        @Override
        public Diff getDiff(WalkableGraph fromGraph, Version fromVersion, Version toVersion)
            throws InterruptedException {
          return preinjectedDiff;
        }
      };
  private final Reporter commonReporter;
  protected final ConfiguredRuleClassProvider ruleClassProvider;
  private final PackageFactory pkgFactory;
  protected StarlarkSemantics starlarkSemantics;
  protected final ImmutableMap<SkyFunctionName, SkyFunction> extraSkyFunctions;
  private final AtomicReference<PathPackageLocator> pkgLocatorRef;
  protected final ExternalFilesHelper externalFilesHelper;
  protected final BlazeDirectories directories;
  private final HashFunction hashFunction;
  private final int legacyGlobbingThreads;
  @VisibleForTesting final ForkJoinPool forkJoinPoolForLegacyGlobbing;
  private final int skyframeThreads;

  /** Abstract base class of a builder for {@link PackageLoader} instances. */
  public abstract static class Builder {
    protected final Path workspaceDir;
    protected final BlazeDirectories directories;
    protected final PathPackageLocator pkgLocator;
    final AtomicReference<PathPackageLocator> pkgLocatorRef;
    private final ExternalPackageHelper externalPackageHelper;
    private ExternalFileAction externalFileAction;
    protected ExternalFilesHelper externalFilesHelper;
    protected ConfiguredRuleClassProvider ruleClassProvider = getDefaultRuleClassProvider();
    protected StarlarkSemantics starlarkSemantics;
    protected Reporter commonReporter = new Reporter(new EventBus());
    protected Map<SkyFunctionName, SkyFunction> extraSkyFunctions = new HashMap<>();
    List<PrecomputedValue.Injected> extraPrecomputedValues = new ArrayList<>();
    int legacyGlobbingThreads = 1;
    int skyframeThreads = 1;

    protected Builder(
        Root workspaceDir,
        Path installBase,
        Path outputBase,
        ImmutableList<BuildFileName> buildFilesByPriority,
        ExternalPackageHelper externalPackageHelper,
        ExternalFileAction externalFileAction) {
      this.workspaceDir = workspaceDir.asPath();
      Path devNull = workspaceDir.getRelative("/dev/null");
      directories =
          new BlazeDirectories(
              new ServerDirectories(installBase, outputBase, devNull),
              this.workspaceDir,
              /* defaultSystemJavabase= */ null,
              "blaze");

      this.pkgLocator =
          new PathPackageLocator(
              directories.getOutputBase(), ImmutableList.of(workspaceDir), buildFilesByPriority);
      this.pkgLocatorRef = new AtomicReference<>(pkgLocator);
      this.externalFileAction = externalFileAction;
      this.externalPackageHelper = externalPackageHelper;
    }

    public Builder setRuleClassProvider(ConfiguredRuleClassProvider ruleClassProvider) {
      this.ruleClassProvider = ruleClassProvider;
      return this;
    }

    public Builder setStarlarkSemantics(StarlarkSemantics semantics) {
      this.starlarkSemantics = semantics;
      return this;
    }

    public Builder useDefaultStarlarkSemantics() {
      this.starlarkSemantics = StarlarkSemantics.DEFAULT;
      return this;
    }

    /** Sets the reporter used by all skyframe evaluations. */
    public Builder setCommonReporter(Reporter commonReporter) {
      this.commonReporter = commonReporter;
      return this;
    }

    public Builder addExtraSkyFunctions(
        ImmutableMap<SkyFunctionName, SkyFunction> extraSkyFunctions) {
      this.extraSkyFunctions.putAll(extraSkyFunctions);
      return this;
    }

    public Builder addExtraPrecomputedValues(PrecomputedValue.Injected... extraPrecomputedValues) {
      return this.addExtraPrecomputedValues(Arrays.asList(extraPrecomputedValues));
    }

    public Builder addExtraPrecomputedValues(
        List<PrecomputedValue.Injected> extraPrecomputedValues) {
      this.extraPrecomputedValues.addAll(extraPrecomputedValues);
      return this;
    }

    public Builder setLegacyGlobbingThreads(int numThreads) {
      this.legacyGlobbingThreads = numThreads;
      return this;
    }

    public Builder setSkyframeThreads(int skyframeThreads) {
      this.skyframeThreads = skyframeThreads;
      return this;
    }

    public Builder setExternalFileAction(ExternalFileAction externalFileAction) {
      this.externalFileAction = externalFileAction;
      return this;
    }

    /** Throws {@link IllegalArgumentException} if builder args are incomplete/inconsistent. */
    protected void validate() {
      if (starlarkSemantics == null) {
        throw new IllegalArgumentException(
            "must call either setStarlarkSemantics or useDefaultStarlarkSemantics");
      }
    }

    public final PackageLoader build() {
      validate();
      externalFilesHelper =
          ExternalFilesHelper.create(
              pkgLocatorRef,
              externalFileAction,
              directories,
              ManagedDirectoriesKnowledge.NO_MANAGED_DIRECTORIES,
              externalPackageHelper);
      return buildImpl();
    }

    protected abstract PackageLoader buildImpl();

    protected abstract ConfiguredRuleClassProvider getDefaultRuleClassProvider();
  }

  AbstractPackageLoader(Builder builder) {
    this.ruleClassProvider = builder.ruleClassProvider;
    this.starlarkSemantics = builder.starlarkSemantics;
    this.commonReporter = builder.commonReporter;
    this.extraSkyFunctions = ImmutableMap.copyOf(builder.extraSkyFunctions);
    this.pkgLocatorRef = builder.pkgLocatorRef;
    this.legacyGlobbingThreads = builder.legacyGlobbingThreads;
    this.forkJoinPoolForLegacyGlobbing =
        NamedForkJoinPool.newNamedPool(
            "package-loader-globbing-pool", builder.legacyGlobbingThreads);
    this.skyframeThreads = builder.skyframeThreads;
    this.directories = builder.directories;
    this.hashFunction = builder.workspaceDir.getFileSystem().getDigestFunction().getHashFunction();

    this.externalFilesHelper = builder.externalFilesHelper;

    this.preinjectedDiff =
        makePreinjectedDiff(
            starlarkSemantics,
            builder.pkgLocator,
            ImmutableList.copyOf(builder.extraPrecomputedValues));
    pkgFactory =
        new PackageFactory(
            ruleClassProvider,
            forkJoinPoolForLegacyGlobbing,
            getEnvironmentExtensions(),
            "PackageLoader",
            DefaultPackageSettings.INSTANCE,
            PackageValidator.NOOP_VALIDATOR,
            PackageLoadingListener.NOOP_LISTENER);
  }

  private static ImmutableDiff makePreinjectedDiff(
      StarlarkSemantics starlarkSemantics,
      PathPackageLocator pkgLocator,
      ImmutableList<PrecomputedValue.Injected> extraPrecomputedValues) {
    final Map<SkyKey, SkyValue> valuesToInject = new HashMap<>();
    Injectable injectable =
        new Injectable() {
          @Override
          public void inject(Map<SkyKey, ? extends SkyValue> values) {
            valuesToInject.putAll(values);
          }

          @Override
          public void inject(SkyKey key, SkyValue value) {
            valuesToInject.put(key, value);
          }
        };
    for (PrecomputedValue.Injected injected : extraPrecomputedValues) {
      injected.inject(injectable);
    }
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(injectable, pkgLocator);
    PrecomputedValue.DEFAULT_VISIBILITY.set(injectable, ConstantRuleVisibility.PRIVATE);
    PrecomputedValue.CONFIG_SETTING_VISIBILITY_POLICY
        .set(injectable, ConfigSettingVisibilityPolicy.LEGACY_OFF);
    PrecomputedValue.STARLARK_SEMANTICS.set(injectable, starlarkSemantics);
    return new ImmutableDiff(ImmutableList.of(), valuesToInject);
  }

  @Override
  public void close() {
    // We don't use ForkJoinPool#shutdownNow since it has a performance bug. See
    // http://b/33482341#comment13.
    forkJoinPoolForLegacyGlobbing.shutdown();
  }

  @Override
  public Package loadPackage(PackageIdentifier pkgId)
      throws NoSuchPackageException, InterruptedException {
    return loadPackages(ImmutableList.of(pkgId)).getLoadedPackages().get(pkgId).get();
  }

  @Override
  public Result loadPackages(Iterable<PackageIdentifier> pkgIds) throws InterruptedException {
    ArrayList<SkyKey> keys = new ArrayList<>();
    for (PackageIdentifier pkgId : ImmutableSet.copyOf(pkgIds)) {
      keys.add(PackageValue.key(pkgId));
    }

    Reporter reporter = new Reporter(commonReporter);
    StoredEventHandler storedEventHandler = new StoredEventHandler();
    reporter.addHandler(storedEventHandler);
    ExecutorService executorServiceForSkyframe =
        AbstractQueueVisitor.createExecutorService(
            skyframeThreads, "skyframe-threadpool-for-package-loader");
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(true)
            // Technically, the Skyframe codepath we use shuts down the executor it creates and uses
            // (say, if we used #setNumThreads but not also #setExecutorServiceSupplier). Still,
            // since this is brittle and not explicitly tested, we act defensively and provide our
            // own executor that we explicitly shutdown ourselves.
            .setExecutorServiceSupplier(Suppliers.ofInstance(executorServiceForSkyframe))
            .setNumThreads(skyframeThreads)
            .setEventHandler(reporter)
            .build();
    try {
      return loadPackagesInternal(keys, evaluationContext, storedEventHandler);
    } finally {
      if (!executorServiceForSkyframe.isShutdown()) {
        ExecutorUtil.uninterruptibleShutdownNow(executorServiceForSkyframe);
      }
    }
  }

  private Result loadPackagesInternal(
      Iterable<SkyKey> pkgKeys,
      EvaluationContext evaluationContext,
      StoredEventHandler storedEventHandler)
      throws InterruptedException {
    EvaluationResult<PackageValue> evalResult =
        makeFreshDriver().evaluate(pkgKeys, evaluationContext);
    ImmutableMap.Builder<PackageIdentifier, PackageLoader.PackageOrException> result =
        ImmutableMap.builder();
    for (SkyKey key : pkgKeys) {
      ErrorInfo error = evalResult.getError(key);
      PackageValue packageValue = evalResult.get(key);
      checkState((error == null) != (packageValue == null));
      PackageIdentifier pkgId = (PackageIdentifier) key.argument();
      result.put(
          pkgId,
          error != null
              ? new PackageOrException(null, exceptionFromErrorInfo(error, pkgId))
              : new PackageOrException(packageValue.getPackage(), null));
    }
    return new Result(result.build(), storedEventHandler.getEvents());
  }

  public ConfiguredRuleClassProvider getRuleClassProvider() {
    return ruleClassProvider;
  }

  public PackageFactory getPackageFactory() {
    return pkgFactory;
  }

  private static NoSuchPackageException exceptionFromErrorInfo(
      ErrorInfo error, PackageIdentifier pkgId) {
    if (!error.getCycleInfo().isEmpty()) {
      return new BuildFileContainsErrorsException(
          pkgId, "Cycle encountered while loading package " + pkgId);
    }
    Throwable e = Preconditions.checkNotNull(error.getException());
    if (e instanceof NoSuchPackageException) {
      return (NoSuchPackageException) e;
    }
    throw new IllegalStateException(
        "Unexpected Exception type from PackageValue for '" + pkgId + "'' with error: " + error, e);
  }

  private BuildDriver makeFreshDriver() {
    return new SequentialBuildDriver(
        InMemoryMemoizingEvaluator.SUPPLIER.create(
            makeFreshSkyFunctions(),
            preinjectedDifferencer,
            EvaluationProgressReceiver.NULL,
            GraphInconsistencyReceiver.THROWING,
            InMemoryMemoizingEvaluator.DEFAULT_STORED_EVENT_FILTER,
            new MemoizingEvaluator.EmittedEventState(),
            /*keepEdges=*/ false));
  }

  protected abstract ImmutableList<EnvironmentExtension> getEnvironmentExtensions();

  protected abstract CrossRepositoryLabelViolationStrategy
      getCrossRepositoryLabelViolationStrategy();

  protected abstract ImmutableList<BuildFileName> getBuildFilesByPriority();

  protected abstract ExternalPackageHelper getExternalPackageHelper();

  protected abstract ActionOnIOExceptionReadingBuildFile getActionOnIOExceptionReadingBuildFile();

  private ImmutableMap<SkyFunctionName, SkyFunction> makeFreshSkyFunctions() {
    AtomicReference<TimestampGranularityMonitor> tsgm =
        new AtomicReference<>(new TimestampGranularityMonitor(BlazeClock.instance()));
    Cache<PackageIdentifier, LoadedPackageCacheEntry> packageFunctionCache =
        CacheBuilder.newBuilder().build();
    AtomicReference<FilesystemCalls> syscallCacheRef =
        new AtomicReference<>(
            PerBuildSyscallCache.newBuilder().setConcurrencyLevel(legacyGlobbingThreads).build());
    pkgFactory.setSyscalls(syscallCacheRef);
    pkgFactory.setMaxDirectoriesToEagerlyVisitInGlobbing(
        MAX_DIRECTORIES_TO_EAGERLY_VISIT_IN_GLOBBING);
    CachingPackageLocator cachingPackageLocator =
        new CachingPackageLocator() {
          @Override
          @Nullable
          public Path getBuildFileForPackage(PackageIdentifier packageName) {
            return pkgLocatorRef.get().getPackageBuildFileNullable(packageName, syscallCacheRef);
          }
        };
    ImmutableMap.Builder<SkyFunctionName, SkyFunction> builder = ImmutableMap.builder();
    builder
        .put(SkyFunctions.PRECOMPUTED, new PrecomputedFunction())
        .put(
            FileStateValue.FILE_STATE,
            new FileStateFunction(tsgm, syscallCacheRef, externalFilesHelper))
        .put(SkyFunctions.FILE_SYMLINK_CYCLE_UNIQUENESS, new FileSymlinkCycleUniquenessFunction())
        .put(
            SkyFunctions.FILE_SYMLINK_INFINITE_EXPANSION_UNIQUENESS,
            new FileSymlinkInfiniteExpansionUniquenessFunction())
        .put(FileValue.FILE, new FileFunction(pkgLocatorRef))
        .put(
            SkyFunctions.PACKAGE_LOOKUP,
            new PackageLookupFunction(
                /* deletedPackages= */ new AtomicReference<>(ImmutableSet.of()),
                getCrossRepositoryLabelViolationStrategy(),
                getBuildFilesByPriority(),
                getExternalPackageHelper()))
        .put(
            SkyFunctions.IGNORED_PACKAGE_PREFIXES,
            new IgnoredPackagePrefixesFunction(
                /*ignoredPackagePrefixesFile=*/ PathFragment.EMPTY_FRAGMENT))
        .put(SkyFunctions.CONTAINING_PACKAGE_LOOKUP, new ContainingPackageLookupFunction())
        .put(SkyFunctions.BZL_COMPILE, new BzlCompileFunction(pkgFactory, hashFunction))
        .put(SkyFunctions.STARLARK_BUILTINS, new StarlarkBuiltinsFunction(pkgFactory))
        .put(
            SkyFunctions.BZL_LOAD,
            BzlLoadFunction.create(
                pkgFactory, directories, hashFunction, CacheBuilder.newBuilder().build()))
        .put(SkyFunctions.WORKSPACE_NAME, new WorkspaceNameFunction())
        .put(
            WorkspaceFileValue.WORKSPACE_FILE,
            new WorkspaceFileFunction(
                ruleClassProvider, pkgFactory, directories, /*bzlLoadFunctionForInlining=*/ null))
        .put(SkyFunctions.EXTERNAL_PACKAGE, new ExternalPackageFunction(getExternalPackageHelper()))
        .put(SkyFunctions.REPOSITORY_MAPPING, new RepositoryMappingFunction())
        .put(
            SkyFunctions.PACKAGE,
            new PackageFunction(
                pkgFactory,
                cachingPackageLocator,
                /*showLoadingProgress=*/ new AtomicBoolean(false),
                packageFunctionCache,
                /*compiledBuildFileCache=*/ CacheBuilder.newBuilder().build(),
                /*numPackagesLoaded=*/ new AtomicInteger(0),
                /*bzlLoadFunctionForInlining=*/ null,
                /*packageProgress=*/ null,
                getActionOnIOExceptionReadingBuildFile(),
                // Tell PackageFunction to optimize for our use-case of no incrementality.
                IncrementalityIntent.NON_INCREMENTAL))
        .putAll(extraSkyFunctions);
    return builder.build();
  }
}
