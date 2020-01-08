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
package com.google.devtools.build.lib.query2.testutil;

import static com.google.devtools.build.lib.packages.Rule.ALL_LABELS;

import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.ConstantRuleVisibility;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageFactory.EnvironmentExtension;
import com.google.devtools.build.lib.packages.StarlarkSemanticsOptions;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.util.MockToolsConfig;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.pkgcache.TargetPatternPreloader;
import com.google.devtools.build.lib.query2.QueryEnvironmentFactory;
import com.google.devtools.build.lib.query2.common.AbstractBlazeQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ThreadSafeMutableSet;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryParser;
import com.google.devtools.build.lib.query2.engine.QueryUtil;
import com.google.devtools.build.lib.query2.engine.QueryUtil.AggregateAllOutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.ThreadSafeOutputFormatterCallback;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.skyframe.BlacklistedPackagePrefixesFunction;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsProvider;
import java.io.IOException;
import java.util.AbstractSet;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.UUID;

/** An implementation of AbstractQueryHelper to support testing bazel query. */
public abstract class SkyframeQueryHelper extends AbstractQueryHelper<Target> {
  protected SkyframeExecutor skyframeExecutor;
  protected FileSystem fileSystem = new InMemoryFileSystem(BlazeClock.instance());
  protected Path rootDirectory;
  protected BlazeDirectories directories;
  private String toolsRepository;

  protected AnalysisMock analysisMock;
  private QueryEnvironmentFactory queryEnvironmentFactory;

  private PackageManager pkgManager;
  private TargetPatternPreloader targetParser;
  private PathFragment relativeWorkingDirectory = PathFragment.EMPTY_FRAGMENT;
  private boolean blockUniverseEvaluationErrors;
  protected final ActionKeyContext actionKeyContext = new ActionKeyContext();

  private final PathFragment blacklistedPackagePrefixesFile = PathFragment.create("blacklist");

  @Override
  public void setUp() throws Exception {
    super.setUp();
    rootDirectory = createDir(getRootDirectoryNameForSetup());
    analysisMock = AnalysisMock.get();
    directories =
        new BlazeDirectories(
            new ServerDirectories(
                fileSystem.getPath("/install"),
                fileSystem.getPath("/output"),
                fileSystem.getPath("/user_root")),
            rootDirectory,
            /* defaultSystemJavabase= */ null,
            analysisMock.getProductName());

    initTargetPatternEvaluator(analysisMock.createRuleClassProvider());

    MockToolsConfig mockToolsConfig = new MockToolsConfig(rootDirectory);
    analysisMock.setupMockClient(mockToolsConfig);
    analysisMock.setupMockWorkspaceFiles(directories.getEmbeddedBinariesRoot());
    analysisMock.setupMockToolsRepository(mockToolsConfig);
    analysisMock.ccSupport().setup(mockToolsConfig);
    analysisMock.pySupport().setup(mockToolsConfig);
    performAdditionalClientSetup(mockToolsConfig);

    this.queryEnvironmentFactory = makeQueryEnvironmentFactory();
  }

  protected abstract String getRootDirectoryNameForSetup();

  protected abstract void performAdditionalClientSetup(MockToolsConfig mockToolsConfig)
      throws IOException;

  protected Path createDir(String pathName) throws IOException {
    Path dir = fileSystem.getPath(pathName);
    dir.createDirectoryAndParents();
    return dir;
  }

  @Override
  public PathFragment getBlacklistedPackagePrefixesFile() {
    return blacklistedPackagePrefixesFile;
  }

  @Override
  public void setBlockUniverseEvaluationErrors(boolean blockUniverseEvaluationErrors) {
    if (this.blockUniverseEvaluationErrors == blockUniverseEvaluationErrors) {
      return;
    }
    this.blockUniverseEvaluationErrors = blockUniverseEvaluationErrors;
  }

  protected QueryEnvironmentFactory makeQueryEnvironmentFactory() {
    return new QueryEnvironmentFactory();
  }

  @Override
  public Path getRootDirectory() {
    return rootDirectory;
  }

  @Override
  public void clearAllFiles() throws IOException {
    rootDirectory.deleteTreesBelow();
  }

  @Override
  public void writeFile(String fileName, String... lines) throws IOException {
    Path file = rootDirectory.getRelative(fileName);
    if (file.exists()) {
      throw new IOException("Could not create scratch file (file exists) " + fileName);
    }
    file.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(file, Joiner.on('\n').join(lines));
  }

  @Override
  public void overwriteFile(String fileName, String... lines) throws IOException {
    Path file = rootDirectory.getRelative(fileName);
    file.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(file, Joiner.on('\n').join(lines));
  }

  @Override
  public void ensureSymbolicLink(String link, String target) throws IOException {
    Path linkPath = rootDirectory.getRelative(link);
    Path targetPath = rootDirectory.getRelative(target);
    linkPath.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.ensureSymbolicLink(linkPath, targetPath);
  }

  @Override
  public AbstractBlazeQueryEnvironment<Target> getQueryEnvironment() {
    return queryEnvironmentFactory.create(
        pkgManager.newTransitiveLoader(),
        skyframeExecutor,
        pkgManager,
        pkgManager,
        targetParser,
        relativeWorkingDirectory,
        keepGoing,
        /*strictScope=*/ true,
        orderedResults,
        universeScope,
        /*loadingPhaseThreads=*/ 1,
        /*labelFilter=*/ ALL_LABELS,
        getReporter(),
        this.settings,
        getExtraQueryFunctions(),
        pkgManager.getPackagePath(),
        blockUniverseEvaluationErrors,
        /*useGraphlessQuery=*/ false);
  }

  protected abstract Iterable<QueryFunction> getExtraQueryFunctions();

  @Override
  public ResultAndTargets<Target> evaluateQuery(String query)
      throws QueryException, InterruptedException {
    try (AbstractBlazeQueryEnvironment<Target> env = getQueryEnvironment()) {
      return evaluateQuery(query, env);
    }
  }

  public ResultAndTargets<Target> evaluateQuery(
      String query, AbstractBlazeQueryEnvironment<Target> env)
      throws QueryException, InterruptedException {
    AggregateAllOutputFormatterCallback<Target, ?> callback =
        QueryUtil.newOrderedAggregateAllOutputFormatterCallback(env);
    QueryEvalResult queryEvalResult;
    try {
      queryEvalResult =
          env.evaluateQuery(env.transformParsedQuery(QueryParser.parse(query, env)), callback);
    } catch (IOException e) {
      // Should be impossible since AggregateAllOutputFormatterCallback doesn't throw IOException.
      throw new IllegalStateException(e);
    }
    return new ResultAndTargets<>(
        queryEvalResult, new OrderedThreadSafeImmutableSet(env, callback.getResult()));
  }

  @Override
  public Set<Target> evaluateQueryRaw(String query) throws QueryException, InterruptedException {
    Set<Target> result = new LinkedHashSet<>();
    ThreadSafeOutputFormatterCallback<Target> callback =
        new ThreadSafeOutputFormatterCallback<Target>() {
          @Override
          public synchronized void processOutput(Iterable<Target> partialResult) {
            Iterables.addAll(result, partialResult);
          }
        };
    try (AbstractBlazeQueryEnvironment<Target> env = getQueryEnvironment()) {
      try {
        env.evaluateQuery(env.transformParsedQuery(QueryParser.parse(query, env)), callback);
      } catch (IOException e) {
        // Should be impossible since the callback we passed in above doesn't throw IOException.
        throw new IllegalStateException(e);
      }
    }
    return result;
  }

  @Override
  public String getToolsRepository() {
    return toolsRepository;
  }

  @Override
  public String getLabel(Target target) {
    return target.getLabel().toString();
  }

  protected void initTargetPatternEvaluator(ConfiguredRuleClassProvider ruleClassProvider) {
    this.toolsRepository = ruleClassProvider.getToolsRepository();
    skyframeExecutor = createSkyframeExecutor(ruleClassProvider);
    PackageCacheOptions packageCacheOptions = Options.getDefaults(PackageCacheOptions.class);
    packageCacheOptions.defaultVisibility = ConstantRuleVisibility.PRIVATE;
    packageCacheOptions.showLoadingProgress = true;
    packageCacheOptions.globbingThreads = 7;
    packageCacheOptions.packagePath = ImmutableList.of(rootDirectory.getPathString());
    PathPackageLocator packageLocator =
        skyframeExecutor.createPackageLocator(
            getReporter(), packageCacheOptions.packagePath, rootDirectory);
    try {
      skyframeExecutor.sync(
          getReporter(),
          packageCacheOptions,
          packageLocator,
          Options.getDefaults(StarlarkSemanticsOptions.class),
          UUID.randomUUID(),
          ImmutableMap.<String, String>of(),
          new TimestampGranularityMonitor(BlazeClock.instance()),
          OptionsProvider.EMPTY);
    } catch (InterruptedException | AbruptExitException e) {
      throw new IllegalStateException(e);
    }
    pkgManager = skyframeExecutor.getPackageManager();
    targetParser = pkgManager.newTargetPatternPreloader();
  }

  @Override
  public void useRuleClassProvider(ConfiguredRuleClassProvider ruleClassProvider) {
    initTargetPatternEvaluator(ruleClassProvider);
  }

  protected SkyframeExecutor createSkyframeExecutor(ConfiguredRuleClassProvider ruleClassProvider) {
    PackageFactory pkgFactory =
        TestConstants.PACKAGE_FACTORY_BUILDER_FACTORY_FOR_TESTING
            .builder(directories)
            .setEnvironmentExtensions(getEnvironmentExtensions())
            .build(ruleClassProvider, fileSystem);
    SkyframeExecutor skyframeExecutor =
        BazelSkyframeExecutorConstants.newBazelSkyframeExecutorBuilder()
            .setPkgFactory(pkgFactory)
            .setFileSystem(fileSystem)
            .setDirectories(directories)
            .setActionKeyContext(actionKeyContext)
            .setDefaultBuildOptions(getDefaultBuildOptions(ruleClassProvider))
            .setBlacklistedPackagePrefixesFunction(
                new BlacklistedPackagePrefixesFunction(blacklistedPackagePrefixesFile))
            .setExtraSkyFunctions(analysisMock.getSkyFunctions(directories))
            .build();
    skyframeExecutor.injectExtraPrecomputedValues(
        ImmutableList.of(
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE, Optional.absent()),
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.REPOSITORY_OVERRIDES, ImmutableMap.of()),
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.DEPENDENCY_FOR_UNCONDITIONAL_FETCHING,
                RepositoryDelegatorFunction.DONT_FETCH_UNCONDITIONALLY)));
    TestConstants.processSkyframeExecutorForTesting(skyframeExecutor);
    return skyframeExecutor;
  }

  protected abstract Iterable<EnvironmentExtension> getEnvironmentExtensions();

  protected abstract BuildOptions getDefaultBuildOptions(
      ConfiguredRuleClassProvider ruleClassProvider);

  @Override
  public void assertPackageNotLoaded(String packageName) throws Exception {
    MemoizingEvaluator evaluator = skyframeExecutor.getEvaluatorForTesting();
    SkyKey key = PackageValue.key(PackageIdentifier.parse(packageName));
    if (evaluator.getExistingValue(key) != null
        || evaluator.getExistingErrorForTesting(key) != null) {
      throw new IllegalStateException("Package was loaded: " + packageName);
    }
  }

  /**
   * A wrapper to maintain an ordered copy of set of targets which also respect equality rules
   * defined by {@link ThreadSafeMutableSet}.
   */
  private static class OrderedThreadSafeImmutableSet extends AbstractSet<Target> {
    private final ThreadSafeMutableSet<Target> targetSet;
    private final List<Target> orderedTargetList;

    private OrderedThreadSafeImmutableSet(QueryEnvironment<Target> env, Set<Target> targets) {
      this.targetSet = env.createThreadSafeMutableSet();
      this.orderedTargetList = new ArrayList<>(targets.size());

      // The order is determined by implementation of iterator on the source set of targets, which
      // can be deterministic or non-deterministic.
      for (Target target : targets) {
        if (targetSet.add(target)) {
          orderedTargetList.add(target);
        }
      }
    }

    @Override
    public Iterator<Target> iterator() {
      return orderedTargetList.iterator();
    }

    @Override
    public int size() {
      return targetSet.size();
    }

    @Override
    public boolean add(Target element) {
      throw new IllegalStateException("Add operation on immutable set is not supported.");
    }

    @Override
    public boolean contains(Object obj) {
      return targetSet.contains(obj);
    }

    @Override
    public boolean remove(Object obj) {
      throw new IllegalStateException("Remove operation on immutable set is not supported.");
    }
  }
}
