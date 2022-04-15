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

package com.google.devtools.build.lib.rules.repository;

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;

import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.hash.HashFunction;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.actions.ThreadStateReceiver;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleResolutionFunction;
import com.google.devtools.build.lib.bazel.bzlmod.BzlmodRepoRuleHelperImpl;
import com.google.devtools.build.lib.bazel.bzlmod.BzlmodRepoRuleValue;
import com.google.devtools.build.lib.bazel.bzlmod.FakeRegistry;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtensionResolutionValue;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileFunction;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.CheckDirectDepsMode;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import com.google.devtools.build.lib.bazel.repository.starlark.StarlarkRepositoryFunction;
import com.google.devtools.build.lib.bazel.repository.starlark.StarlarkRepositoryModule;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue.SuccessfulRepositoryDirectoryValue;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.AlreadyReportedRepositoryAccessException;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.skyframe.BzlCompileFunction;
import com.google.devtools.build.lib.skyframe.BzlLoadFunction;
import com.google.devtools.build.lib.skyframe.BzlmodRepoRuleFunction;
import com.google.devtools.build.lib.skyframe.ContainingPackageLookupFunction;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.ExternalPackageFunction;
import com.google.devtools.build.lib.skyframe.FileFunction;
import com.google.devtools.build.lib.skyframe.FileStateFunction;
import com.google.devtools.build.lib.skyframe.IgnoredPackagePrefixesFunction;
import com.google.devtools.build.lib.skyframe.LocalRepositoryLookupFunction;
import com.google.devtools.build.lib.skyframe.ManagedDirectoriesKnowledge;
import com.google.devtools.build.lib.skyframe.PackageFunction;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction.CrossRepositoryLabelViolationStrategy;
import com.google.devtools.build.lib.skyframe.PrecomputedFunction;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.WorkspaceFileFunction;
import com.google.devtools.build.lib.starlarkbuildapi.repository.RepositoryBootstrap;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileStateKey;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SequencedRecordingDifferencer;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Tests for {@link RepositoryDelegatorFunction} */
@RunWith(JUnit4.class)
public class RepositoryDelegatorTest extends FoundationTestCase {
  private Path overrideDirectory;
  private MemoizingEvaluator evaluator;
  private TestManagedDirectoriesKnowledge managedDirectoriesKnowledge;
  private RecordingDifferencer differencer;
  private TestStarlarkRepositoryFunction testStarlarkRepositoryFunction;
  private Path rootPath;
  private FakeRegistry.Factory registryFactory;

  @Before
  public void setupDelegator() throws Exception {
    rootPath = scratch.dir("/outputbase");
    scratch.file(
        rootPath.getRelative("MODULE.bazel").getPathString(), "module(name='test',version='0.1')");
    BlazeDirectories directories =
        new BlazeDirectories(
            new ServerDirectories(rootPath, rootPath, rootPath),
            rootPath,
            /* defaultSystemJavabase= */ null,
            TestConstants.PRODUCT_NAME);
    managedDirectoriesKnowledge = new TestManagedDirectoriesKnowledge();
    DownloadManager downloader = Mockito.mock(DownloadManager.class);
    RepositoryFunction localRepositoryFunction = new LocalRepositoryFunction();
    testStarlarkRepositoryFunction =
        new TestStarlarkRepositoryFunction(rootPath, downloader, managedDirectoriesKnowledge);
    ImmutableMap<String, RepositoryFunction> repositoryHandlers =
        ImmutableMap.of(LocalRepositoryRule.NAME, localRepositoryFunction);
    RepositoryDelegatorFunction delegatorFunction =
        new RepositoryDelegatorFunction(
            repositoryHandlers,
            testStarlarkRepositoryFunction,
            /*isFetch=*/ new AtomicBoolean(true),
            /*clientEnvironmentSupplier=*/ ImmutableMap::of,
            directories,
            managedDirectoriesKnowledge,
            BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER);
    AtomicReference<PathPackageLocator> pkgLocator =
        new AtomicReference<>(
            new PathPackageLocator(
                rootPath,
                ImmutableList.of(Root.fromPath(rootPath)),
                BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY));
    ExternalFilesHelper externalFilesHelper =
        ExternalFilesHelper.createForTesting(
            pkgLocator,
            ExternalFileAction.DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_PATHS,
            directories);
    differencer = new SequencedRecordingDifferencer();

    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder
        .clearWorkspaceFileSuffixForTesting()
        .addStarlarkBootstrap(new RepositoryBootstrap(new StarlarkRepositoryModule()));
    ConfiguredRuleClassProvider ruleClassProvider = builder.build();

    PackageFactory pkgFactory =
        AnalysisMock.get()
            .getPackageFactoryBuilderForTesting(directories)
            .build(ruleClassProvider, fileSystem);

    registryFactory = new FakeRegistry.Factory();

    HashFunction hashFunction = fileSystem.getDigestFunction().getHashFunction();
    evaluator =
        new InMemoryMemoizingEvaluator(
            ImmutableMap.<SkyFunctionName, SkyFunction>builder()
                .put(
                    FileStateKey.FILE_STATE,
                    new FileStateFunction(
                        Suppliers.ofInstance(
                            new TimestampGranularityMonitor(BlazeClock.instance())),
                        SyscallCache.NO_CACHE,
                        externalFilesHelper))
                .put(FileValue.FILE, new FileFunction(pkgLocator))
                .put(SkyFunctions.REPOSITORY_DIRECTORY, delegatorFunction)
                .put(
                    SkyFunctions.PACKAGE,
                    new PackageFunction(
                        null,
                        null,
                        null,
                        null,
                        null,
                        /*packageProgress=*/ null,
                        PackageFunction.ActionOnIOExceptionReadingBuildFile.UseOriginalIOException
                            .INSTANCE,
                        PackageFunction.IncrementalityIntent.INCREMENTAL,
                        k -> ThreadStateReceiver.NULL_INSTANCE))
                .put(
                    SkyFunctions.PACKAGE_LOOKUP,
                    new PackageLookupFunction(
                        new AtomicReference<>(ImmutableSet.of()),
                        CrossRepositoryLabelViolationStrategy.ERROR,
                        BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY,
                        BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER))
                .put(
                    WorkspaceFileValue.WORKSPACE_FILE,
                    new WorkspaceFileFunction(
                        ruleClassProvider,
                        pkgFactory,
                        directories,
                        /*bzlLoadFunctionForInlining=*/ null))
                .put(
                    SkyFunctions.LOCAL_REPOSITORY_LOOKUP,
                    new LocalRepositoryLookupFunction(
                        BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER))
                .put(
                    SkyFunctions.EXTERNAL_PACKAGE,
                    new ExternalPackageFunction(
                        BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER))
                .put(SkyFunctions.PRECOMPUTED, new PrecomputedFunction())
                .put(SkyFunctions.BZL_COMPILE, new BzlCompileFunction(pkgFactory, hashFunction))
                .put(
                    SkyFunctions.BZL_LOAD,
                    BzlLoadFunction.create(
                        pkgFactory, directories, hashFunction, Caffeine.newBuilder().build()))
                .put(SkyFunctions.CONTAINING_PACKAGE_LOOKUP, new ContainingPackageLookupFunction())
                .put(
                    SkyFunctions.IGNORED_PACKAGE_PREFIXES,
                    new IgnoredPackagePrefixesFunction(
                        /*ignoredPackagePrefixesFile=*/ PathFragment.EMPTY_FRAGMENT))
                .put(SkyFunctions.RESOLVED_HASH_VALUES, new ResolvedHashesFunction())
                .put(SkyFunctions.MODULE_FILE, new ModuleFileFunction(registryFactory, rootPath))
                .put(SkyFunctions.BAZEL_MODULE_RESOLUTION, new BazelModuleResolutionFunction())
                .put(
                    SkyFunctions.MODULE_EXTENSION_RESOLUTION,
                    new SkyFunction() {
                      @Override
                      public SkyValue compute(SkyKey skyKey, Environment env) {
                        // Dummy SkyFunction that returns nothing.
                        return ModuleExtensionResolutionValue.create(
                            ImmutableMap.of(), ImmutableMap.of(), ImmutableListMultimap.of());
                      }
                    })
                .put(
                    BzlmodRepoRuleValue.BZLMOD_REPO_RULE,
                    new BzlmodRepoRuleFunction(
                        pkgFactory, ruleClassProvider, directories, new BzlmodRepoRuleHelperImpl()))
                .build(),
            differencer);
    overrideDirectory = scratch.dir("/foo");
    scratch.file("/foo/WORKSPACE");
    RepositoryDelegatorFunction.REPOSITORY_OVERRIDES.set(differencer, ImmutableMap.of());
    RepositoryDelegatorFunction.DEPENDENCY_FOR_UNCONDITIONAL_FETCHING.set(
        differencer, RepositoryDelegatorFunction.DONT_FETCH_UNCONDITIONALLY);
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(differencer, pkgLocator.get());
    PrecomputedValue.STARLARK_SEMANTICS.set(differencer, StarlarkSemantics.DEFAULT);
    RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE.set(
        differencer, Optional.empty());
    PrecomputedValue.REPO_ENV.set(differencer, ImmutableMap.of());
    RepositoryDelegatorFunction.OUTPUT_VERIFICATION_REPOSITORY_RULES.set(
        differencer, ImmutableSet.of());
    RepositoryDelegatorFunction.RESOLVED_FILE_FOR_VERIFICATION.set(differencer, Optional.empty());
    RepositoryDelegatorFunction.ENABLE_BZLMOD.set(differencer, true);
    ModuleFileFunction.IGNORE_DEV_DEPS.set(differencer, false);
    BazelModuleResolutionFunction.CHECK_DIRECT_DEPENDENCIES.set(
        differencer, CheckDirectDepsMode.WARNING);
  }

  @Test
  public void testOverride() throws Exception {
    RepositoryDelegatorFunction.REPOSITORY_OVERRIDES.set(
        differencer,
        ImmutableMap.of(RepositoryName.createUnvalidated("foo"), overrideDirectory.asFragment()));

    StoredEventHandler eventHandler = new StoredEventHandler();
    SkyKey key = RepositoryDirectoryValue.key(RepositoryName.createUnvalidated("foo"));
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(false)
            .setNumThreads(8)
            .setEventHandler(eventHandler)
            .build();
    EvaluationResult<SkyValue> result =
        evaluator.evaluate(ImmutableList.of(key), evaluationContext);
    assertThat(result.hasError()).isFalse();
    RepositoryDirectoryValue repositoryDirectoryValue = (RepositoryDirectoryValue) result.get(key);
    Path expectedPath = scratch.dir("/outputbase/external/foo");
    Path actualPath = repositoryDirectoryValue.getPath();
    assertThat(actualPath).isEqualTo(expectedPath);
    assertThat(actualPath.isSymbolicLink()).isTrue();
    assertThat(actualPath.readSymbolicLink()).isEqualTo(overrideDirectory.asFragment());
  }

  @Test
  public void testRepositoryDirtinessChecker() throws Exception {
    TimestampGranularityMonitor tsgm = new TimestampGranularityMonitor(new ManualClock());
    TestManagedDirectoriesKnowledge knowledge = new TestManagedDirectoriesKnowledge();

    RepositoryDirectoryDirtinessChecker checker =
        new RepositoryDirectoryDirtinessChecker(rootPath, knowledge);
    RepositoryName repositoryName = RepositoryName.create("repo");
    RepositoryDirectoryValue.Key key = RepositoryDirectoryValue.key(repositoryName);

    SuccessfulRepositoryDirectoryValue usual =
        RepositoryDirectoryValue.builder()
            .setPath(rootDirectory.getRelative("a"))
            .setDigest(new byte[] {1})
            .build();

    assertThat(checker.check(key, usual, SyscallCache.NO_CACHE, tsgm).isDirty()).isFalse();

    SuccessfulRepositoryDirectoryValue fetchDelayed =
        RepositoryDirectoryValue.builder()
            .setPath(rootDirectory.getRelative("b"))
            .setFetchingDelayed()
            .setDigest(new byte[] {1})
            .build();

    assertThat(checker.check(key, fetchDelayed, SyscallCache.NO_CACHE, tsgm).isDirty()).isTrue();

    RepositoryName managedName = RepositoryName.create("managed");
    RepositoryDirectoryValue.Key managedKey = RepositoryDirectoryValue.key(managedName);
    SuccessfulRepositoryDirectoryValue withManagedDirectories =
        RepositoryDirectoryValue.builder()
            .setPath(rootDirectory.getRelative("c"))
            .setDigest(new byte[] {1})
            .build();

    knowledge.setManagedDirectories(ImmutableMap.of(PathFragment.create("m"), managedName));
    assertThat(
            checker
                .check(managedKey, withManagedDirectories, SyscallCache.NO_CACHE, tsgm)
                .isDirty())
        .isTrue();

    Path managedDirectoryM = rootPath.getRelative("m");
    assertThat(managedDirectoryM.createDirectory()).isTrue();

    assertThat(
            checker
                .check(managedKey, withManagedDirectories, SyscallCache.NO_CACHE, tsgm)
                .isDirty())
        .isFalse();

    managedDirectoryM.deleteTree();
    assertThat(
            checker
                .check(managedKey, withManagedDirectories, SyscallCache.NO_CACHE, tsgm)
                .isDirty())
        .isTrue();
  }

  @Test
  public void testManagedDirectoriesCauseRepositoryReFetches() throws Exception {
    scratch.file(rootPath.getRelative("BUILD").getPathString());
    scratch.file(
        rootPath.getRelative("repo_rule.bzl").getPathString(),
        "def _impl(rctx):",
        " rctx.file('BUILD', '')",
        "fictive_repo_rule = repository_rule(implementation = _impl)");
    scratch.overwriteFile(
        rootPath.getRelative("WORKSPACE").getPathString(),
        "workspace(name = 'abc')",
        "load(':repo_rule.bzl', 'fictive_repo_rule')",
        "fictive_repo_rule(name = 'repo1')");

    // Managed directories from workspace() attribute will not be parsed by this test, since
    // we are not calling SequencedSkyframeExecutor.
    // That's why we will directly fill managed directories value (the corresponding structure
    // is passed to RepositoryDelegatorFunction during construction).
    managedDirectoriesKnowledge.setManagedDirectories(
        ImmutableMap.of(PathFragment.create("dir1"), RepositoryName.create("repo1")));

    loadRepo("repo1");

    assertThat(testStarlarkRepositoryFunction.isFetchCalled()).isTrue();
    testStarlarkRepositoryFunction.reset();

    loadRepo("repo1");
    // Nothing changed, fetch does not happen.
    assertThat(testStarlarkRepositoryFunction.isFetchCalled()).isFalse();
    testStarlarkRepositoryFunction.reset();

    // Delete managed directory, fetch should happen again.
    Path managedDirectory = rootPath.getRelative("dir1");
    managedDirectory.deleteTree();
    loadRepo("repo1");
    assertThat(testStarlarkRepositoryFunction.isFetchCalled()).isTrue();
    testStarlarkRepositoryFunction.reset();

    // Change managed directories declaration, fetch should happen.
    // NB: we are making sure that managed directories exist to check only the declaration changes
    // were percepted.
    rootPath.getRelative("dir1").createDirectory();
    rootPath.getRelative("dir2").createDirectory();

    managedDirectoriesKnowledge.setManagedDirectories(
        ImmutableMap.of(
            PathFragment.create("dir1"),
            RepositoryName.create("repo1"),
            PathFragment.create("dir2"),
            RepositoryName.create("repo1")));
    loadRepo("repo1");

    assertThat(testStarlarkRepositoryFunction.isFetchCalled()).isTrue();
    testStarlarkRepositoryFunction.reset();

    managedDirectoriesKnowledge.setManagedDirectories(ImmutableMap.of());
    loadRepo("repo1");

    assertThat(testStarlarkRepositoryFunction.isFetchCalled()).isTrue();
    testStarlarkRepositoryFunction.reset();
  }

  @Test
  public void testFetchRepositoryException_eventHandled() throws Exception {
    scratch.file(
        rootPath.getRelative("rule.bzl").getPathString(),
        "def _impl(ctx):",
        "    pass",
        "sample = rule(",
        "    implementation = _impl,",
        "    toolchains = ['//:toolchain_type'],",
        ")");
    scratch.file(
        rootPath.getRelative("BUILD").getPathString(),
        "load('rule.bzl', 'sample')",
        "toolchain_type(name = 'toolchain_type')",
        "sample(name = 'sample')");
    scratch.file(
        rootPath.getRelative("repo_rule.bzl").getPathString(),
        "def _impl(repo_ctx):",
        "# Error: no file written",
        "    pass",
        "broken_repo = repository_rule(implementation = _impl)");
    scratch.overwriteFile(
        rootPath.getRelative("WORKSPACE").getPathString(),
        "load('repo_rule.bzl', 'broken_repo')",
        "broken_repo(name = 'broken')");

    StoredEventHandler eventHandler = new StoredEventHandler();
    SkyKey key = RepositoryDirectoryValue.key(RepositoryName.createUnvalidated("broken"));
    // Make it be evaluated every time, as we are testing evaluation.
    differencer.invalidate(ImmutableSet.of(key));
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(false)
            .setNumThreads(8)
            .setEventHandler(eventHandler)
            .build();
    EvaluationResult<SkyValue> result =
        evaluator.evaluate(ImmutableList.of(key), evaluationContext);

    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getException())
        .isInstanceOf(AlreadyReportedRepositoryAccessException.class);
    assertThat(eventHandler.hasErrors()).isTrue();
    assertThat(eventHandler.getEvents()).hasSize(1);
  }

  @Test
  public void loadRepositoryNotDefined() throws Exception {
    // WORKSPACE is empty
    scratch.overwriteFile(rootPath.getRelative("WORKSPACE").getPathString(), "");

    StoredEventHandler eventHandler = new StoredEventHandler();
    SkyKey key = RepositoryDirectoryValue.key(RepositoryName.createUnvalidated("foo"));
    // Make it be evaluated every time, as we are testing evaluation.
    differencer.invalidate(ImmutableSet.of(key));
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(false)
            .setNumThreads(8)
            .setEventHandler(eventHandler)
            .build();
    EvaluationResult<SkyValue> result =
        evaluator.evaluate(ImmutableList.of(key), evaluationContext);
    assertThat(result.hasError()).isFalse();
    RepositoryDirectoryValue repositoryDirectoryValue = (RepositoryDirectoryValue) result.get(key);
    assertThat(repositoryDirectoryValue.repositoryExists()).isFalse();
    assertThat(repositoryDirectoryValue.getErrorMsg()).contains("Repository '@foo' is not defined");
  }

  @Test
  public void loadRepositoryFromBzlmod() throws Exception {
    scratch.overwriteFile(
        rootPath.getRelative("MODULE.bazel").getPathString(),
        "module(name='A',version='0.1')",
        "bazel_dep(name='B',version='1.0')");
    FakeRegistry registry =
        registryFactory
            .newFakeRegistry(scratch.dir("modules").getPathString())
            .addModule(createModuleKey("B", "1.0"), "module(name='B', version='1.0');");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));
    scratch.file(rootPath.getRelative("BUILD").getPathString());
    scratch.file(
        rootPath.getRelative("repo_rule.bzl").getPathString(),
        "def _impl(rctx):",
        " rctx.file('BUILD', '')",
        "fictive_repo_rule = repository_rule(implementation = _impl)");
    // WORKSPACE.bzlmod is preferred when Bzlmod is enabled.
    scratch.file(
        rootPath.getRelative("WORKSPACE.bzlmod").getPathString(),
        "load(':repo_rule.bzl', 'fictive_repo_rule')",
        "fictive_repo_rule(name = 'B.1.0')",
        "fictive_repo_rule(name = 'C')");
    scratch.file(
        rootPath.getRelative("WORKSPACE").getPathString(),
        "load(':repo_rule.bzl', 'fictive_repo_rule')",
        "fictive_repo_rule(name = 'B.1.0')");

    StoredEventHandler eventHandler = new StoredEventHandler();
    SkyKey key = RepositoryDirectoryValue.key(RepositoryName.createUnvalidated("B.1.0"));
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(false)
            .setNumThreads(8)
            .setEventHandler(eventHandler)
            .build();
    EvaluationResult<SkyValue> result =
        evaluator.evaluate(ImmutableList.of(key), evaluationContext);

    // B.1.0 should be fetched from MODULE.bazel file instead of WORKSPACE file.
    // Because FakeRegistry will look for the contents of B.1.0 under $scratch/modules/B.1.0 which
    // doesn't exist, the fetch should fail as expected.
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getException())
        .hasMessageThat()
        .contains("but it does not exist or is not a directory");

    // C should still be fetched from WORKSPACE.bzlmod successfully.
    loadRepo("C");
  }

  @Test
  public void loadInvisibleRepository() throws Exception {

    StoredEventHandler eventHandler = new StoredEventHandler();
    SkyKey key =
        RepositoryDirectoryValue.key(
            RepositoryName.createUnvalidated("foo").toNonVisible("fake_owner_repo"));
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(false)
            .setNumThreads(8)
            .setEventHandler(eventHandler)
            .build();
    EvaluationResult<SkyValue> result =
        evaluator.evaluate(ImmutableList.of(key), evaluationContext);

    assertThat(result.hasError()).isFalse();
    RepositoryDirectoryValue repositoryDirectoryValue = (RepositoryDirectoryValue) result.get(key);
    assertThat(repositoryDirectoryValue.repositoryExists()).isFalse();
    assertThat(repositoryDirectoryValue.getErrorMsg())
        .contains("Repository '@foo' is not visible from repository '@fake_owner_repo'");
  }

  private void loadRepo(String strippedRepoName) throws InterruptedException {
    StoredEventHandler eventHandler = new StoredEventHandler();
    SkyKey key = RepositoryDirectoryValue.key(RepositoryName.createUnvalidated(strippedRepoName));
    // Make it be evaluated every time, as we are testing evaluation.
    differencer.invalidate(ImmutableSet.of(key));
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(false)
            .setNumThreads(8)
            .setEventHandler(eventHandler)
            .build();
    EvaluationResult<SkyValue> result =
        evaluator.evaluate(ImmutableList.of(key), evaluationContext);
    assertThat(result.hasError()).isFalse();
    RepositoryDirectoryValue repositoryDirectoryValue = (RepositoryDirectoryValue) result.get(key);
    assertThat(repositoryDirectoryValue.repositoryExists()).isTrue();
  }

  private static class TestStarlarkRepositoryFunction extends StarlarkRepositoryFunction {
    private boolean fetchCalled = false;
    private final Path workspaceRoot;
    private final TestManagedDirectoriesKnowledge managedDirectoriesKnowledge;

    private TestStarlarkRepositoryFunction(
        Path workspaceRoot,
        DownloadManager downloader,
        TestManagedDirectoriesKnowledge managedDirectoriesKnowledge) {
      super(downloader);
      this.workspaceRoot = workspaceRoot;
      this.managedDirectoriesKnowledge = managedDirectoriesKnowledge;
    }

    public void reset() {
      fetchCalled = false;
    }

    private boolean isFetchCalled() {
      return fetchCalled;
    }

    @Nullable
    @Override
    public RepositoryDirectoryValue.Builder fetch(
        Rule rule,
        Path outputDirectory,
        BlazeDirectories directories,
        Environment env,
        Map<String, String> markerData,
        SkyKey key)
        throws RepositoryFunctionException, InterruptedException {
      fetchCalled = true;
      RepositoryDirectoryValue.Builder builder =
          super.fetch(rule, outputDirectory, directories, env, markerData, key);
      ImmutableSet<PathFragment> managedDirectories =
          managedDirectoriesKnowledge.getManagedDirectories((RepositoryName) key.argument());
      try {
        for (PathFragment managedDirectory : managedDirectories) {
          workspaceRoot.getRelative(managedDirectory).createDirectory();
        }
      } catch (IOException e) {
        throw new RepositoryFunctionException(e, Transience.PERSISTENT);
      }
      return builder;
    }
  }

  private static class TestManagedDirectoriesKnowledge implements ManagedDirectoriesKnowledge {

    private ImmutableMap<PathFragment, RepositoryName> map = ImmutableMap.of();

    public void setManagedDirectories(ImmutableMap<PathFragment, RepositoryName> map) {
      this.map = map;
    }

    @Nullable
    @Override
    public RepositoryName getOwnerRepository(PathFragment relativePathFragment) {
      return map.get(relativePathFragment);
    }

    @Override
    public ImmutableSet<PathFragment> getManagedDirectories(RepositoryName repositoryName) {
      return map.keySet().stream()
          .filter(path -> repositoryName.equals(map.get(path)))
          .collect(toImmutableSet());
    }

    @Override
    public boolean workspaceHeaderReloaded(
        @Nullable WorkspaceFileValue oldValue, @Nullable WorkspaceFileValue newValue) {
      throw new IllegalStateException();
    }
  }
}
