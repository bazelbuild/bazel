// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction.CrossRepositoryLabelViolationStrategy;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SequencedRecordingDifferencer;
import com.google.devtools.build.skyframe.SequentialBuildDriver;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link LocalRepositoryLookupFunction}. */
@RunWith(JUnit4.class)
public class LocalRepositoryLookupFunctionTest extends FoundationTestCase {
  private AtomicReference<ImmutableSet<PackageIdentifier>> deletedPackages;
  private MemoizingEvaluator evaluator;
  private SequentialBuildDriver driver;
  private RecordingDifferencer differencer;

  @Before
  public final void setUp() throws Exception {
    AnalysisMock analysisMock = AnalysisMock.get();
    AtomicReference<PathPackageLocator> pkgLocator =
        new AtomicReference<>(
            new PathPackageLocator(
                outputBase,
                ImmutableList.of(Root.fromPath(rootDirectory)),
                BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY));
    deletedPackages = new AtomicReference<>(ImmutableSet.<PackageIdentifier>of());
    BlazeDirectories directories =
        new BlazeDirectories(
            new ServerDirectories(rootDirectory, outputBase, rootDirectory),
            rootDirectory,
            /* defaultSystemJavabase= */ null,
            analysisMock.getProductName());
    ExternalFilesHelper externalFilesHelper = ExternalFilesHelper.createForTesting(
        pkgLocator, ExternalFileAction.DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_PATHS, directories);

    Map<SkyFunctionName, SkyFunction> skyFunctions = new HashMap<>();
    skyFunctions.put(
        SkyFunctions.PACKAGE_LOOKUP,
        new PackageLookupFunction(
            deletedPackages,
            CrossRepositoryLabelViolationStrategy.ERROR,
            BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY,
            BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER));
    skyFunctions.put(
        FileStateValue.FILE_STATE,
        new FileStateFunction(
            new AtomicReference<TimestampGranularityMonitor>(),
            new AtomicReference<>(UnixGlob.DEFAULT_SYSCALLS),
            externalFilesHelper));
    skyFunctions.put(FileValue.FILE, new FileFunction(pkgLocator));
    skyFunctions.put(SkyFunctions.DIRECTORY_LISTING, new DirectoryListingFunction());
    skyFunctions.put(
        SkyFunctions.DIRECTORY_LISTING_STATE,
        new DirectoryListingStateFunction(
            externalFilesHelper, new AtomicReference<>(UnixGlob.DEFAULT_SYSCALLS)));
    RuleClassProvider ruleClassProvider = analysisMock.createRuleClassProvider();
    skyFunctions.put(SkyFunctions.WORKSPACE_AST, new WorkspaceASTFunction(ruleClassProvider));
    skyFunctions.put(
        WorkspaceFileValue.WORKSPACE_FILE,
        new WorkspaceFileFunction(
            ruleClassProvider,
            analysisMock
                .getPackageFactoryBuilderForTesting(directories)
                .build(ruleClassProvider, fileSystem),
            directories,
            /*starlarkImportLookupFunctionForInlining=*/ null));
    skyFunctions.put(
        SkyFunctions.EXTERNAL_PACKAGE,
        new ExternalPackageFunction(BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER));
    skyFunctions.put(
        SkyFunctions.LOCAL_REPOSITORY_LOOKUP,
        new LocalRepositoryLookupFunction(BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER));
    skyFunctions.put(
        SkyFunctions.FILE_SYMLINK_CYCLE_UNIQUENESS, new FileSymlinkCycleUniquenessFunction());

    differencer = new SequencedRecordingDifferencer();
    evaluator = new InMemoryMemoizingEvaluator(skyFunctions, differencer);
    driver = new SequentialBuildDriver(evaluator);
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(differencer, pkgLocator.get());
    PrecomputedValue.STARLARK_SEMANTICS.set(differencer, StarlarkSemantics.DEFAULT);
    RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE.set(
        differencer, Optional.<RootedPath>absent());
  }

  private static SkyKey createKey(RootedPath directory) {
    return LocalRepositoryLookupValue.key(directory);
  }

  private LocalRepositoryLookupValue lookupDirectory(RootedPath directory)
      throws InterruptedException {
    SkyKey key = createKey(directory);
    return lookupDirectory(key).get(key);
  }

  private EvaluationResult<LocalRepositoryLookupValue> lookupDirectory(SkyKey directoryKey)
      throws InterruptedException {
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(false)
            .setNumThreads(SkyframeExecutor.DEFAULT_THREAD_COUNT)
            .setEventHander(NullEventHandler.INSTANCE)
            .build();
    return driver.<LocalRepositoryLookupValue>evaluate(
        ImmutableList.of(directoryKey), evaluationContext);
  }

  @Test
  public void testNoPath() throws Exception {
    LocalRepositoryLookupValue repositoryLookupValue =
        lookupDirectory(
            RootedPath.toRootedPath(Root.fromPath(rootDirectory), PathFragment.EMPTY_FRAGMENT));
    assertThat(repositoryLookupValue).isNotNull();
    assertThat(repositoryLookupValue.getRepository()).isEqualTo(RepositoryName.MAIN);
    assertThat(repositoryLookupValue.getPath()).isEqualTo(PathFragment.EMPTY_FRAGMENT);
  }

  @Test
  public void testActualPackage() throws Exception {
    scratch.file("some/path/BUILD");

    LocalRepositoryLookupValue repositoryLookupValue =
        lookupDirectory(
            RootedPath.toRootedPath(
                Root.fromPath(rootDirectory), PathFragment.create("some/path")));
    assertThat(repositoryLookupValue).isNotNull();
    assertThat(repositoryLookupValue.getRepository()).isEqualTo(RepositoryName.MAIN);
    assertThat(repositoryLookupValue.getPath()).isEqualTo(PathFragment.EMPTY_FRAGMENT);
  }

  @Test
  public void testLocalRepository() throws Exception {
    scratch.overwriteFile("WORKSPACE", "local_repository(name='local', path='local/repo')");
    scratch.file("local/repo/WORKSPACE");
    scratch.file("local/repo/BUILD");

    LocalRepositoryLookupValue repositoryLookupValue =
        lookupDirectory(
            RootedPath.toRootedPath(
                Root.fromPath(rootDirectory), PathFragment.create("local/repo")));
    assertThat(repositoryLookupValue).isNotNull();
    assertThat(repositoryLookupValue.getRepository().getName()).isEqualTo("@local");
    assertThat(repositoryLookupValue.getPath()).isEqualTo(PathFragment.create("local/repo"));
  }

  @Test
  public void testLocalRepository_absolutePath() throws Exception {
    scratch.overwriteFile("WORKSPACE", "local_repository(name='local', path='/abs/local/repo')");
    scratch.file("/abs/local/repo/WORKSPACE");
    scratch.file("/abs/local/repo/BUILD");

    LocalRepositoryLookupValue repositoryLookupValue =
        lookupDirectory(
            RootedPath.toRootedPath(
                Root.fromPath(rootDirectory.getRelative("/abs")),
                PathFragment.create("local/repo")));
    assertThat(repositoryLookupValue).isNotNull();
    assertThat(repositoryLookupValue.getRepository().getName()).isEqualTo("@local");
    assertThat(repositoryLookupValue.getPath()).isEqualTo(PathFragment.create("/abs/local/repo"));
  }

  @Test
  public void testLocalRepository_nonNormalizedPath() throws Exception {
    scratch.overwriteFile("WORKSPACE", "local_repository(name='local', path='./local/repo')");
    scratch.file("local/repo/WORKSPACE");
    scratch.file("local/repo/BUILD");

    LocalRepositoryLookupValue repositoryLookupValue =
        lookupDirectory(
            RootedPath.toRootedPath(
                Root.fromPath(rootDirectory), PathFragment.create("local/repo")));
    assertThat(repositoryLookupValue).isNotNull();
    assertThat(repositoryLookupValue.getRepository().getName()).isEqualTo("@local");
    assertThat(repositoryLookupValue.getPath()).isEqualTo(PathFragment.create("local/repo"));
  }

  @Test
  public void testLocalRepository_absolutePath_nonNormalized() throws Exception {
    scratch.overwriteFile("WORKSPACE", "local_repository(name='local', path='/abs/local/./repo')");
    scratch.file("/abs/local/repo/WORKSPACE");
    scratch.file("/abs/local/repo/BUILD");

    LocalRepositoryLookupValue repositoryLookupValue =
        lookupDirectory(
            RootedPath.toRootedPath(
                Root.fromPath(rootDirectory.getRelative("/abs")),
                PathFragment.create("local/repo")));
    assertThat(repositoryLookupValue).isNotNull();
    assertThat(repositoryLookupValue.getRepository().getName()).isEqualTo("@local");
    assertThat(repositoryLookupValue.getPath()).isEqualTo(PathFragment.create("/abs/local/repo"));
  }

  @Test
  public void testLocalRepositorySubPackage() throws Exception {
    scratch.overwriteFile("WORKSPACE", "local_repository(name='local', path='local/repo')");
    scratch.file("local/repo/WORKSPACE");
    scratch.file("local/repo/BUILD");
    scratch.file("local/repo/sub/package/BUILD");

    LocalRepositoryLookupValue repositoryLookupValue =
        lookupDirectory(
            RootedPath.toRootedPath(
                Root.fromPath(rootDirectory), PathFragment.create("local/repo/sub/package")));
    assertThat(repositoryLookupValue).isNotNull();
    assertThat(repositoryLookupValue.getRepository().getName()).isEqualTo("@local");
    assertThat(repositoryLookupValue.getPath()).isEqualTo(PathFragment.create("local/repo"));
  }

  @Test
  public void testWorkspaceButNoLocalRepository() throws Exception {
    scratch.overwriteFile("WORKSPACE", "");
    scratch.file("local/repo/WORKSPACE");
    scratch.file("local/repo/BUILD");

    LocalRepositoryLookupValue repositoryLookupValue =
        lookupDirectory(
            RootedPath.toRootedPath(
                Root.fromPath(rootDirectory), PathFragment.create("local/repo")));
    assertThat(repositoryLookupValue).isNotNull();
    assertThat(repositoryLookupValue.getRepository()).isEqualTo(RepositoryName.MAIN);
    assertThat(repositoryLookupValue.getPath()).isEqualTo(PathFragment.EMPTY_FRAGMENT);
  }

  @Test
  public void testLocalRepository_LocalWorkspace_SymlinkCycle() throws Exception {
    scratch.overwriteFile("WORKSPACE", "local_repository(name='local', path='local/repo')");
    Path localRepoWorkspace = scratch.resolve("local/repo/WORKSPACE");
    Path localRepoWorkspaceLink = scratch.resolve("local/repo/WORKSPACE.link");
    FileSystemUtils.createDirectoryAndParents(localRepoWorkspace.getParentDirectory());
    FileSystemUtils.createDirectoryAndParents(localRepoWorkspaceLink.getParentDirectory());
    localRepoWorkspace.createSymbolicLink(localRepoWorkspaceLink);
    localRepoWorkspaceLink.createSymbolicLink(localRepoWorkspace);
    scratch.file("local/repo/BUILD");

    SkyKey localRepositoryKey =
        createKey(
            RootedPath.toRootedPath(
                Root.fromPath(rootDirectory), PathFragment.create("local/repo")));
    EvaluationResult<LocalRepositoryLookupValue> result = lookupDirectory(localRepositoryKey);

    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(localRepositoryKey)
        .hasExceptionThat()
        .hasMessageThat()
        .isEqualTo(
            "FileSymlinkException while checking if there is a WORKSPACE file in "
                + "/workspace/local/repo");
  }

  @Test
  public void testLocalRepository_MainWorkspace_NotFound() throws Exception {
    // Do not add a local_repository to WORKSPACE.
    scratch.overwriteFile("WORKSPACE", "");
    scratch.deleteFile("WORKSPACE");
    scratch.file("local/repo/WORKSPACE");
    scratch.file("local/repo/BUILD");

    LocalRepositoryLookupValue repositoryLookupValue =
        lookupDirectory(
            RootedPath.toRootedPath(
                Root.fromPath(rootDirectory), PathFragment.create("local/repo")));
    assertThat(repositoryLookupValue).isNotNull();
    // In this case, the repository should be MAIN as we can't find any local_repository rules.
    assertThat(repositoryLookupValue.getRepository()).isEqualTo(RepositoryName.MAIN);
    assertThat(repositoryLookupValue.getPath()).isEqualTo(PathFragment.EMPTY_FRAGMENT);
  }

  // TODO(katre): Add tests for the following exceptions
  // While reading dir/WORKSPACE:
  // - IOException
  // - FileSymlinkException
  // - InconsistentFilesystemException
  // While loading //external
  // - BuildFileNotFoundException
  // - InconsistentFilesystemException
  // While reading //external:WORKSPACE
  // - PackageFunctionException
  // - NameConflictException
  // - WorkspaceFileException
}
