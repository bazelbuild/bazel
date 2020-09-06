// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.packages.BuildFileName;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryFunction;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryRule;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryLoaderFunction;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction.CrossRepositoryLabelViolationStrategy;
import com.google.devtools.build.lib.skyframe.PackageLookupValue.ErrorReason;
import com.google.devtools.build.lib.skyframe.PackageLookupValue.IncorrectRepositoryReferencePackageLookupValue;
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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

/** Tests for {@link PackageLookupFunction}. */
public abstract class PackageLookupFunctionTest extends FoundationTestCase {
  private AtomicReference<ImmutableSet<PackageIdentifier>> deletedPackages;
  private MemoizingEvaluator evaluator;
  private SequentialBuildDriver driver;
  private RecordingDifferencer differencer;
  private Path emptyPackagePath;
  private static final String IGNORED_PACKAGE_PREFIXES_FILE_PATH_STRING = "config/ignored.txt";

  protected abstract CrossRepositoryLabelViolationStrategy crossRepositoryLabelViolationStrategy();

  @Before
  public final void setUp() throws Exception {
    emptyPackagePath = rootDirectory.getRelative("somewhere/else");
    scratch.file("parentpackage/BUILD");

    AnalysisMock analysisMock = AnalysisMock.get();
    AtomicReference<PathPackageLocator> pkgLocator =
        new AtomicReference<>(
            new PathPackageLocator(
                outputBase,
                ImmutableList.of(Root.fromPath(emptyPackagePath), Root.fromPath(rootDirectory)),
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
            crossRepositoryLabelViolationStrategy(),
            BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY,
            BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER));
    skyFunctions.put(
        SkyFunctions.PACKAGE, new PackageFunction(null, null, null, null, null, null, null, null));
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
    skyFunctions.put(
        SkyFunctions.IGNORED_PACKAGE_PREFIXES,
        new IgnoredPackagePrefixesFunction(
            PathFragment.create(IGNORED_PACKAGE_PREFIXES_FILE_PATH_STRING)));
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
            /*bzlLoadFunctionForInlining=*/ null));
    skyFunctions.put(
        SkyFunctions.EXTERNAL_PACKAGE,
        new ExternalPackageFunction(BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER));
    skyFunctions.put(
        SkyFunctions.LOCAL_REPOSITORY_LOOKUP,
        new LocalRepositoryLookupFunction(BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER));
    skyFunctions.put(
        SkyFunctions.FILE_SYMLINK_CYCLE_UNIQUENESS, new FileSymlinkCycleUniquenessFunction());

    ImmutableMap<String, RepositoryFunction> repositoryHandlers =
        ImmutableMap.of(
            LocalRepositoryRule.NAME, (RepositoryFunction) new LocalRepositoryFunction());
    skyFunctions.put(
        SkyFunctions.REPOSITORY_DIRECTORY,
        new RepositoryDelegatorFunction(
            repositoryHandlers,
            null,
            new AtomicBoolean(true),
            ImmutableMap::of,
            directories,
            ManagedDirectoriesKnowledge.NO_MANAGED_DIRECTORIES,
            BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER));
    skyFunctions.put(SkyFunctions.REPOSITORY, new RepositoryLoaderFunction());

    differencer = new SequencedRecordingDifferencer();
    evaluator = new InMemoryMemoizingEvaluator(skyFunctions, differencer);
    driver = new SequentialBuildDriver(evaluator);
    PrecomputedValue.BUILD_ID.set(differencer, UUID.randomUUID());
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(differencer, pkgLocator.get());
    PrecomputedValue.STARLARK_SEMANTICS.set(differencer, StarlarkSemantics.DEFAULT);
    RepositoryDelegatorFunction.REPOSITORY_OVERRIDES.set(
        differencer, ImmutableMap.<RepositoryName, PathFragment>of());
    RepositoryDelegatorFunction.DEPENDENCY_FOR_UNCONDITIONAL_FETCHING.set(
        differencer, RepositoryDelegatorFunction.DONT_FETCH_UNCONDITIONALLY);
    RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE.set(
        differencer, Optional.empty());
  }

  protected PackageLookupValue lookupPackage(String packageName) throws InterruptedException {
    return lookupPackage(PackageIdentifier.createInMainRepo(packageName));
  }

  protected PackageLookupValue lookupPackage(PackageIdentifier packageId)
      throws InterruptedException {
    SkyKey key = PackageLookupValue.key(packageId);
    return lookupPackage(key).get(key);
  }

  protected EvaluationResult<PackageLookupValue> lookupPackage(SkyKey packageIdentifierSkyKey)
      throws InterruptedException {
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(false)
            .setNumThreads(SkyframeExecutor.DEFAULT_THREAD_COUNT)
            .setEventHandler(NullEventHandler.INSTANCE)
            .build();
    return driver.<PackageLookupValue>evaluate(
        ImmutableList.of(packageIdentifierSkyKey), evaluationContext);
  }

  @Test
  public void testNoBuildFile() throws Exception {
    scratch.file("parentpackage/nobuildfile/foo.txt");
    PackageLookupValue packageLookupValue = lookupPackage("parentpackage/nobuildfile");
    assertThat(packageLookupValue.packageExists()).isFalse();
    assertThat(packageLookupValue.getErrorReason()).isEqualTo(ErrorReason.NO_BUILD_FILE);
    assertThat(packageLookupValue.getErrorMsg()).isNotNull();
  }

  @Test
  public void testNoBuildFileAndNoParentPackage() throws Exception {
    scratch.file("noparentpackage/foo.txt");
    PackageLookupValue packageLookupValue = lookupPackage("noparentpackage");
    assertThat(packageLookupValue.packageExists()).isFalse();
    assertThat(packageLookupValue.getErrorReason()).isEqualTo(ErrorReason.NO_BUILD_FILE);
    assertThat(packageLookupValue.getErrorMsg()).isNotNull();
  }

  @Test
  public void testDeletedPackage() throws Exception {
    scratch.file("parentpackage/deletedpackage/BUILD");
    deletedPackages.set(ImmutableSet.of(
        PackageIdentifier.createInMainRepo("parentpackage/deletedpackage")));
    PackageLookupValue packageLookupValue = lookupPackage("parentpackage/deletedpackage");
    assertThat(packageLookupValue.packageExists()).isFalse();
    assertThat(packageLookupValue.getErrorReason()).isEqualTo(ErrorReason.DELETED_PACKAGE);
    assertThat(packageLookupValue.getErrorMsg()).isNotNull();
  }

  @Test
  public void testIgnoredPackage() throws Exception {
    scratch.file("ignored/subdir/BUILD");
    scratch.file("ignored/BUILD");
    Path ignored = scratch.overwriteFile(IGNORED_PACKAGE_PREFIXES_FILE_PATH_STRING, "ignored");

    ImmutableSet<String> pkgs = ImmutableSet.of("ignored/subdir", "ignored");
    for (String pkg : pkgs) {
      PackageLookupValue packageLookupValue = lookupPackage(pkg);
      assertThat(packageLookupValue.packageExists()).isFalse();
      assertThat(packageLookupValue.getErrorReason()).isEqualTo(ErrorReason.DELETED_PACKAGE);
      assertThat(packageLookupValue.getErrorMsg()).isNotNull();
    }

    scratch.overwriteFile(IGNORED_PACKAGE_PREFIXES_FILE_PATH_STRING, "not_ignored");
    RootedPath rootedIgnoreFile =
        RootedPath.toRootedPath(
            Root.fromPath(ignored.getParentDirectory().getParentDirectory()),
            PathFragment.create("config/ignored.txt"));
    differencer.invalidate(ImmutableSet.of(FileStateValue.key(rootedIgnoreFile)));
    for (String pkg : pkgs) {
      PackageLookupValue packageLookupValue = lookupPackage(pkg);
      assertThat(packageLookupValue.packageExists()).isTrue();
    }
  }

  @Test
  public void testInvalidPackageName() throws Exception {
    scratch.file("parentpackage/invalidpackagename:42/BUILD");
    PackageLookupValue packageLookupValue = lookupPackage("parentpackage/invalidpackagename:42");
    assertThat(packageLookupValue.packageExists()).isFalse();
    assertThat(packageLookupValue.getErrorReason()).isEqualTo(ErrorReason.INVALID_PACKAGE_NAME);
    assertThat(packageLookupValue.getErrorMsg()).isNotNull();
  }

  @Test
  public void testDirectoryNamedBuild() throws Exception {
    scratch.dir("parentpackage/isdirectory/BUILD");
    PackageLookupValue packageLookupValue = lookupPackage("parentpackage/isdirectory");
    assertThat(packageLookupValue.packageExists()).isFalse();
    assertThat(packageLookupValue.getErrorReason()).isEqualTo(ErrorReason.NO_BUILD_FILE);
    assertThat(packageLookupValue.getErrorMsg()).isNotNull();
  }

  @Test
  public void testEverythingIsGood_BUILD() throws Exception {
    scratch.file("parentpackage/everythinggood/BUILD");
    PackageLookupValue packageLookupValue = lookupPackage("parentpackage/everythinggood");
    assertThat(packageLookupValue.packageExists()).isTrue();
    assertThat(packageLookupValue.getRoot()).isEqualTo(Root.fromPath(rootDirectory));
    assertThat(packageLookupValue.getBuildFileName()).isEqualTo(BuildFileName.BUILD);
  }

  @Test
  public void testEverythingIsGood_BUILD_bazel() throws Exception {
    scratch.file("parentpackage/everythinggood/BUILD.bazel");
    PackageLookupValue packageLookupValue = lookupPackage("parentpackage/everythinggood");
    assertThat(packageLookupValue.packageExists()).isTrue();
    assertThat(packageLookupValue.getRoot()).isEqualTo(Root.fromPath(rootDirectory));
    assertThat(packageLookupValue.getBuildFileName()).isEqualTo(BuildFileName.BUILD_DOT_BAZEL);
  }

  @Test
  public void testEverythingIsGood_both() throws Exception {
    scratch.file("parentpackage/everythinggood/BUILD");
    scratch.file("parentpackage/everythinggood/BUILD.bazel");
    PackageLookupValue packageLookupValue = lookupPackage("parentpackage/everythinggood");
    assertThat(packageLookupValue.packageExists()).isTrue();
    assertThat(packageLookupValue.getRoot()).isEqualTo(Root.fromPath(rootDirectory));
    assertThat(packageLookupValue.getBuildFileName()).isEqualTo(BuildFileName.BUILD_DOT_BAZEL);
  }

  @Test
  public void testBuildFilesInMultiplePackagePaths() throws Exception {
    scratch.file(emptyPackagePath.getPathString() + "/foo/BUILD");
    scratch.file("foo/BUILD.bazel");

    // BUILD file in the first package path should be preferred to BUILD.bazel in the second.
    PackageLookupValue packageLookupValue = lookupPackage("foo");
    assertThat(packageLookupValue.packageExists()).isTrue();
    assertThat(packageLookupValue.getRoot()).isEqualTo(Root.fromPath(emptyPackagePath));
    assertThat(packageLookupValue.getBuildFileName()).isEqualTo(BuildFileName.BUILD);
  }

  @Test
  public void testEmptyPackageName() throws Exception {
    scratch.file("BUILD");
    PackageLookupValue packageLookupValue = lookupPackage("");
    assertThat(packageLookupValue.packageExists()).isTrue();
    assertThat(packageLookupValue.getRoot()).isEqualTo(Root.fromPath(rootDirectory));
    assertThat(packageLookupValue.getBuildFileName()).isEqualTo(BuildFileName.BUILD);
  }

  @Test
  public void testWorkspaceLookup() throws Exception {
    scratch.overwriteFile("WORKSPACE");
    PackageLookupValue packageLookupValue = lookupPackage(
        PackageIdentifier.createInMainRepo("external"));
    assertThat(packageLookupValue.packageExists()).isTrue();
    assertThat(packageLookupValue.getRoot()).isEqualTo(Root.fromPath(rootDirectory));
  }

  @Test
  public void testPackageLookupValueHashCodeAndEqualsContract() throws Exception {
    Root root1 = Root.fromPath(rootDirectory.getRelative("root1"));
    Root root2 = Root.fromPath(rootDirectory.getRelative("root2"));
    // Our (seeming) duplication of parameters here is intentional. Some of the subclasses of
    // PackageLookupValue are supposed to have reference equality semantics, and some are supposed
    // to have logical equality semantics.
    new EqualsTester()
        .addEqualityGroup(
            PackageLookupValue.success(root1, BuildFileName.BUILD),
            PackageLookupValue.success(root1, BuildFileName.BUILD))
        .addEqualityGroup(
            PackageLookupValue.success(root2, BuildFileName.BUILD),
            PackageLookupValue.success(root2, BuildFileName.BUILD))
        .addEqualityGroup(
            PackageLookupValue.NO_BUILD_FILE_VALUE, PackageLookupValue.NO_BUILD_FILE_VALUE)
        .addEqualityGroup(
            PackageLookupValue.DELETED_PACKAGE_VALUE, PackageLookupValue.DELETED_PACKAGE_VALUE)
        .addEqualityGroup(
            PackageLookupValue.invalidPackageName("nope1"),
            PackageLookupValue.invalidPackageName("nope1"))
        .addEqualityGroup(
            PackageLookupValue.invalidPackageName("nope2"),
            PackageLookupValue.invalidPackageName("nope2"))
        .testEquals();
  }

  protected void createAndCheckInvalidPackageLabel(boolean expectedPackageExists) throws Exception {
    scratch.overwriteFile("WORKSPACE", "local_repository(name='local', path='local/repo')");
    scratch.file("local/repo/WORKSPACE");
    scratch.file("local/repo/BUILD");

    // First, use the correct label.
    PackageLookupValue packageLookupValue =
        lookupPackage(PackageIdentifier.create("@local", PathFragment.EMPTY_FRAGMENT));
    assertThat(packageLookupValue.packageExists()).isTrue();

    // Then, use the incorrect label.
    packageLookupValue = lookupPackage(PackageIdentifier.createInMainRepo("local/repo"));
    assertThat(packageLookupValue.packageExists()).isEqualTo(expectedPackageExists);
  }

  /**
   * Runs all tests in the base {@link PackageLookupFunctionTest} class with the {@link
   * CrossRepositoryLabelViolationStrategy#IGNORE} enum set, and also additional tests specific to
   * that setting.
   */
  @RunWith(JUnit4.class)
  public static class IgnoreLabelViolationsTest extends PackageLookupFunctionTest {
    @Override
    protected CrossRepositoryLabelViolationStrategy crossRepositoryLabelViolationStrategy() {
      return CrossRepositoryLabelViolationStrategy.IGNORE;
    }

    // Add any ignore-specific tests here.

    @Test
    public void testInvalidPackageLabelIsIgnored() throws Exception {
      createAndCheckInvalidPackageLabel(true);
    }
  }

  /**
   * Runs all tests in the base {@link PackageLookupFunctionTest} class with the {@link
   * CrossRepositoryLabelViolationStrategy#ERROR} enum set, and also additional tests specific to
   * that setting.
   */
  @RunWith(JUnit4.class)
  public static class ErrorLabelViolationsTest extends PackageLookupFunctionTest {
    @Override
    protected CrossRepositoryLabelViolationStrategy crossRepositoryLabelViolationStrategy() {
      return CrossRepositoryLabelViolationStrategy.ERROR;
    }

    // Add any error-specific tests here.

    @Test
    public void testInvalidPackageLabelIsError() throws Exception {
      createAndCheckInvalidPackageLabel(false);
    }

    @Test
    public void testSymlinkCycleInWorkspace() throws Exception {
      scratch.overwriteFile("WORKSPACE", "local_repository(name='local', path='local/repo')");
      Path localRepoWorkspace = scratch.resolve("local/repo/WORKSPACE");
      Path localRepoWorkspaceLink = scratch.resolve("local/repo/WORKSPACE.link");
      FileSystemUtils.createDirectoryAndParents(localRepoWorkspace.getParentDirectory());
      FileSystemUtils.createDirectoryAndParents(localRepoWorkspaceLink.getParentDirectory());
      localRepoWorkspace.createSymbolicLink(localRepoWorkspaceLink);
      localRepoWorkspaceLink.createSymbolicLink(localRepoWorkspace);
      scratch.file("local/repo/BUILD");

      SkyKey skyKey = PackageLookupValue.key(PackageIdentifier.createInMainRepo("local/repo"));
      EvaluationResult<PackageLookupValue> result = lookupPackage(skyKey);
      assertThatEvaluationResult(result)
          .hasErrorEntryForKeyThat(skyKey)
          .hasExceptionThat()
          .isInstanceOf(BuildFileNotFoundException.class);
      assertThatEvaluationResult(result)
          .hasErrorEntryForKeyThat(skyKey)
          .hasExceptionThat()
          .hasMessageThat()
          .isEqualTo(
              "no such package 'local/repo': Unable to determine the local repository for "
                  + "directory /workspace/local/repo");
    }
  }

  /** Tests for detection of invalid package identifiers for local repositories. */
  @RunWith(Parameterized.class)
  public static class CorrectedLocalRepositoryTest extends PackageLookupFunctionTest {

    /**
     * Create parameters for this test. The contents are:
     *
     * <ol>
     *   <li>description
     *   <li>repository path
     *   <li>package path - under the repository
     *   <li>expected corrected package identifier
     * </ol>
     */
    @Parameters(name = "{0}")
    public static List<Object[]> parameters() {
      List<Object[]> params = new ArrayList<>();

      params.add(new String[] {"simpleRepo_emptyPackage", "local", "", "@local//"});
      params.add(new String[] {"simpleRepo_singlePackage", "local", "package", "@local//package"});
      params.add(
          new String[] {
            "simpleRepo_subPackage", "local", "package/subpackage", "@local//package/subpackage"
          });
      params.add(new String[] {"deepRepo_emptyPackage", "local/repo", "", "@local//"});
      params.add(new String[] {"deepRepo_subPackage", "local/repo", "package", "@local//package"});

      return params;
    }

    private final String repositoryPath;
    private final String packagePath;
    private final String expectedCorrectedPackageIdentifier;

    public CorrectedLocalRepositoryTest(
        String unusedDescription,
        String repositoryPath,
        String packagePath,
        String expectedCorrectedPackageIdentifier) {
      this.repositoryPath = repositoryPath;
      this.packagePath = packagePath;
      this.expectedCorrectedPackageIdentifier = expectedCorrectedPackageIdentifier;
    }

    @Override
    protected CrossRepositoryLabelViolationStrategy crossRepositoryLabelViolationStrategy() {
      return CrossRepositoryLabelViolationStrategy.ERROR;
    }

    @Test
    public void testCorrectPackageDetection_relativePath() throws Exception {
      scratch.overwriteFile(
          "WORKSPACE", "local_repository(name='local', path='" + repositoryPath + "')");
      scratch.file(PathFragment.create(repositoryPath).getRelative("WORKSPACE").getPathString());
      scratch.file(
          PathFragment.create(repositoryPath)
              .getRelative(packagePath)
              .getRelative("BUILD")
              .getPathString());

      PackageIdentifier packageIdentifier =
          PackageIdentifier.createInMainRepo(
              PathFragment.create(repositoryPath).getRelative(packagePath));
      PackageLookupValue packageLookupValue = lookupPackage(packageIdentifier);
      assertThat(packageLookupValue.packageExists()).isFalse();
      assertThat(packageLookupValue)
          .isInstanceOf(IncorrectRepositoryReferencePackageLookupValue.class);

      IncorrectRepositoryReferencePackageLookupValue incorrectPackageLookupValue =
          (IncorrectRepositoryReferencePackageLookupValue) packageLookupValue;
      assertThat(incorrectPackageLookupValue.getInvalidPackageIdentifier())
          .isEqualTo(packageIdentifier);
      assertThat(incorrectPackageLookupValue.getCorrectedPackageIdentifier().toString())
          .isEqualTo(expectedCorrectedPackageIdentifier);
    }

    @Test
    public void testCorrectPackageDetection_absolutePath() throws Exception {
      scratch.overwriteFile(
          "WORKSPACE",
          "local_repository(name='local', path=__workspace_dir__ + '/" + repositoryPath + "')");
      scratch.file(PathFragment.create(repositoryPath).getRelative("WORKSPACE").getPathString());
      scratch.file(
          PathFragment.create(repositoryPath)
              .getRelative(packagePath)
              .getRelative("BUILD")
              .getPathString());

      PackageIdentifier packageIdentifier =
          PackageIdentifier.createInMainRepo(
              PathFragment.create(repositoryPath).getRelative(packagePath));
      PackageLookupValue packageLookupValue = lookupPackage(packageIdentifier);
      assertThat(packageLookupValue.packageExists()).isFalse();
      assertThat(packageLookupValue)
          .isInstanceOf(IncorrectRepositoryReferencePackageLookupValue.class);

      IncorrectRepositoryReferencePackageLookupValue incorrectPackageLookupValue =
          (IncorrectRepositoryReferencePackageLookupValue) packageLookupValue;
      assertThat(incorrectPackageLookupValue.getInvalidPackageIdentifier())
          .isEqualTo(packageIdentifier);
      assertThat(incorrectPackageLookupValue.getCorrectedPackageIdentifier().toString())
          .isEqualTo(expectedCorrectedPackageIdentifier);
    }
  }
}
