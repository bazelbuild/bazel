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

import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.bazel.rules.BazelRulesModule;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryFunction;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryRule;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryLoaderFunction;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction.CrossRepositoryLabelViolationStrategy;
import com.google.devtools.build.lib.skyframe.PackageLookupValue.BuildFileName;
import com.google.devtools.build.lib.skyframe.PackageLookupValue.ErrorReason;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SequentialBuildDriver;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PackageLookupFunction}. */
public abstract class PackageLookupFunctionTest extends FoundationTestCase {
  private AtomicReference<ImmutableSet<PackageIdentifier>> deletedPackages;
  private MemoizingEvaluator evaluator;
  private SequentialBuildDriver driver;
  private RecordingDifferencer differencer;

  protected abstract CrossRepositoryLabelViolationStrategy crossRepositoryLabelViolationStrategy();

  @Before
  public final void setUp() throws Exception {
    Path emptyPackagePath = rootDirectory.getRelative("somewhere/else");
    scratch.file("parentpackage/BUILD");

    AnalysisMock analysisMock = AnalysisMock.get();
    AtomicReference<PathPackageLocator> pkgLocator = new AtomicReference<>(
        new PathPackageLocator(outputBase, ImmutableList.of(emptyPackagePath, rootDirectory)));
    deletedPackages = new AtomicReference<>(ImmutableSet.<PackageIdentifier>of());
    BlazeDirectories directories =
        new BlazeDirectories(
            rootDirectory, outputBase, rootDirectory, analysisMock.getProductName());
    ExternalFilesHelper externalFilesHelper = new ExternalFilesHelper(
        pkgLocator, ExternalFileAction.DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_PATHS, directories);

    Map<SkyFunctionName, SkyFunction> skyFunctions = new HashMap<>();
    skyFunctions.put(
        SkyFunctions.PACKAGE_LOOKUP,
        new PackageLookupFunction(deletedPackages, crossRepositoryLabelViolationStrategy()));
    skyFunctions.put(
        SkyFunctions.PACKAGE,
        new PackageFunction(null, null, null, null, null, null, null));
    skyFunctions.put(SkyFunctions.FILE_STATE, new FileStateFunction(
        new AtomicReference<TimestampGranularityMonitor>(), externalFilesHelper));
    skyFunctions.put(SkyFunctions.FILE, new FileFunction(pkgLocator));
    skyFunctions.put(SkyFunctions.DIRECTORY_LISTING, new DirectoryListingFunction());
    skyFunctions.put(
        SkyFunctions.DIRECTORY_LISTING_STATE,
        new DirectoryListingStateFunction(externalFilesHelper));
    skyFunctions.put(SkyFunctions.BLACKLISTED_PACKAGE_PREFIXES,
        new BlacklistedPackagePrefixesFunction());
    RuleClassProvider ruleClassProvider = analysisMock.createRuleClassProvider();
    skyFunctions.put(SkyFunctions.WORKSPACE_AST, new WorkspaceASTFunction(ruleClassProvider));
    skyFunctions.put(
        SkyFunctions.WORKSPACE_FILE,
        new WorkspaceFileFunction(
            ruleClassProvider,
            analysisMock
                .getPackageFactoryForTesting()
                .create(
                    ruleClassProvider,
                    new BazelRulesModule().getPackageEnvironmentExtension(),
                    scratch.getFileSystem()),
            directories));
    skyFunctions.put(SkyFunctions.EXTERNAL_PACKAGE, new ExternalPackageFunction());
    skyFunctions.put(SkyFunctions.LOCAL_REPOSITORY_LOOKUP, new LocalRepositoryLookupFunction());
    skyFunctions.put(
        SkyFunctions.FILE_SYMLINK_CYCLE_UNIQUENESS, new FileSymlinkCycleUniquenessFunction());

    ImmutableMap<String, RepositoryFunction> repositoryHandlers =
        ImmutableMap.of(
            LocalRepositoryRule.NAME, (RepositoryFunction) new LocalRepositoryFunction());
    skyFunctions.put(
        SkyFunctions.REPOSITORY_DIRECTORY,
        new RepositoryDelegatorFunction(repositoryHandlers, null, new AtomicBoolean(true)));
    skyFunctions.put(SkyFunctions.REPOSITORY, new RepositoryLoaderFunction());

    differencer = new RecordingDifferencer();
    evaluator = new InMemoryMemoizingEvaluator(skyFunctions, differencer);
    driver = new SequentialBuildDriver(evaluator);
    PrecomputedValue.BUILD_ID.set(differencer, UUID.randomUUID());
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(differencer, pkgLocator.get());
    PrecomputedValue.BLACKLISTED_PACKAGE_PREFIXES_FILE.set(
        differencer, PathFragment.EMPTY_FRAGMENT);
    PrecomputedValue.BLAZE_DIRECTORIES.set(differencer, directories);
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
    return driver.<PackageLookupValue>evaluate(
        ImmutableList.of(packageIdentifierSkyKey),
        false,
        SkyframeExecutor.DEFAULT_THREAD_COUNT,
        NullEventHandler.INSTANCE);
  }

  @Test
  public void testNoBuildFile() throws Exception {
    scratch.file("parentpackage/nobuildfile/foo.txt");
    PackageLookupValue packageLookupValue = lookupPackage("parentpackage/nobuildfile");
    assertFalse(packageLookupValue.packageExists());
    assertEquals(ErrorReason.NO_BUILD_FILE, packageLookupValue.getErrorReason());
    assertNotNull(packageLookupValue.getErrorMsg());
  }

  @Test
  public void testNoBuildFileAndNoParentPackage() throws Exception {
    scratch.file("noparentpackage/foo.txt");
    PackageLookupValue packageLookupValue = lookupPackage("noparentpackage");
    assertFalse(packageLookupValue.packageExists());
    assertEquals(ErrorReason.NO_BUILD_FILE, packageLookupValue.getErrorReason());
    assertNotNull(packageLookupValue.getErrorMsg());
  }

  @Test
  public void testDeletedPackage() throws Exception {
    scratch.file("parentpackage/deletedpackage/BUILD");
    deletedPackages.set(ImmutableSet.of(
        PackageIdentifier.createInMainRepo("parentpackage/deletedpackage")));
    PackageLookupValue packageLookupValue = lookupPackage("parentpackage/deletedpackage");
    assertFalse(packageLookupValue.packageExists());
    assertEquals(ErrorReason.DELETED_PACKAGE, packageLookupValue.getErrorReason());
    assertNotNull(packageLookupValue.getErrorMsg());
  }


  @Test
  public void testBlacklistedPackage() throws Exception {
    scratch.file("blacklisted/subdir/BUILD");
    scratch.file("blacklisted/BUILD");
    PrecomputedValue.BLACKLISTED_PACKAGE_PREFIXES_FILE.set(differencer,
        new PathFragment("config/blacklisted.txt"));
    Path blacklist = scratch.file("config/blacklisted.txt", "blacklisted");

    ImmutableSet<String> pkgs = ImmutableSet.of("blacklisted/subdir", "blacklisted");
    for (String pkg : pkgs) {
      PackageLookupValue packageLookupValue = lookupPackage(pkg);
      assertFalse(packageLookupValue.packageExists());
      assertEquals(ErrorReason.DELETED_PACKAGE, packageLookupValue.getErrorReason());
      assertNotNull(packageLookupValue.getErrorMsg());
    }

    scratch.overwriteFile("config/blacklisted.txt", "not_blacklisted");
    RootedPath rootedBlacklist = RootedPath.toRootedPath(
        blacklist.getParentDirectory().getParentDirectory(),
        new PathFragment("config/blacklisted.txt"));
    differencer.invalidate(ImmutableSet.of(FileStateValue.key(rootedBlacklist)));
    for (String pkg : pkgs) {
      PackageLookupValue packageLookupValue = lookupPackage(pkg);
      assertTrue(packageLookupValue.packageExists());
    }
  }

  @Test
  public void testInvalidPackageName() throws Exception {
    scratch.file("parentpackage/invalidpackagename%42/BUILD");
    PackageLookupValue packageLookupValue = lookupPackage("parentpackage/invalidpackagename%42");
    assertFalse(packageLookupValue.packageExists());
    assertEquals(ErrorReason.INVALID_PACKAGE_NAME,
        packageLookupValue.getErrorReason());
    assertNotNull(packageLookupValue.getErrorMsg());
  }

  @Test
  public void testDirectoryNamedBuild() throws Exception {
    scratch.dir("parentpackage/isdirectory/BUILD");
    PackageLookupValue packageLookupValue = lookupPackage("parentpackage/isdirectory");
    assertFalse(packageLookupValue.packageExists());
    assertEquals(ErrorReason.NO_BUILD_FILE,
        packageLookupValue.getErrorReason());
    assertNotNull(packageLookupValue.getErrorMsg());
  }

  @Test
  public void testEverythingIsGood() throws Exception {
    scratch.file("parentpackage/everythinggood/BUILD");
    PackageLookupValue packageLookupValue = lookupPackage("parentpackage/everythinggood");
    assertTrue(packageLookupValue.packageExists());
    assertEquals(rootDirectory, packageLookupValue.getRoot());
    assertEquals(BuildFileName.BUILD, packageLookupValue.getBuildFileName());
  }

  @Test
  public void testEmptyPackageName() throws Exception {
    scratch.file("BUILD");
    PackageLookupValue packageLookupValue = lookupPackage("");
    assertTrue(packageLookupValue.packageExists());
    assertEquals(rootDirectory, packageLookupValue.getRoot());
    assertEquals(BuildFileName.BUILD, packageLookupValue.getBuildFileName());
  }

  @Test
  public void testWorkspaceLookup() throws Exception {
    scratch.overwriteFile("WORKSPACE");
    PackageLookupValue packageLookupValue = lookupPackage(
        PackageIdentifier.createInMainRepo("external"));
    assertTrue(packageLookupValue.packageExists());
    assertEquals(rootDirectory, packageLookupValue.getRoot());
  }

  @Test
  public void testPackageLookupValueHashCodeAndEqualsContract() throws Exception {
    Path root1 = rootDirectory.getRelative("root1");
    Path root2 = rootDirectory.getRelative("root2");
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
    assertTrue(packageLookupValue.packageExists());

    // Then, use the incorrect label.
    packageLookupValue = lookupPackage(PackageIdentifier.createInMainRepo("local/repo"));
    assertEquals(expectedPackageExists, packageLookupValue.packageExists());
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
          .hasMessage(
              "no such package 'local/repo': Unable to determine the local repository for "
                  + "directory /workspace/local/repo");
    }
  }
}
