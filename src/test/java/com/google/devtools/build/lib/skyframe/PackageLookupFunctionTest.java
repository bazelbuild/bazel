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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.bazel.rules.BazelRulesModule;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.PackageLookupValue.ErrorReason;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SequentialBuildDriver;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Tests for {@link PackageLookupFunction}.
 */
@RunWith(JUnit4.class)
public class PackageLookupFunctionTest extends FoundationTestCase {
  private AtomicReference<ImmutableSet<PackageIdentifier>> deletedPackages;
  private MemoizingEvaluator evaluator;
  private SequentialBuildDriver driver;
  private RecordingDifferencer differencer;

  @Before
  public final void setUp() throws Exception {
    Path emptyPackagePath = rootDirectory.getRelative("somewhere/else");
    scratch.file("parentpackage/BUILD");

    AtomicReference<PathPackageLocator> pkgLocator = new AtomicReference<>(
        new PathPackageLocator(outputBase, ImmutableList.of(emptyPackagePath, rootDirectory)));
    deletedPackages = new AtomicReference<>(ImmutableSet.<PackageIdentifier>of());
    ExternalFilesHelper externalFilesHelper = new ExternalFilesHelper(pkgLocator, false);
    TimestampGranularityMonitor tsgm = new TimestampGranularityMonitor(BlazeClock.instance());
    BlazeDirectories directories = new BlazeDirectories(rootDirectory, outputBase, rootDirectory);

    Map<SkyFunctionName, SkyFunction> skyFunctions = new HashMap<>();
    skyFunctions.put(SkyFunctions.PACKAGE_LOOKUP,
        new PackageLookupFunction(deletedPackages));
    skyFunctions.put(
        SkyFunctions.PACKAGE,
        new PackageFunction(null, null, null, null, null, null, null));
    skyFunctions.put(SkyFunctions.FILE_STATE, new FileStateFunction(tsgm, externalFilesHelper));
    skyFunctions.put(SkyFunctions.FILE, new FileFunction(pkgLocator));
    skyFunctions.put(SkyFunctions.BLACKLISTED_PACKAGE_PREFIXES,
        new BlacklistedPackagePrefixesFunction());
    RuleClassProvider ruleClassProvider = TestRuleClassProvider.getRuleClassProvider();
    skyFunctions.put(SkyFunctions.WORKSPACE_AST,
        new WorkspaceASTFunction(TestRuleClassProvider.getRuleClassProvider()));
    skyFunctions.put(
        SkyFunctions.WORKSPACE_FILE,
        new WorkspaceFileFunction(
            ruleClassProvider,
            new PackageFactory(
                ruleClassProvider, new BazelRulesModule().getPackageEnvironmentExtension()),
            directories));
    differencer = new RecordingDifferencer();
    evaluator = new InMemoryMemoizingEvaluator(skyFunctions, differencer);
    driver = new SequentialBuildDriver(evaluator);
    PrecomputedValue.BUILD_ID.set(differencer, UUID.randomUUID());
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(differencer, pkgLocator.get());
    PrecomputedValue.BLACKLISTED_PACKAGE_PREFIXES_FILE.set(
        differencer, PathFragment.EMPTY_FRAGMENT);
  }

  private PackageLookupValue lookupPackage(String packageName) throws InterruptedException {
    return lookupPackage(PackageIdentifier.createInDefaultRepo(packageName));
  }

  private PackageLookupValue lookupPackage(PackageIdentifier packageId)
      throws InterruptedException {
    SkyKey key = PackageLookupValue.key(packageId);
    return driver.<PackageLookupValue>evaluate(
        ImmutableList.of(key), false, SkyframeExecutor.DEFAULT_THREAD_COUNT,
        NullEventHandler.INSTANCE).get(key);
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
        PackageIdentifier.createInDefaultRepo("parentpackage/deletedpackage")));
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
  }

  @Test
  public void testEmptyPackageName() throws Exception {
    scratch.file("BUILD");
    PackageLookupValue packageLookupValue = lookupPackage("");
    assertTrue(packageLookupValue.packageExists());
    assertEquals(rootDirectory, packageLookupValue.getRoot());
  }

  @Test
  public void testWorkspaceLookup() throws Exception {
    scratch.overwriteFile("WORKSPACE");
    PackageLookupValue packageLookupValue = lookupPackage("external");
    assertTrue(packageLookupValue.packageExists());
    assertEquals(rootDirectory, packageLookupValue.getRoot());
  }

  // TODO(kchodorow): Clean this up (see TODOs in PackageLookupValue).
  @Test
  public void testExternalPackageLookupSemantics() {
    PackageLookupValue existing = PackageLookupValue.workspace(rootDirectory);
    assertTrue(existing.isExternalPackage());
    assertTrue(existing.packageExists());
    PackageLookupValue nonExistent = PackageLookupValue.workspace(rootDirectory.getRelative("x/y"));
    assertTrue(nonExistent.isExternalPackage());
    assertFalse(nonExistent.packageExists());
  }

  @Test
  public void testPackageLookupValueHashCodeAndEqualsContract() throws Exception {
    Path root1 = rootDirectory.getRelative("root1");
    Path root2 = rootDirectory.getRelative("root2");
    // Our (seeming) duplication of parameters here is intentional. Some of the subclasses of
    // PackageLookupValue are supposed to have reference equality semantics, and some are supposed
    // to have logical equality semantics.
    new EqualsTester()
        .addEqualityGroup(PackageLookupValue.success(root1), PackageLookupValue.success(root1))
        .addEqualityGroup(PackageLookupValue.success(root2), PackageLookupValue.success(root2))
        .addEqualityGroup(
            PackageLookupValue.NO_BUILD_FILE_VALUE, PackageLookupValue.NO_BUILD_FILE_VALUE)
        .addEqualityGroup(
            PackageLookupValue.DELETED_PACKAGE_VALUE, PackageLookupValue.DELETED_PACKAGE_VALUE)
        .addEqualityGroup(PackageLookupValue.invalidPackageName("nope1"),
            PackageLookupValue.invalidPackageName("nope1"))
        .addEqualityGroup(PackageLookupValue.invalidPackageName("nope2"),
             PackageLookupValue.invalidPackageName("nope2"))
        .testEquals();
  }
}
