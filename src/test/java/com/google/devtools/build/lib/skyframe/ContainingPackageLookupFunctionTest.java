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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageFactory.EnvironmentExtension;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryFunction;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryRule;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryLoaderFunction;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction.CrossRepositoryLabelViolationStrategy;
import com.google.devtools.build.lib.syntax.SkylarkSemantics;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.PathFragment;
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
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link ContainingPackageLookupFunction}.
 */
@RunWith(JUnit4.class)
public class ContainingPackageLookupFunctionTest extends FoundationTestCase {

  private AtomicReference<ImmutableSet<PackageIdentifier>> deletedPackages;
  private MemoizingEvaluator evaluator;
  private SequentialBuildDriver driver;
  private RecordingDifferencer differencer;

  @Before
  public final void setUp() throws Exception  {
    AnalysisMock analysisMock = AnalysisMock.get();

    AtomicReference<PathPackageLocator> pkgLocator =
        new AtomicReference<>(
            new PathPackageLocator(
                outputBase,
                ImmutableList.of(rootDirectory),
                BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY));
    deletedPackages = new AtomicReference<>(ImmutableSet.<PackageIdentifier>of());
    BlazeDirectories directories =
        new BlazeDirectories(
            new ServerDirectories(rootDirectory, outputBase),
            rootDirectory,
            analysisMock.getProductName());
    ExternalFilesHelper externalFilesHelper =
        new ExternalFilesHelper(
            pkgLocator,
            ExternalFileAction.DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_PATHS,
            directories);

    Map<SkyFunctionName, SkyFunction> skyFunctions = new HashMap<>();
    skyFunctions.put(SkyFunctions.CONTAINING_PACKAGE_LOOKUP, new ContainingPackageLookupFunction());

    skyFunctions.put(
        SkyFunctions.PACKAGE_LOOKUP,
        new PackageLookupFunction(
            deletedPackages,
            CrossRepositoryLabelViolationStrategy.ERROR,
            BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY));
    skyFunctions.put(
        SkyFunctions.PACKAGE, new PackageFunction(null, null, null, null, null, null, null));
    skyFunctions.put(SkyFunctions.BLACKLISTED_PACKAGE_PREFIXES,
        new BlacklistedPackagePrefixesFunction());
    skyFunctions.put(SkyFunctions.FILE_STATE, new FileStateFunction(
        new AtomicReference<TimestampGranularityMonitor>(), externalFilesHelper));
    skyFunctions.put(SkyFunctions.FILE, new FileFunction(pkgLocator));
    skyFunctions.put(SkyFunctions.DIRECTORY_LISTING, new DirectoryListingFunction());
    skyFunctions.put(
        SkyFunctions.DIRECTORY_LISTING_STATE,
        new DirectoryListingStateFunction(externalFilesHelper));
    RuleClassProvider ruleClassProvider = analysisMock.createRuleClassProvider();
    skyFunctions.put(SkyFunctions.WORKSPACE_AST, new WorkspaceASTFunction(ruleClassProvider));
    skyFunctions.put(
        SkyFunctions.WORKSPACE_FILE,
        new WorkspaceFileFunction(
            ruleClassProvider,
            analysisMock
                .getPackageFactoryBuilderForTesting(directories)
                .setEnvironmentExtensions(
                    ImmutableList.<EnvironmentExtension>of(
                        new PackageFactory.EmptyEnvironmentExtension()))
                .build(ruleClassProvider, scratch.getFileSystem()),
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
        new RepositoryDelegatorFunction(
            repositoryHandlers, null, new AtomicBoolean(true), ImmutableMap::of, directories));
    skyFunctions.put(SkyFunctions.REPOSITORY, new RepositoryLoaderFunction());

    differencer = new SequencedRecordingDifferencer();
    evaluator = new InMemoryMemoizingEvaluator(skyFunctions, differencer);
    driver = new SequentialBuildDriver(evaluator);
    PrecomputedValue.BUILD_ID.set(differencer, UUID.randomUUID());
    PrecomputedValue.BLACKLISTED_PACKAGE_PREFIXES_FILE.set(differencer,
        PathFragment.EMPTY_FRAGMENT);
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(differencer, pkgLocator.get());
    PrecomputedValue.SKYLARK_SEMANTICS.set(differencer, SkylarkSemantics.DEFAULT_SEMANTICS);
    RepositoryDelegatorFunction.REPOSITORY_OVERRIDES.set(
        differencer, ImmutableMap.<RepositoryName, PathFragment>of());
  }

  private ContainingPackageLookupValue lookupContainingPackage(String packageName)
      throws InterruptedException {
    return lookupContainingPackage(PackageIdentifier.createInMainRepo(packageName));
  }

  private ContainingPackageLookupValue lookupContainingPackage(PackageIdentifier packageIdentifier)
      throws InterruptedException {
    SkyKey key = ContainingPackageLookupValue.key(packageIdentifier);
    return driver
        .<ContainingPackageLookupValue>evaluate(
            ImmutableList.of(key),
            false,
            SkyframeExecutor.DEFAULT_THREAD_COUNT,
            NullEventHandler.INSTANCE)
        .get(key);
  }

  @Test
  public void testNoContainingPackage() throws Exception {
    ContainingPackageLookupValue value = lookupContainingPackage("a/b");
    assertThat(value.hasContainingPackage()).isFalse();
  }

  @Test
  public void testContainingPackageIsParent() throws Exception {
    scratch.file("a/BUILD");
    ContainingPackageLookupValue value = lookupContainingPackage("a/b");
    assertThat(value.hasContainingPackage()).isTrue();
    assertThat(value.getContainingPackageName()).isEqualTo(PackageIdentifier.createInMainRepo("a"));
    assertThat(value.getContainingPackageRoot()).isEqualTo(rootDirectory);
  }

  @Test
  public void testContainingPackageIsSelf() throws Exception {
    scratch.file("a/b/BUILD");
    ContainingPackageLookupValue value = lookupContainingPackage("a/b");
    assertThat(value.hasContainingPackage()).isTrue();
    assertThat(value.getContainingPackageName())
        .isEqualTo(PackageIdentifier.createInMainRepo("a/b"));
    assertThat(value.getContainingPackageRoot()).isEqualTo(rootDirectory);
  }

  @Test
  public void testContainingPackageIsExternalRepositoryViaExternalRepository() throws Exception {
    scratch.overwriteFile(
        "WORKSPACE",
        "local_repository(name='a', path='a')");
    scratch.file("a/WORKSPACE");
    scratch.file("a/BUILD");
    scratch.file("a/b/BUILD");
    ContainingPackageLookupValue value =
        lookupContainingPackage(
            PackageIdentifier.create(RepositoryName.create("@a"), PathFragment.create("b")));
    assertThat(value.hasContainingPackage()).isTrue();
    assertThat(value.getContainingPackageName())
        .isEqualTo(PackageIdentifier.create(RepositoryName.create("@a"), PathFragment.create("b")));
  }

  @Test
  public void testContainingPackageIsExternalRepositoryViaLocalPath() throws Exception {
    scratch.overwriteFile(
        "WORKSPACE",
        "local_repository(name='a', path='a')");
    scratch.file("a/WORKSPACE");
    scratch.file("a/BUILD");
    scratch.file("a/b/BUILD");
    ContainingPackageLookupValue value = lookupContainingPackage("a/b");
    assertThat(value.hasContainingPackage()).isTrue();
    assertThat(value.getContainingPackageName())
        .isEqualTo(PackageIdentifier.create(RepositoryName.create("@a"), PathFragment.create("b")));
  }

  @Test
  public void testEqualsAndHashCodeContract() throws Exception {
    ContainingPackageLookupValue valueA1 = ContainingPackageLookupValue.NONE;
    ContainingPackageLookupValue valueA2 = ContainingPackageLookupValue.NONE;
    ContainingPackageLookupValue valueB1 =
        ContainingPackageLookupValue.withContainingPackage(
            PackageIdentifier.createInMainRepo("b"), rootDirectory);
    ContainingPackageLookupValue valueB2 =
        ContainingPackageLookupValue.withContainingPackage(
            PackageIdentifier.createInMainRepo("b"), rootDirectory);
    PackageIdentifier cFrag = PackageIdentifier.createInMainRepo("c");
    ContainingPackageLookupValue valueC1 =
        ContainingPackageLookupValue.withContainingPackage(cFrag, rootDirectory);
    ContainingPackageLookupValue valueC2 =
        ContainingPackageLookupValue.withContainingPackage(cFrag, rootDirectory);
    ContainingPackageLookupValue valueCOther =
        ContainingPackageLookupValue.withContainingPackage(
            cFrag, rootDirectory.getRelative("other_root"));
    new EqualsTester()
        .addEqualityGroup(valueA1, valueA2)
        .addEqualityGroup(valueB1, valueB2)
        .addEqualityGroup(valueC1, valueC2)
        .addEqualityGroup(valueCOther)
        .testEquals();
  }
}
