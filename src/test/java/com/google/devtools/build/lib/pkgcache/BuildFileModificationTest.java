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
package com.google.devtools.build.lib.pkgcache;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.DefaultBuildOptionsForTesting;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.StarlarkSemanticsOptions;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.SkyframeExecutorTestHelper;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.OptionsParser;
import java.nio.charset.StandardCharsets;
import java.util.Optional;
import java.util.UUID;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for package loading.
 */
@RunWith(JUnit4.class)
public class BuildFileModificationTest extends FoundationTestCase {

  private ManualClock clock = new ManualClock();
  private AnalysisMock analysisMock;
  private ConfiguredRuleClassProvider ruleClassProvider;
  private SkyframeExecutor skyframeExecutor;
  private final ActionKeyContext actionKeyContext = new ActionKeyContext();

  @Before
  public final void disableLogging() throws Exception {
    Logger.getLogger("com.google.devtools").setLevel(Level.SEVERE);
  }

  @Before
  public final void initializeSkyframeExecutor() throws Exception {
    analysisMock = AnalysisMock.get();
    ruleClassProvider = analysisMock.createRuleClassProvider();
    BlazeDirectories directories =
        new BlazeDirectories(
            new ServerDirectories(outputBase, outputBase, outputBase),
            rootDirectory,
            /* defaultSystemJavabase= */ null,
            analysisMock.getProductName());
    PackageFactory pkgFactory =
        analysisMock
            .getPackageFactoryBuilderForTesting(directories)
            .build(ruleClassProvider, fileSystem);
    skyframeExecutor =
        BazelSkyframeExecutorConstants.newBazelSkyframeExecutorBuilder()
            .setPkgFactory(pkgFactory)
            .setFileSystem(fileSystem)
            .setDirectories(directories)
            .setActionKeyContext(actionKeyContext)
            .setDefaultBuildOptions(
                DefaultBuildOptionsForTesting.getDefaultBuildOptionsForTest(ruleClassProvider))
            .setExtraSkyFunctions(analysisMock.getSkyFunctions(directories))
            .build();
    skyframeExecutor.injectExtraPrecomputedValues(
        ImmutableList.of(
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE, Optional.empty())));
    SkyframeExecutorTestHelper.process(skyframeExecutor);
    OptionsParser parser =
        OptionsParser.builder()
            .optionsClasses(PackageOptions.class, StarlarkSemanticsOptions.class)
            .build();
    setUpSkyframe(
        parser.getOptions(PackageOptions.class), parser.getOptions(StarlarkSemanticsOptions.class));
  }

  private void setUpSkyframe(
      PackageOptions packageOptions, StarlarkSemanticsOptions starlarkSemanticsOptions) {
    PathPackageLocator pkgLocator =
        PathPackageLocator.create(
            null,
            packageOptions.packagePath,
            reporter,
            rootDirectory,
            rootDirectory,
            BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY);
    packageOptions.showLoadingProgress = true;
    packageOptions.globbingThreads = 7;
    skyframeExecutor.preparePackageLoading(
        pkgLocator,
        packageOptions,
        starlarkSemanticsOptions,
        UUID.randomUUID(),
        ImmutableMap.<String, String>of(),
        new TimestampGranularityMonitor(clock));
    skyframeExecutor.setActionEnv(ImmutableMap.<String, String>of());
    skyframeExecutor.setDeletedPackages(ImmutableSet.copyOf(packageOptions.getDeletedPackages()));
  }

  @Override
  protected FileSystem createFileSystem() {
    return new InMemoryFileSystem(clock);
  }

  private void invalidatePackages() throws InterruptedException {
    skyframeExecutor.invalidateFilesUnderPathForTesting(
        reporter, ModifiedFileSet.EVERYTHING_MODIFIED, Root.fromPath(rootDirectory));
  }

  private Package getPackage(String packageName)
      throws NoSuchPackageException, InterruptedException {
    return skyframeExecutor.getPackageManager().getPackage(reporter,
        PackageIdentifier.createInMainRepo(packageName));
  }

  @Test
  public void testCTimeChangeDetectedWithError() throws Exception {
    reporter.removeHandler(failFastHandler);
    Path build = scratch.file(
        "a/BUILD", "cc_library(name='a', feet='stinky')".getBytes(StandardCharsets.ISO_8859_1));
    Package a1 = getPackage("a");
    assertThat(a1.containsErrors()).isTrue();
    assertContainsEvent("//a:a: no such attribute 'feet'");
    eventCollector.clear();
    // writeContent updates mtime and ctime. Note that we keep the content length exactly the same.
    clock.advanceMillis(1);
    FileSystemUtils.writeContent(
        build, "cc_library(name='a', srcs=['a.cc'])".getBytes(StandardCharsets.ISO_8859_1));

    invalidatePackages();
    Package a2 = getPackage("a");
    assertThat(a2).isNotSameInstanceAs(a1);
    assertThat(a2.containsErrors()).isFalse();
    assertNoEvents();
  }

  @Test
  public void testCTimeChangeDetected() throws Exception {
    Path path = scratch.file(
        "pkg/BUILD", "cc_library(name = 'foo')\n".getBytes(StandardCharsets.ISO_8859_1));
    Package oldPkg = getPackage("pkg");

    // Note that the content has exactly the same length as before.
    clock.advanceMillis(1);
    FileSystemUtils.writeContent(
        path, "cc_library(name = 'bar')\n".getBytes(StandardCharsets.ISO_8859_1));
    assertThat(getPackage("pkg"))
        .isSameInstanceAs(oldPkg); // Change only becomes visible after invalidatePackages.

    invalidatePackages();

    Package newPkg = getPackage("pkg");
    assertThat(newPkg).isNotSameInstanceAs(oldPkg);
    assertThat(newPkg.getTarget("bar")).isNotNull();
  }

  @Test
  public void testLengthChangeDetected() throws Exception {
    reporter.removeHandler(failFastHandler);
    Path build = scratch.file(
        "a/BUILD", "cc_library(name='a', srcs=['a.cc'])".getBytes(StandardCharsets.ISO_8859_1));
    Package a1 = getPackage("a");
    eventCollector.clear();
    // Note that we didn't advance the clock, so ctime/mtime is the same as before.
    // However, the file contents are one byte longer.
    FileSystemUtils.writeContent(
        build, "cc_library(name='ab', srcs=['a.cc'])".getBytes(StandardCharsets.ISO_8859_1));

    invalidatePackages();
    Package a2 = getPackage("a");
    assertThat(a2).isNotSameInstanceAs(a1);
    assertNoEvents();
  }

  @Test
  public void testTouchedBuildFileCausesReloadAfterSync() throws Exception {
    Path path = scratch.file("pkg/BUILD",
                             "cc_library(name = 'foo')");

    Package oldPkg = getPackage("pkg");
    // Change ctime to 1.
    clock.advanceMillis(1);
    path.setLastModifiedTime(1001);
    assertThat(getPackage("pkg")).isSameInstanceAs(oldPkg); // change not yet visible

    invalidatePackages();

    Package newPkg = getPackage("pkg");
    assertThat(newPkg).isNotSameInstanceAs(oldPkg);
  }
}
