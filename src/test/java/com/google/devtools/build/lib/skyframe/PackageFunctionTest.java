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
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;

import com.google.common.base.Optional;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.ConstantRuleVisibility;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageValidator;
import com.google.devtools.build.lib.packages.PackageValidator.InvalidPackageException;
import com.google.devtools.build.lib.packages.StarlarkSemanticsOptions;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.common.options.Options;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

/**
 * Unit tests of specific functionality of PackageFunction. Note that it's already tested indirectly
 * in several other places.
 */
@RunWith(JUnit4.class)
public class PackageFunctionTest extends BuildViewTestCase {

  @Rule public final MockitoRule mockito = MockitoJUnit.rule();

  @Mock private PackageValidator mockPackageValidator;

  private CustomInMemoryFs fs = new CustomInMemoryFs(new ManualClock());

  private void preparePackageLoading(Path... roots) {
    preparePackageLoadingWithCustomStarklarkSemanticsOptions(
        Options.getDefaults(StarlarkSemanticsOptions.class), roots);
  }

  private void preparePackageLoadingWithCustomStarklarkSemanticsOptions(
      StarlarkSemanticsOptions starlarkSemanticsOptions, Path... roots) {
    PackageOptions packageOptions = Options.getDefaults(PackageOptions.class);
    packageOptions.defaultVisibility = ConstantRuleVisibility.PUBLIC;
    packageOptions.showLoadingProgress = true;
    packageOptions.globbingThreads = 7;
    getSkyframeExecutor()
        .preparePackageLoading(
            new PathPackageLocator(
                outputBase,
                Arrays.stream(roots).map(Root::fromPath).collect(ImmutableList.toImmutableList()),
                BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY),
            packageOptions,
            starlarkSemanticsOptions,
            UUID.randomUUID(),
            ImmutableMap.<String, String>of(),
            new TimestampGranularityMonitor(BlazeClock.instance()));
    skyframeExecutor.setActionEnv(ImmutableMap.<String, String>of());
  }

  @Override
  protected FileSystem createFileSystem() {
    return fs;
  }

  @Override
  protected PackageValidator getPackageValidator() {
    return mockPackageValidator;
  }

  private Package validPackageWithoutErrors(SkyKey skyKey) throws InterruptedException {
    return validPackageInternal(skyKey, /*checkPackageError=*/ true);
  }

  private Package validPackage(SkyKey skyKey) throws InterruptedException {
    return validPackageInternal(skyKey, /*checkPackageError=*/ false);
  }

  private Package validPackageInternal(SkyKey skyKey, boolean checkPackageError)
      throws InterruptedException {
    SkyframeExecutor skyframeExecutor = getSkyframeExecutor();
    skyframeExecutor.injectExtraPrecomputedValues(
        ImmutableList.of(
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE,
                Optional.<RootedPath>absent())));
    EvaluationResult<PackageValue> result =
        SkyframeExecutorTestUtils.evaluate(
            skyframeExecutor, skyKey, /*keepGoing=*/ false, reporter);
    if (result.hasError()) {
      fail(result.getError(skyKey).getException().getMessage());
    }
    PackageValue value = result.get(skyKey);
    if (checkPackageError) {
      assertThat(value.getPackage().containsErrors()).isFalse();
    }
    return value.getPackage();
  }

  @Test
  public void testValidPackage() throws Exception {
    scratch.file("pkg/BUILD");
    validPackageWithoutErrors(PackageValue.key(PackageIdentifier.parse("@//pkg")));
  }

  @Test
  public void testInvalidPackage() throws Exception {
    scratch.file("pkg/BUILD", "sh_library(name='foo', srcs=['foo.sh'])");
    scratch.file("pkg/foo.sh");

    doAnswer(
            inv -> {
              Package pkg = inv.getArgument(0, Package.class);
              if (pkg.getName().equals("pkg")) {
                inv.getArgument(1, ExtendedEventHandler.class).handle(Event.warn("warning event"));
                throw new InvalidPackageException(pkg.getPackageIdentifier(), "no good");
              }
              return null;
            })
        .when(mockPackageValidator)
        .validate(any(Package.class), any(ExtendedEventHandler.class));

    invalidatePackages();

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//pkg"));
    EvaluationResult<PackageValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /*keepGoing=*/ false, reporter);
    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(skyKey)
        .hasExceptionThat()
        .isInstanceOf(InvalidPackageException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(skyKey)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("no such package 'pkg': no good");
    assertContainsEvent("warning event");
  }

  @Test
  public void testSkyframeExecutorClearedPackagesResultsInReload() throws Exception {
    scratch.file("pkg/BUILD", "sh_library(name='foo', srcs=['foo.sh'])");
    scratch.file("pkg/foo.sh");

    invalidatePackages();

    // Use number of times the package was validated as a proxy for number of times it was loaded.
    AtomicInteger validationCount = new AtomicInteger();
    doAnswer(
            inv -> {
              if (inv.getArgument(0, Package.class).getName().equals("pkg")) {
                validationCount.incrementAndGet();
              }
              return null;
            })
        .when(mockPackageValidator)
        .validate(any(Package.class), any(ExtendedEventHandler.class));

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//pkg"));
    EvaluationResult<PackageValue> result1 =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /*keepGoing=*/ false, reporter);
    assertThatEvaluationResult(result1).hasNoError();

    skyframeExecutor.clearLoadedPackages();

    EvaluationResult<PackageValue> result2 =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /*keepGoing=*/ false, reporter);
    assertThatEvaluationResult(result2).hasNoError();

    assertThat(validationCount.get()).isEqualTo(2);
  }

  @Test
  public void testPropagatesFilesystemInconsistencies() throws Exception {
    reporter.removeHandler(failFastHandler);
    RecordingDifferencer differencer = getSkyframeExecutor().getDifferencerForTesting();
    Root pkgRoot = getSkyframeExecutor().getPathEntries().get(0);
    Path fooBuildFile = scratch.file("foo/BUILD");
    Path fooDir = fooBuildFile.getParentDirectory();

    // Our custom filesystem says that fooDir is neither a file nor directory nor symlink
    FileStatus inconsistentFileStatus =
        new FileStatus() {
          @Override
          public boolean isFile() {
            return false;
          }

          @Override
          public boolean isDirectory() {
            return false;
          }

          @Override
          public boolean isSymbolicLink() {
            return false;
          }

          @Override
          public boolean isSpecialFile() {
            return false;
          }

          @Override
          public long getSize() throws IOException {
            return 0;
          }

          @Override
          public long getLastModifiedTime() throws IOException {
            return 0;
          }

          @Override
          public long getLastChangeTime() throws IOException {
            return 0;
          }

          @Override
          public long getNodeId() throws IOException {
            return 0;
          }
        };

    fs.stubStat(fooBuildFile, inconsistentFileStatus);
    RootedPath pkgRootedPath = RootedPath.toRootedPath(pkgRoot, fooDir);
    SkyValue fooDirValue = FileStateValue.create(pkgRootedPath, tsgm);
    differencer.inject(ImmutableMap.of(FileStateValue.key(pkgRootedPath), fooDirValue));
    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//foo"));
    String expectedMessage =
        "according to stat, existing path /workspace/foo/BUILD is neither"
            + " a file nor directory nor symlink.";
    EvaluationResult<PackageValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /*keepGoing=*/ false, reporter);
    assertThat(result.hasError()).isTrue();
    ErrorInfo errorInfo = result.getError(skyKey);
    String errorMessage = errorInfo.getException().getMessage();
    assertThat(errorMessage).contains("Inconsistent filesystem operations");
    assertThat(errorMessage).contains(expectedMessage);
  }

  @Test
  public void testPropagatesFilesystemInconsistencies_Globbing() throws Exception {
    getSkyframeExecutor().turnOffSyscallCacheForTesting();
    reporter.removeHandler(failFastHandler);
    RecordingDifferencer differencer = getSkyframeExecutor().getDifferencerForTesting();
    Root pkgRoot = getSkyframeExecutor().getPathEntries().get(0);
    scratch.file(
        "foo/BUILD",
        "sh_library(name = 'foo', srcs = glob(['bar/**/baz.sh']))",
        "x = 1//0" // causes 'foo' to be marked in error
        );
    Path bazFile = scratch.file("foo/bar/baz/baz.sh");
    Path bazDir = bazFile.getParentDirectory();
    Path barDir = bazDir.getParentDirectory();

    // Our custom filesystem says "foo/bar/baz" does not exist but it also says that "foo/bar"
    // has a child directory "baz".
    fs.stubStat(bazDir, null);
    RootedPath barDirRootedPath = RootedPath.toRootedPath(pkgRoot, barDir);
    differencer.inject(
        ImmutableMap.of(
            DirectoryListingStateValue.key(barDirRootedPath),
            DirectoryListingStateValue.create(
                ImmutableList.of(new Dirent("baz", Dirent.Type.DIRECTORY)))));
    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//foo"));
    String expectedMessage = "/workspace/foo/bar/baz is no longer an existing directory";
    EvaluationResult<PackageValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /*keepGoing=*/ false, reporter);
    assertThat(result.hasError()).isTrue();
    ErrorInfo errorInfo = result.getError(skyKey);
    String errorMessage = errorInfo.getException().getMessage();
    assertThat(errorMessage).contains("Inconsistent filesystem operations");
    assertThat(errorMessage).contains(expectedMessage);
  }

  /** Regression test for unexpected exception type from PackageValue. */
  @Test
  public void testDiscrepancyBetweenLegacyAndSkyframePackageLoadingErrors() throws Exception {
    // Normally, legacy globbing and skyframe globbing share a cache for `readdir` filesystem calls.
    // In order to exercise a situation where they observe different results for filesystem calls,
    // we disable the cache. This might happen in a real scenario, e.g. if the cache hits a limit
    // and evicts entries.
    getSkyframeExecutor().turnOffSyscallCacheForTesting();
    reporter.removeHandler(failFastHandler);
    Path fooBuildFile =
        scratch.file("foo/BUILD", "sh_library(name = 'foo', srcs = glob(['bar/*.sh']))");
    Path fooDir = fooBuildFile.getParentDirectory();
    Path barDir = fooDir.getRelative("bar");
    scratch.file("foo/bar/baz.sh");
    fs.scheduleMakeUnreadableAfterReaddir(barDir);

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//foo"));
    String expectedMessage = "Encountered error 'Directory is not readable'";
    EvaluationResult<PackageValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /*keepGoing=*/ false, reporter);
    assertThat(result.hasError()).isTrue();
    ErrorInfo errorInfo = result.getError(skyKey);
    String errorMessage = errorInfo.getException().getMessage();
    assertThat(errorMessage).contains("Inconsistent filesystem operations");
    assertThat(errorMessage).contains(expectedMessage);
  }

  @SuppressWarnings("unchecked") // Cast of srcs attribute to Iterable<Label>.
  @Test
  public void testGlobOrderStable() throws Exception {
    scratch.file("foo/BUILD", "sh_library(name = 'foo', srcs = glob(['**/*.txt']))");
    scratch.file("foo/b.txt");
    scratch.file("foo/c/c.txt");
    preparePackageLoading(rootDirectory);
    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//foo"));
    Package pkg = validPackageWithoutErrors(skyKey);
    assertThat(
            (Iterable<Label>)
                pkg.getTarget("foo").getAssociatedRule().getAttributeContainer().getAttr("srcs"))
        .containsExactly(
            Label.parseAbsoluteUnchecked("//foo:b.txt"),
            Label.parseAbsoluteUnchecked("//foo:c/c.txt"))
        .inOrder();
    scratch.file("foo/d.txt");
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            ModifiedFileSet.builder().modify(PathFragment.create("foo/d.txt")).build(),
            Root.fromPath(rootDirectory));
    pkg = validPackageWithoutErrors(skyKey);
    assertThat(
            (Iterable<Label>)
                pkg.getTarget("foo").getAssociatedRule().getAttributeContainer().getAttr("srcs"))
        .containsExactly(
            Label.parseAbsoluteUnchecked("//foo:b.txt"),
            Label.parseAbsoluteUnchecked("//foo:c/c.txt"),
            Label.parseAbsoluteUnchecked("//foo:d.txt"))
        .inOrder();
  }

  @Test
  public void testGlobOrderStableWithLegacyAndSkyframeComponents() throws Exception {
    scratch.file("foo/BUILD", "sh_library(name = 'foo', srcs = glob(['*.txt']))");
    scratch.file("foo/b.txt");
    scratch.file("foo/a.config");
    preparePackageLoading(rootDirectory);
    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//foo"));
    assertSrcs(validPackageWithoutErrors(skyKey), "foo", "//foo:b.txt");
    scratch.overwriteFile(
        "foo/BUILD", "sh_library(name = 'foo', srcs = glob(['*.txt', '*.config']))");
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            ModifiedFileSet.builder().modify(PathFragment.create("foo/BUILD")).build(),
            Root.fromPath(rootDirectory));
    assertSrcs(validPackageWithoutErrors(skyKey), "foo", "//foo:a.config", "//foo:b.txt");
    scratch.overwriteFile(
        "foo/BUILD", "sh_library(name = 'foo', srcs = glob(['*.txt', '*.config'])) # comment");
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            ModifiedFileSet.builder().modify(PathFragment.create("foo/BUILD")).build(),
            Root.fromPath(rootDirectory));
    assertSrcs(validPackageWithoutErrors(skyKey), "foo", "//foo:a.config", "//foo:b.txt");
    getSkyframeExecutor().resetEvaluator();
    PackageOptions packageOptions = Options.getDefaults(PackageOptions.class);
    packageOptions.defaultVisibility = ConstantRuleVisibility.PUBLIC;
    packageOptions.showLoadingProgress = true;
    packageOptions.globbingThreads = 7;
    getSkyframeExecutor()
        .preparePackageLoading(
            new PathPackageLocator(
                outputBase,
                ImmutableList.of(Root.fromPath(rootDirectory)),
                BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY),
            packageOptions,
            Options.getDefaults(StarlarkSemanticsOptions.class),
            UUID.randomUUID(),
            ImmutableMap.<String, String>of(),
            tsgm);
    getSkyframeExecutor().setActionEnv(ImmutableMap.<String, String>of());
    assertSrcs(validPackageWithoutErrors(skyKey), "foo", "//foo:a.config", "//foo:b.txt");
  }

  @Test
  public void globEscapesAt() throws Exception {
    scratch.file("foo/BUILD", "filegroup(name = 'foo', srcs = glob(['*.txt']))");
    scratch.file("foo/@f.txt");
    preparePackageLoading(rootDirectory);
    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//foo"));
    assertSrcs(validPackageWithoutErrors(skyKey), "foo", "//foo:@f.txt");

    scratch.overwriteFile("foo/BUILD", "filegroup(name = 'foo', srcs = glob(['*.txt'])) # comment");
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            ModifiedFileSet.builder().modify(PathFragment.create("foo/BUILD")).build(),
            Root.fromPath(rootDirectory));
    assertSrcs(validPackageWithoutErrors(skyKey), "foo", "//foo:@f.txt");
  }

  /**
   * Tests that a symlink to a file outside of the package root is handled consistently. If the
   * default behavior of Bazel was changed from {@code
   * ExternalFileAction#DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_PATHS} to {@code
   * ExternalFileAction#ASSUME_NON_EXISTENT_AND_IMMUTABLE_FOR_EXTERNAL_PATHS} then foo/link.sh
   * should no longer appear in the srcs of //foo:foo. However, either way the srcs should be the
   * same independent of the evaluation being incremental or clean.
   */
  @Test
  public void testGlobWithExternalSymlink() throws Exception {
    scratch.file(
        "foo/BUILD",
        "sh_library(name = 'foo', srcs = glob(['*.sh']))",
        "sh_library(name = 'bar', srcs = glob(['link.sh']))",
        "sh_library(name = 'baz', srcs = glob(['subdir_link/*.txt']))");
    scratch.file("foo/ordinary.sh");
    Path externalTarget = scratch.file("../ops/target.txt");
    FileSystemUtils.ensureSymbolicLink(scratch.resolve("foo/link.sh"), externalTarget);
    FileSystemUtils.ensureSymbolicLink(
        scratch.resolve("foo/subdir_link"), externalTarget.getParentDirectory());
    preparePackageLoading(rootDirectory);
    SkyKey fooKey = PackageValue.key(PackageIdentifier.parse("@//foo"));
    Package fooPkg = validPackageWithoutErrors(fooKey);
    assertSrcs(fooPkg, "foo", "//foo:link.sh", "//foo:ordinary.sh");
    assertSrcs(fooPkg, "bar", "//foo:link.sh");
    assertSrcs(fooPkg, "baz", "//foo:subdir_link/target.txt");
    scratch.overwriteFile(
        "foo/BUILD",
        "sh_library(name = 'foo', srcs = glob(['*.sh'])) #comment",
        "sh_library(name = 'bar', srcs = glob(['link.sh']))",
        "sh_library(name = 'baz', srcs = glob(['subdir_link/*.txt']))");
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            ModifiedFileSet.builder().modify(PathFragment.create("foo/BUILD")).build(),
            Root.fromPath(rootDirectory));
    Package fooPkg2 = validPackageWithoutErrors(fooKey);
    assertThat(fooPkg2).isNotEqualTo(fooPkg);
    assertSrcs(fooPkg2, "foo", "//foo:link.sh", "//foo:ordinary.sh");
    assertSrcs(fooPkg2, "bar", "//foo:link.sh");
    assertSrcs(fooPkg2, "baz", "//foo:subdir_link/target.txt");
  }

  private static void assertSrcs(Package pkg, String targetName, String... expected)
      throws NoSuchTargetException {
    List<Label> expectedLabels = new ArrayList<>();
    for (String item : expected) {
      expectedLabels.add(Label.parseAbsoluteUnchecked(item));
    }
    assertThat(getSrcs(pkg, targetName)).containsExactlyElementsIn(expectedLabels).inOrder();
  }

  @SuppressWarnings("unchecked")
  private static Iterable<Label> getSrcs(Package pkg, String targetName)
      throws NoSuchTargetException {
    return (Iterable<Label>)
        pkg.getTarget(targetName).getAssociatedRule().getAttributeContainer().getAttr("srcs");
  }

  @Test
  public void testOneNewElementInMultipleGlob() throws Exception {
    scratch.file(
        "foo/BUILD",
        "sh_library(name = 'foo', srcs = glob(['*.sh']))",
        "sh_library(name = 'bar', srcs = glob(['*.sh', '*.txt']))");
    preparePackageLoading(rootDirectory);
    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//foo"));
    Package pkg = validPackageWithoutErrors(skyKey);
    scratch.file("foo/irrelevant");
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            ModifiedFileSet.builder().modify(PathFragment.create("foo/irrelevant")).build(),
            Root.fromPath(rootDirectory));
    assertThat(validPackageWithoutErrors(skyKey)).isSameInstanceAs(pkg);
  }

  @Test
  public void testNoNewElementInMultipleGlob() throws Exception {
    scratch.file(
        "foo/BUILD",
        "sh_library(name = 'foo', srcs = glob(['*.sh', '*.txt']))",
        "sh_library(name = 'bar', srcs = glob(['*.sh', '*.txt']))");
    preparePackageLoading(rootDirectory);
    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//foo"));
    Package pkg = validPackageWithoutErrors(skyKey);
    scratch.file("foo/irrelevant");
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            ModifiedFileSet.builder().modify(PathFragment.create("foo/irrelevant")).build(),
            Root.fromPath(rootDirectory));
    assertThat(validPackageWithoutErrors(skyKey)).isSameInstanceAs(pkg);
  }

  @Test
  public void testTransitiveStarlarkDepsStoredInPackage() throws Exception {
    scratch.file("foo/BUILD", "load('//bar:ext.bzl', 'a')");
    scratch.file("bar/BUILD");
    scratch.file("bar/ext.bzl", "load('//baz:ext.bzl', 'b')", "a = b");
    scratch.file("baz/BUILD");
    scratch.file("baz/ext.bzl", "b = 1");
    scratch.file("qux/BUILD");
    scratch.file("qux/ext.bzl", "c = 1");

    preparePackageLoading(rootDirectory);

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//foo"));
    Package pkg = validPackageWithoutErrors(skyKey);
    assertThat(pkg.getStarlarkFileDependencies())
        .containsExactly(
            Label.parseAbsolute("//bar:ext.bzl", ImmutableMap.of()),
            Label.parseAbsolute("//baz:ext.bzl", ImmutableMap.of()));

    scratch.overwriteFile("bar/ext.bzl", "load('//qux:ext.bzl', 'c')", "a = c");
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            ModifiedFileSet.builder().modify(PathFragment.create("bar/ext.bzl")).build(),
            Root.fromPath(rootDirectory));

    pkg = validPackageWithoutErrors(skyKey);
    assertThat(pkg.getStarlarkFileDependencies())
        .containsExactly(
            Label.parseAbsolute("//bar:ext.bzl", ImmutableMap.of()),
            Label.parseAbsolute("//qux:ext.bzl", ImmutableMap.of()));
  }

  @Test
  public void testNonExistingStarlarkExtension() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:bad_extension.bzl', 'some_symbol')",
        "genrule(name = gr,",
        "    outs = ['out.txt'],",
        "    cmd = 'echo hello >@')");
    invalidatePackages();

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//test/starlark"));
    EvaluationResult<PackageValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /*keepGoing=*/ false, reporter);
    assertThat(result.hasError()).isTrue();
    ErrorInfo errorInfo = result.getError(skyKey);
    String expectedMsg =
        "error loading package 'test/starlark': "
            + "cannot load '//test/starlark:bad_extension.bzl': no such file";
    assertThat(errorInfo.getException()).hasMessageThat().isEqualTo(expectedMsg);
  }

  @Test
  public void testNonExistingStarlarkExtensionFromExtension() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "test/starlark/extension.bzl",
        "load('//test/starlark:bad_extension.bzl', 'some_symbol')",
        "a = 'a'");
    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:extension.bzl', 'a')",
        "genrule(name = gr,",
        "    outs = ['out.txt'],",
        "    cmd = 'echo hello >@')");
    invalidatePackages();

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//test/starlark"));
    EvaluationResult<PackageValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /*keepGoing=*/ false, reporter);
    assertThat(result.hasError()).isTrue();
    ErrorInfo errorInfo = result.getError(skyKey);
    assertThat(errorInfo.getException())
        .hasMessageThat()
        .isEqualTo(
            "error loading package 'test/starlark': "
                + "in /workspace/test/starlark/extension.bzl: "
                + "cannot load '//test/starlark:bad_extension.bzl': no such file");
  }

  @Test
  public void testSymlinkCycleWithStarlarkExtension() throws Exception {
    reporter.removeHandler(failFastHandler);
    Path extensionFilePath = scratch.resolve("/workspace/test/starlark/extension.bzl");
    FileSystemUtils.ensureSymbolicLink(extensionFilePath, PathFragment.create("extension.bzl"));
    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:extension.bzl', 'a')",
        "genrule(name = gr,",
        "    outs = ['out.txt'],",
        "    cmd = 'echo hello >@')");
    invalidatePackages();

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//test/starlark"));
    EvaluationResult<PackageValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /*keepGoing=*/ false, reporter);
    assertThat(result.hasError()).isTrue();
    ErrorInfo errorInfo = result.getError(skyKey);
    assertThat(errorInfo.getRootCauseOfException()).isEqualTo(skyKey);
    assertThat(errorInfo.getException())
        .hasMessageThat()
        .isEqualTo(
            "error loading package 'test/starlark': Encountered error while reading extension "
                + "file 'test/starlark/extension.bzl': Symlink cycle");
  }

  @Test
  public void testIOErrorLookingForSubpackageForLabelIsHandled() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("foo/BUILD", "sh_library(name = 'foo', srcs = ['bar/baz.sh'])");
    Path barBuildFile = scratch.file("foo/bar/BUILD");
    fs.stubStatError(barBuildFile, new IOException("nope"));
    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//foo"));
    EvaluationResult<PackageValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /*keepGoing=*/ false, reporter);
    assertThat(result.hasError()).isTrue();
    assertContainsEvent("nope");
  }

  @Test
  public void testLoadOK() throws Exception {
    scratch.file("p/a.bzl", "a = 1; b = 1; d = 1");
    scratch.file("p/subdir/a.bzl", "c = 1; e = 1");
    scratch.file(
        "p/BUILD",
        //
        "load(':a.bzl', 'a')",
        "load('a.bzl', 'b')",
        "load('subdir/a.bzl', 'c')",
        "load('//p:a.bzl', 'd')",
        "load('//p:subdir/a.bzl', 'e')");
    validPackageWithoutErrors(PackageValue.key(PackageIdentifier.parse("@//p")));
  }

  // See WorkspaceASTFunctionTest for tests that exercise load('@repo...').

  @Test
  public void testLoadBadLabel() throws Exception {
    scratch.file("p/BUILD", "load('this\tis not a label', 'a')");
    reporter.removeHandler(failFastHandler);
    SkyKey key = PackageValue.key(PackageIdentifier.parse("@//p"));
    SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, /*keepGoing=*/ false, reporter);
    assertContainsEvent(
        "in load statement: invalid target name 'this<?>is not a label': target names may not"
            + " contain non-printable characters");
  }

  @Test
  public void testLoadFromExternalPackage() throws Exception {
    scratch.file("p/BUILD", "load('//external:file.bzl', 'a')");
    reporter.removeHandler(failFastHandler);
    SkyKey key = PackageValue.key(PackageIdentifier.parse("@//p"));
    SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, /*keepGoing=*/ false, reporter);
    assertContainsEvent("Starlark files may not be loaded from the //external package");
  }

  @Test
  public void testLoadWithoutBzlSuffix() throws Exception {
    scratch.file("p/BUILD", "load('//p:file.starlark', 'a')");
    reporter.removeHandler(failFastHandler);
    SkyKey key = PackageValue.key(PackageIdentifier.parse("@//p"));
    SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, /*keepGoing=*/ false, reporter);
    assertContainsEvent("The label must reference a file with extension '.bzl'");
  }

  @Test
  public void testBadWorkspaceFile() throws Exception {
    Path workspacePath = scratch.overwriteFile("WORKSPACE", "junk");
    SkyKey skyKey = PackageValue.key(PackageIdentifier.createInMainRepo("external"));
    getSkyframeExecutor()
        .invalidate(
            Predicates.equalTo(
                FileStateValue.key(
                    RootedPath.toRootedPath(
                        Root.fromPath(workspacePath.getParentDirectory()),
                        PathFragment.create(workspacePath.getBaseName())))));

    reporter.removeHandler(failFastHandler);
    EvaluationResult<PackageValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /*keepGoing=*/ false, reporter);
    assertThat(result.hasError()).isFalse();
    assertThat(result.get(skyKey).getPackage().containsErrors()).isTrue();
  }

  // Regression test for the two ugly consequences of a bug where GlobFunction incorrectly matched
  // dangling symlinks.
  @Test
  public void testIncrementalSkyframeHybridGlobbingOnDanglingSymlink() throws Exception {
    Path packageDirPath =
        scratch.file("foo/BUILD", "exports_files(glob(['*.txt']))").getParentDirectory();
    scratch.file("foo/existing.txt");
    FileSystemUtils.ensureSymbolicLink(packageDirPath.getChild("dangling.txt"), "nope");

    preparePackageLoading(rootDirectory);

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//foo"));
    Package pkg = validPackageWithoutErrors(skyKey);
    assertThat(pkg.containsErrors()).isFalse();
    assertThat(pkg.getTarget("existing.txt").getName()).isEqualTo("existing.txt");
    assertThrows(NoSuchTargetException.class, () -> pkg.getTarget("dangling.txt"));

    scratch.overwriteFile(
        "foo/BUILD", "exports_files(glob(['*.txt']))", "#some-irrelevant-comment");

    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            ModifiedFileSet.builder().modify(PathFragment.create("foo/BUILD")).build(),
            Root.fromPath(rootDirectory));

    Package pkg2 = validPackageWithoutErrors(skyKey);
    assertThat(pkg2.containsErrors()).isFalse();
    assertThat(pkg2.getTarget("existing.txt").getName()).isEqualTo("existing.txt");
    assertThrows(NoSuchTargetException.class, () -> pkg2.getTarget("dangling.txt"));
    // One consequence of the bug was that dangling symlinks were matched by globs evaluated by
    // Skyframe globbing, meaning there would incorrectly be corresponding targets in packages
    // that had skyframe cache hits during skyframe hybrid globbing.

    scratch.file("foo/nope");
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            ModifiedFileSet.builder().modify(PathFragment.create("foo/nope")).build(),
            Root.fromPath(rootDirectory));

    Package newPkg = validPackageWithoutErrors(skyKey);
    assertThat(newPkg.containsErrors()).isFalse();
    assertThat(newPkg.getTarget("existing.txt").getName()).isEqualTo("existing.txt");
    // Another consequence of the bug is that change pruning would incorrectly cut off changes that
    // caused a dangling symlink potentially matched by a glob to come into existence.
    assertThat(newPkg.getTarget("dangling.txt").getName()).isEqualTo("dangling.txt");
    assertThat(newPkg).isNotSameInstanceAs(pkg);
  }

  // Regression test for Skyframe globbing incorrectly matching the package's directory path on
  // 'glob(['**'], exclude_directories = 0)'. We test for this directly by triggering
  // hybrid globbing (gives coverage for both legacy globbing and skyframe globbing).
  @Test
  public void testRecursiveGlobNeverMatchesPackageDirectory() throws Exception {
    scratch.file(
        "foo/BUILD",
        "[sh_library(name = x + '-matched') for x in glob(['**'], exclude_directories = 0)]");
    scratch.file("foo/bar");

    preparePackageLoading(rootDirectory);

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//foo"));
    Package pkg = validPackageWithoutErrors(skyKey);
    assertThat(pkg.containsErrors()).isFalse();
    assertThat(pkg.getTarget("bar-matched").getName()).isEqualTo("bar-matched");
    assertThrows(NoSuchTargetException.class, () -> pkg.getTarget("-matched"));

    scratch.overwriteFile(
        "foo/BUILD",
        "[sh_library(name = x + '-matched') for x in glob(['**'], exclude_directories = 0)]",
        "#some-irrelevant-comment");
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            ModifiedFileSet.builder().modify(PathFragment.create("foo/BUILD")).build(),
            Root.fromPath(rootDirectory));

    Package pkg2 = validPackageWithoutErrors(skyKey);
    assertThat(pkg2.containsErrors()).isFalse();
    assertThat(pkg2.getTarget("bar-matched").getName()).isEqualTo("bar-matched");
    assertThrows(NoSuchTargetException.class, () -> pkg2.getTarget("-matched"));
  }

  @Test
  public void testPackageLoadingErrorOnIOExceptionReadingBuildFile() throws Exception {
    Path fooBuildFilePath = scratch.file("foo/BUILD");
    IOException exn = new IOException("nope");
    fs.throwExceptionOnGetInputStream(fooBuildFilePath, exn);

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//foo"));
    EvaluationResult<PackageValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /*keepGoing=*/ false, reporter);
    assertThat(result.hasError()).isTrue();
    ErrorInfo errorInfo = result.getError(skyKey);
    String errorMessage = errorInfo.getException().getMessage();
    assertThat(errorMessage).contains("nope");
    assertThat(errorInfo.getException()).isInstanceOf(NoSuchPackageException.class);
    assertThat(errorInfo.getException()).hasCauseThat().isInstanceOf(IOException.class);
  }

  @Test
  public void testPackageLoadingErrorOnIOExceptionReadingBzlFile() throws Exception {
    scratch.file("foo/BUILD", "load('//foo:bzl.bzl', 'x')");
    Path fooBzlFilePath = scratch.file("foo/bzl.bzl");
    IOException exn = new IOException("nope");
    fs.throwExceptionOnGetInputStream(fooBzlFilePath, exn);

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//foo"));
    EvaluationResult<PackageValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /*keepGoing=*/ false, reporter);
    assertThat(result.hasError()).isTrue();
    ErrorInfo errorInfo = result.getError(skyKey);
    String errorMessage = errorInfo.getException().getMessage();
    assertThat(errorMessage).contains("nope");
    assertThat(errorInfo.getException()).isInstanceOf(NoSuchPackageException.class);
    assertThat(errorInfo.getException()).hasCauseThat().isInstanceOf(IOException.class);
  }

  @Test
  public void testLabelsCrossesSubpackageBoundaries() throws Exception {
    reporter.removeHandler(failFastHandler);

    scratch.file("pkg/BUILD", "exports_files(['sub/blah'])");
    scratch.file("pkg/sub/BUILD");
    invalidatePackages();

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//pkg"));
    EvaluationResult<PackageValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /*keepGoing=*/ false, reporter);
    assertThatEvaluationResult(result).hasNoError();
    assertThat(result.get(skyKey).getPackage().containsErrors()).isTrue();
    assertContainsEvent("Label '//pkg:sub/blah' is invalid because 'pkg/sub' is a subpackage");
  }

  @Test
  public void testSymlinkCycleEncounteredWhileHandlingLabelCrossingSubpackageBoundaries()
      throws Exception {
    reporter.removeHandler(failFastHandler);

    scratch.file("pkg/BUILD", "exports_files(['sub/blah'])");
    Path subBuildFilePath = scratch.dir("pkg/sub").getChild("BUILD");
    FileSystemUtils.ensureSymbolicLink(subBuildFilePath, subBuildFilePath);
    invalidatePackages();

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//pkg"));
    EvaluationResult<PackageValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /*keepGoing=*/ false, reporter);
    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(skyKey)
        .hasExceptionThat()
        .isInstanceOf(BuildFileNotFoundException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(skyKey)
        .hasExceptionThat()
        .hasMessageThat()
        .contains(
            "no such package 'pkg/sub': Symlink cycle detected while trying to find BUILD file");
    assertContainsEvent("circular symlinks detected");
  }

  @Test
  public void testGlobAllowEmpty_ParamValueMustBeBoolean() throws Exception {
    reporter.removeHandler(failFastHandler);

    scratch.file("pkg/BUILD", "x = " + "glob(['*.foo'], allow_empty = 5)");
    invalidatePackages();

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//pkg"));
    Package pkg = validPackage(skyKey);

    String expectedEventString = "expected boolean for argument `allow_empty`, got `5`";
    MoreAsserts.assertContainsEvent(pkg.getEvents(), expectedEventString);
    assertContainsEvent(expectedEventString);
  }

  @Test
  public void testGlobAllowEmpty_FunctionParam() throws Exception {
    scratch.file("pkg/BUILD", "x = " + "glob(['*.foo'], allow_empty=True)");
    invalidatePackages();

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//pkg"));
    Package pkg = validPackage(skyKey);
    assertThat(pkg.containsErrors()).isFalse();
    assertThat(pkg.getEvents()).isEmpty();
  }

  @Test
  public void testGlobAllowEmpty_StarlarkOption() throws Exception {
    preparePackageLoadingWithCustomStarklarkSemanticsOptions(
        Options.parse(StarlarkSemanticsOptions.class, "--incompatible_disallow_empty_glob=false")
            .getOptions(),
        rootDirectory);

    scratch.file("pkg/BUILD", "x = " + "glob(['*.foo'])");
    invalidatePackages();

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//pkg"));
    Package pkg = validPackage(skyKey);
    assertThat(pkg.containsErrors()).isFalse();
    assertThat(pkg.getEvents()).isEmpty();
  }

  @Test
  public void testGlobDisallowEmpty_FunctionParam_WasNonEmptyAndBecomesEmpty() throws Exception {
    scratch.file("pkg/BUILD", "x = " + "glob(['*.foo'], allow_empty=False)");
    scratch.file("pkg/blah.foo");
    invalidatePackages();

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//pkg"));

    Package pkg = validPackage(skyKey);
    assertThat(pkg.containsErrors()).isFalse();
    assertThat(pkg.getEvents()).isEmpty();

    scratch.deleteFile("pkg/blah.foo");
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            ModifiedFileSet.builder().modify(PathFragment.create("pkg/blah.foo")).build(),
            Root.fromPath(rootDirectory));

    reporter.removeHandler(failFastHandler);
    pkg = validPackage(skyKey);
    assertThat(pkg.containsErrors()).isTrue();
    String expectedEventString =
        "glob pattern '*.foo' didn't match anything, but allow_empty is set to False";
    MoreAsserts.assertContainsEvent(pkg.getEvents(), expectedEventString);
    assertContainsEvent(expectedEventString);
  }

  @Test
  public void testGlobDisallowEmpty_StarlarkOption_WasNonEmptyAndBecomesEmpty() throws Exception {
    preparePackageLoadingWithCustomStarklarkSemanticsOptions(
        Options.parse(StarlarkSemanticsOptions.class, "--incompatible_disallow_empty_glob=true")
            .getOptions(),
        rootDirectory);

    scratch.file("pkg/BUILD", "x = " + "glob(['*.foo'])");
    scratch.file("pkg/blah.foo");
    invalidatePackages();

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//pkg"));

    Package pkg = validPackage(skyKey);
    assertThat(pkg.containsErrors()).isFalse();
    assertThat(pkg.getEvents()).isEmpty();

    scratch.deleteFile("pkg/blah.foo");
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            ModifiedFileSet.builder().modify(PathFragment.create("pkg/blah.foo")).build(),
            Root.fromPath(rootDirectory));

    reporter.removeHandler(failFastHandler);
    pkg = validPackage(skyKey);
    assertThat(pkg.containsErrors()).isTrue();
    String expectedEventString =
        "glob pattern '*.foo' didn't match anything, but allow_empty is set to False";
    MoreAsserts.assertContainsEvent(pkg.getEvents(), expectedEventString);
    assertContainsEvent(expectedEventString);
  }

  @Test
  public void testGlobDisallowEmpty_FunctionParam_WasEmptyAndStaysEmpty() throws Exception {
    scratch.file("pkg/BUILD", "x = " + "glob(['*.foo'], allow_empty=False)");
    invalidatePackages();

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//pkg"));
    reporter.removeHandler(failFastHandler);

    Package pkg = validPackage(skyKey);
    assertThat(pkg.containsErrors()).isTrue();
    String expectedEventString =
        "glob pattern '*.foo' didn't match anything, but allow_empty is set to False";
    MoreAsserts.assertContainsEvent(pkg.getEvents(), expectedEventString);
    assertContainsEvent(expectedEventString);

    scratch.overwriteFile("pkg/BUILD", "x = " + "glob(['*.foo'], allow_empty=False) #comment");
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            ModifiedFileSet.builder().modify(PathFragment.create("pkg/BUILD")).build(),
            Root.fromPath(rootDirectory));

    pkg = validPackage(skyKey);
    assertThat(pkg.containsErrors()).isTrue();
    MoreAsserts.assertContainsEvent(pkg.getEvents(), expectedEventString);
    assertContainsEvent(expectedEventString);
  }

  @Test
  public void testGlobDisallowEmpty_StarlarkOption_WasEmptyAndStaysEmpty() throws Exception {
    preparePackageLoadingWithCustomStarklarkSemanticsOptions(
        Options.parse(StarlarkSemanticsOptions.class, "--incompatible_disallow_empty_glob=true")
            .getOptions(),
        rootDirectory);

    scratch.file("pkg/BUILD", "x = " + "glob(['*.foo'])");
    invalidatePackages();

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//pkg"));
    reporter.removeHandler(failFastHandler);

    Package pkg = validPackage(skyKey);
    assertThat(pkg.containsErrors()).isTrue();
    String expectedEventString =
        "glob pattern '*.foo' didn't match anything, but allow_empty is set to False";
    MoreAsserts.assertContainsEvent(pkg.getEvents(), expectedEventString);
    assertContainsEvent(expectedEventString);

    scratch.overwriteFile("pkg/BUILD", "x = " + "glob(['*.foo']) #comment");
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            ModifiedFileSet.builder().modify(PathFragment.create("pkg/BUILD")).build(),
            Root.fromPath(rootDirectory));

    pkg = validPackage(skyKey);
    assertThat(pkg.containsErrors()).isTrue();
    MoreAsserts.assertContainsEvent(pkg.getEvents(), expectedEventString);
    assertContainsEvent(expectedEventString);
  }

  @Test
  public void testGlobDisallowEmpty_FunctionParam_WasEmptyDueToExcludeAndStaysEmpty()
      throws Exception {
    scratch.file("pkg/BUILD", "x = glob(include=['*.foo'], exclude=['blah.*'], allow_empty=False)");
    scratch.file("pkg/blah.foo");
    invalidatePackages();

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//pkg"));
    reporter.removeHandler(failFastHandler);

    Package pkg = validPackage(skyKey);
    assertThat(pkg.containsErrors()).isTrue();
    String expectedEventString =
        "all files in the glob have been excluded, but allow_empty is set to False.";
    MoreAsserts.assertContainsEvent(pkg.getEvents(), expectedEventString);
    assertContainsEvent(expectedEventString);

    scratch.overwriteFile(
        "pkg/BUILD",
        "x = glob(include=['*.foo'], exclude=['blah.*'], allow_empty=False) # comment");
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            ModifiedFileSet.builder().modify(PathFragment.create("pkg/BUILD")).build(),
            Root.fromPath(rootDirectory));

    pkg = validPackage(skyKey);
    assertThat(pkg.containsErrors()).isTrue();
    MoreAsserts.assertContainsEvent(pkg.getEvents(), expectedEventString);
    assertContainsEvent(expectedEventString);
  }

  @Test
  public void testGlobDisallowEmpty_StarlarkOption_WasEmptyDueToExcludeAndStaysEmpty()
      throws Exception {
    preparePackageLoadingWithCustomStarklarkSemanticsOptions(
        Options.parse(StarlarkSemanticsOptions.class, "--incompatible_disallow_empty_glob=true")
            .getOptions(),
        rootDirectory);

    scratch.file("pkg/BUILD", "x = glob(include=['*.foo'], exclude=['blah.*'])");
    scratch.file("pkg/blah.foo");
    invalidatePackages();

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//pkg"));
    reporter.removeHandler(failFastHandler);

    Package pkg = validPackage(skyKey);
    assertThat(pkg.containsErrors()).isTrue();
    String expectedEventString =
        "all files in the glob have been excluded, but allow_empty is set to False.";
    MoreAsserts.assertContainsEvent(pkg.getEvents(), expectedEventString);
    assertContainsEvent(expectedEventString);

    scratch.overwriteFile("pkg/BUILD", "x = glob(include=['*.foo'], exclude=['blah.*']) # comment");
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            ModifiedFileSet.builder().modify(PathFragment.create("pkg/BUILD")).build(),
            Root.fromPath(rootDirectory));

    pkg = validPackage(skyKey);
    assertThat(pkg.containsErrors()).isTrue();
    MoreAsserts.assertContainsEvent(pkg.getEvents(), expectedEventString);
    assertContainsEvent(expectedEventString);
  }

  @Test
  public void testGlobDisallowEmpty_FunctionParam_WasEmptyAndBecomesNonEmpty() throws Exception {
    scratch.file("pkg/BUILD", "x = " + "glob(['*.foo'], allow_empty=False)");
    invalidatePackages();

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//pkg"));

    reporter.removeHandler(failFastHandler);
    Package pkg = validPackage(skyKey);
    assertThat(pkg.containsErrors()).isTrue();
    String expectedEventString =
        "glob pattern '*.foo' didn't match anything, but allow_empty is set to False";
    MoreAsserts.assertContainsEvent(pkg.getEvents(), expectedEventString);
    assertContainsEvent(expectedEventString);

    scratch.file("pkg/blah.foo");
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            ModifiedFileSet.builder().modify(PathFragment.create("pkg/blah.foo")).build(),
            Root.fromPath(rootDirectory));

    reporter.addHandler(failFastHandler);
    pkg = validPackage(skyKey);
    assertThat(pkg.containsErrors()).isFalse();
    assertThat(pkg.getEvents()).isEmpty();
  }

  @Test
  public void testGlobDisallowEmpty_StarlarkOption_WasEmptyAndBecomesNonEmpty() throws Exception {
    preparePackageLoadingWithCustomStarklarkSemanticsOptions(
        Options.parse(StarlarkSemanticsOptions.class, "--incompatible_disallow_empty_glob=true")
            .getOptions(),
        rootDirectory);

    scratch.file("pkg/BUILD", "x = " + "glob(['*.foo'])");
    invalidatePackages();

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("@//pkg"));

    reporter.removeHandler(failFastHandler);
    Package pkg = validPackage(skyKey);
    assertThat(pkg.containsErrors()).isTrue();
    String expectedEventString =
        "glob pattern '*.foo' didn't match anything, but allow_empty is set to False";
    MoreAsserts.assertContainsEvent(pkg.getEvents(), expectedEventString);
    assertContainsEvent(expectedEventString);

    scratch.file("pkg/blah.foo");
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            ModifiedFileSet.builder().modify(PathFragment.create("pkg/blah.foo")).build(),
            Root.fromPath(rootDirectory));

    reporter.addHandler(failFastHandler);
    pkg = validPackage(skyKey);
    assertThat(pkg.containsErrors()).isFalse();
    assertThat(pkg.getEvents()).isEmpty();
  }

  @Test
  public void veryBrokenPackagePostsDoneToProgressReceiver() throws Exception {
    reporter.removeHandler(failFastHandler);

    scratch.file("pkg/BUILD", "load('//does_not:exist.bzl', 'broken'");
    SkyKey key = PackageValue.key(PackageIdentifier.parse("@//pkg"));
    EvaluationResult<PackageValue> result =
        SkyframeExecutorTestUtils.evaluate(getSkyframeExecutor(), key, false, reporter);
    assertThatEvaluationResult(result).hasError();
    assertThat(getSkyframeExecutor().getPackageProgressReceiver().progressState())
        .isEqualTo(new Pair<String, String>("1 packages loaded", ""));
  }

  @Test
  public void testLegacyGlobbingEncountersSymlinkCycleAndThrowsIOException() throws Exception {
    reporter.removeHandler(failFastHandler);
    getSkyframeExecutor().turnOffSyscallCacheForTesting();

    // When a package's BUILD file and the relevant filesystem state is such that legacy globbing
    // will encounter an IOException due to a directory symlink cycle,
    Path fooBUILDPath = scratch.file("foo/BUILD", "glob(['cycle/**/foo.txt'])");
    Path fooCyclePath = fooBUILDPath.getParentDirectory().getChild("cycle");
    FileSystemUtils.ensureSymbolicLink(fooCyclePath, fooCyclePath);
    IOException ioExnFromFS =
        assertThrows(IOException.class, () -> fooCyclePath.statIfFound(Symlinks.FOLLOW));
    // And it is indeed the case that the FileSystem throws an IOException when the cycle's Path is
    // stat'd (following symlinks, as legacy globbing does).
    assertThat(ioExnFromFS).hasMessageThat().contains("Too many levels of symbolic links");

    // Then, when we evaluate the PackageValue node for the Package in keepGoing mode,
    SkyKey pkgKey = PackageValue.key(PackageIdentifier.parse("@//foo"));
    EvaluationResult<PackageValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), pkgKey, /*keepGoing=*/ true, reporter);
    // The result is a *non-transient* Skyframe error.
    assertThatEvaluationResult(result).hasErrorEntryForKeyThat(pkgKey).isNotTransient();
    // And that error is a NoSuchPackageException
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(pkgKey)
        .hasExceptionThat()
        .isInstanceOf(NoSuchPackageException.class);
    // With a useful error message,
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(pkgKey)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("Symlink cycle: /workspace/foo/cycle");
    // And appropriate Skyframe root cause (N.B. since we want PackageFunction to rethrow in
    // situations like this, we want the PackageValue node to be its own root cause).
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(pkgKey)
        .rootCauseOfExceptionIs(pkgKey);

    // Then, when we modify the BUILD file so as to force package loading,
    scratch.overwriteFile(
        "foo/BUILD", "glob(['cycle/**/foo.txt']) # dummy comment to force package loading");
    // But we don't make any filesystem changes that would invalidate the GlobValues, meaning that
    // PackageFunction will observe cache hits from Skyframe globbing,
    //
    // And we also have our filesystem blow up if the directory symlink cycle is encountered (thus,
    // the absence of a crash indicates the lack of legacy globbing),
    fs.stubStatError(
        fooCyclePath,
        new IOException() {
          @Override
          public String getMessage() {
            throw new IllegalStateException("should't get here!");
          }
        });
    // And we evaluate the PackageValue node for the Package in keepGoing mode,
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            ModifiedFileSet.builder().modify(PathFragment.create("foo/BUILD")).build(),
            Root.fromPath(rootDirectory));
    // The results are exactly the same as before,
    result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), pkgKey, /*keepGoing=*/ true, reporter);
    assertThatEvaluationResult(result).hasErrorEntryForKeyThat(pkgKey).isNotTransient();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(pkgKey)
        .hasExceptionThat()
        .isInstanceOf(NoSuchPackageException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(pkgKey)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("Symlink cycle: /workspace/foo/cycle");
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(pkgKey)
        .rootCauseOfExceptionIs(pkgKey);
    // Thus showing that clean and incremental package loading have the same semantics in the
    // presence of a symlink cycle encountered during glob evaluation.
  }

  private static class CustomInMemoryFs extends InMemoryFileSystem {
    private abstract static class FileStatusOrException {
      abstract FileStatus get() throws IOException;

      private static class ExceptionImpl extends FileStatusOrException {
        private final IOException exn;

        private ExceptionImpl(IOException exn) {
          this.exn = exn;
        }

        @Override
        FileStatus get() throws IOException {
          throw exn;
        }
      }

      private static class FileStatusImpl extends FileStatusOrException {

        @Nullable private final FileStatus fileStatus;

        private FileStatusImpl(@Nullable FileStatus fileStatus) {
          this.fileStatus = fileStatus;
        }

        @Override
        @Nullable
        FileStatus get() {
          return fileStatus;
        }
      }
    }

    private final Map<Path, FileStatusOrException> stubbedStats = Maps.newHashMap();
    private final Set<Path> makeUnreadableAfterReaddir = Sets.newHashSet();
    private final Map<Path, IOException> pathsToErrorOnGetInputStream = Maps.newHashMap();

    public CustomInMemoryFs(ManualClock manualClock) {
      super(manualClock);
    }

    public void stubStat(Path path, @Nullable FileStatus stubbedResult) {
      stubbedStats.put(path, new FileStatusOrException.FileStatusImpl(stubbedResult));
    }

    public void stubStatError(Path path, IOException stubbedResult) {
      stubbedStats.put(path, new FileStatusOrException.ExceptionImpl(stubbedResult));
    }

    @Override
    public FileStatus statIfFound(Path path, boolean followSymlinks) throws IOException {
      if (stubbedStats.containsKey(path)) {
        return stubbedStats.get(path).get();
      }
      return super.statIfFound(path, followSymlinks);
    }

    public void scheduleMakeUnreadableAfterReaddir(Path path) {
      makeUnreadableAfterReaddir.add(path);
    }

    @Override
    public Collection<Dirent> readdir(Path path, boolean followSymlinks) throws IOException {
      Collection<Dirent> result = super.readdir(path, followSymlinks);
      if (makeUnreadableAfterReaddir.contains(path)) {
        path.setReadable(false);
      }
      return result;
    }

    public void throwExceptionOnGetInputStream(Path path, IOException exn) {
      pathsToErrorOnGetInputStream.put(path, exn);
    }

    @Override
    protected InputStream getInputStream(Path path) throws IOException {
      IOException exnToThrow = pathsToErrorOnGetInputStream.get(path);
      if (exnToThrow != null) {
        throw exnToThrow;
      }
      return super.getInputStream(path);
    }
  }
}
