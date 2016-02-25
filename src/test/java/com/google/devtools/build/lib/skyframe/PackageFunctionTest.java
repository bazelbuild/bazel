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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.ConstantRuleVisibility;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Preprocessor;
import com.google.devtools.build.lib.packages.util.SubincludePreprocessor;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.util.Collection;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

import javax.annotation.Nullable;

/**
 * Unit tests of specific functionality of PackageFunction. Note that it's already tested
 * indirectly in several other places.
 */
@RunWith(JUnit4.class)
public class PackageFunctionTest extends BuildViewTestCase {

  private CustomInMemoryFs fs = new CustomInMemoryFs(new ManualClock());

  @Override
  protected Preprocessor.Factory.Supplier getPreprocessorFactorySupplier() {
    return new SubincludePreprocessor.FactorySupplier(scratch.getFileSystem());
  }

  @Override
  protected FileSystem createFileSystem() {
    return fs;
  }

  private PackageValue validPackage(SkyKey skyKey) throws InterruptedException {
    EvaluationResult<PackageValue> result = SkyframeExecutorTestUtils.evaluate(
        getSkyframeExecutor(), skyKey, /*keepGoing=*/false, reporter);
    if (result.hasError()) {
      fail(result.getError(skyKey).getException().getMessage());
    }
    PackageValue value = result.get(skyKey);
    assertFalse(value.getPackage().containsErrors());
    return value;
  }

  @Test
  public void testInconsistentNewPackage() throws Exception {
    scratch.file("pkg/BUILD", "subinclude('//foo:sub')");
    scratch.file("foo/sub");

    getSkyframeExecutor().preparePackageLoading(
        new PathPackageLocator(outputBase, ImmutableList.of(rootDirectory)),
        ConstantRuleVisibility.PUBLIC, true,
        7, "", UUID.randomUUID());

    SkyKey pkgLookupKey = PackageLookupValue.key(new PathFragment("foo"));
    EvaluationResult<PackageLookupValue> result = SkyframeExecutorTestUtils.evaluate(
        getSkyframeExecutor(), pkgLookupKey, /*keepGoing=*/false, reporter);
    assertFalse(result.hasError());
    assertFalse(result.get(pkgLookupKey).packageExists());

    scratch.file("foo/BUILD");

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("pkg"));
    result = SkyframeExecutorTestUtils.evaluate(getSkyframeExecutor(),
        skyKey, /*keepGoing=*/false, reporter);
    assertTrue(result.hasError());
    Throwable exception = result.getError(skyKey).getException();
    assertThat(exception.getMessage()).contains("Inconsistent filesystem operations");
    assertThat(exception.getMessage()).contains("Unexpected package");
  }

  @Test
  public void testInconsistentMissingPackage() throws Exception {
    reporter.removeHandler(failFastHandler);
    Path root1 = fs.getPath("/root1");
    scratch.file("/root1/WORKSPACE");
    scratch.file("/root1/foo/sub");
    scratch.file("/root1/pkg/BUILD", "subinclude('//foo:sub')");

    Path root2 = fs.getPath("/root2");
    scratch.file("/root2/foo/BUILD");
    scratch.file("/root2/foo/sub");

    getSkyframeExecutor().preparePackageLoading(
        new PathPackageLocator(outputBase, ImmutableList.of(root1, root2)),
        ConstantRuleVisibility.PUBLIC, true,
        7, "", UUID.randomUUID());

    SkyKey pkgLookupKey = PackageLookupValue.key(PackageIdentifier.parse("foo"));
    EvaluationResult<PackageLookupValue> result = SkyframeExecutorTestUtils.evaluate(
        getSkyframeExecutor(), pkgLookupKey, /*keepGoing=*/false, reporter);
    assertFalse(result.hasError());
    assertEquals(root2, result.get(pkgLookupKey).getRoot());

    scratch.file("/root1/foo/BUILD");

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("pkg"));
    result = SkyframeExecutorTestUtils.evaluate(getSkyframeExecutor(),
        skyKey, /*keepGoing=*/false, reporter);
    assertTrue(result.hasError());
    Throwable exception = result.getError(skyKey).getException();
    System.out.println("exception: " + exception.getMessage());
    assertThat(exception.getMessage()).contains("Inconsistent filesystem operations");
    assertThat(exception.getMessage()).contains("Inconsistent package location");
  }

  @Test
  public void testPropagatesFilesystemInconsistencies() throws Exception {
    reporter.removeHandler(failFastHandler);
    RecordingDifferencer differencer = getSkyframeExecutor().getDifferencerForTesting();
    Path pkgRoot = getSkyframeExecutor().getPathEntries().get(0);
    Path fooBuildFile = scratch.file("foo/BUILD");
    Path fooDir = fooBuildFile.getParentDirectory();

    // Our custom filesystem says "foo/BUILD" exists but its parent "foo" is a file.
    FileStatus inconsistentParentFileStatus = new FileStatus() {
      @Override
      public boolean isFile() {
        return true;
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
    fs.stubStat(fooDir, inconsistentParentFileStatus);
    RootedPath pkgRootedPath = RootedPath.toRootedPath(pkgRoot, fooDir);
    SkyValue fooDirValue = FileStateValue.create(pkgRootedPath,
        getSkyframeExecutor().getTimestampGranularityMonitorForTesting());
    differencer.inject(ImmutableMap.of(FileStateValue.key(pkgRootedPath), fooDirValue));
    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("foo"));
    String expectedMessage = "/workspace/foo/BUILD exists but its parent path /workspace/foo isn't "
        + "an existing directory";
    EvaluationResult<PackageValue> result = SkyframeExecutorTestUtils.evaluate(
        getSkyframeExecutor(), skyKey, /*keepGoing=*/false, reporter);
    assertTrue(result.hasError());
    ErrorInfo errorInfo = result.getError(skyKey);
    String errorMessage = errorInfo.getException().getMessage();
    assertThat(errorMessage).contains("Inconsistent filesystem operations");
    assertThat(errorMessage).contains(expectedMessage);
  }

  @Test
  public void testPropagatesFilesystemInconsistencies_Globbing() throws Exception {
    reporter.removeHandler(failFastHandler);
    RecordingDifferencer differencer = getSkyframeExecutor().getDifferencerForTesting();
    Path pkgRoot = getSkyframeExecutor().getPathEntries().get(0);
    scratch.file("foo/BUILD",
        "subinclude('//a:a')",
        "sh_library(name = 'foo', srcs = glob(['bar/**/baz.sh']))");
    scratch.file("a/BUILD");
    scratch.file("a/a");
    Path bazFile = scratch.file("foo/bar/baz/baz.sh");
    Path bazDir = bazFile.getParentDirectory();
    Path barDir = bazDir.getParentDirectory();

    long bazFileNodeId = bazFile.stat().getNodeId();
    // Our custom filesystem says "foo/bar/baz" does not exist but it also says that "foo/bar"
    // has a child directory "baz".
    fs.stubStat(bazDir, null);
    RootedPath barDirRootedPath = RootedPath.toRootedPath(pkgRoot, barDir);
    FileStateValue barDirFileStateValue = FileStateValue.create(barDirRootedPath,
        getSkyframeExecutor().getTimestampGranularityMonitorForTesting());
    FileValue barDirFileValue = FileValue.value(barDirRootedPath, barDirFileStateValue,
        barDirRootedPath, barDirFileStateValue);
    DirectoryListingValue barDirListing = DirectoryListingValue.value(barDirRootedPath,
        barDirFileValue, DirectoryListingStateValue.create(ImmutableList.of(
            new Dirent("baz", Dirent.Type.DIRECTORY))));
    differencer.inject(ImmutableMap.of(DirectoryListingValue.key(barDirRootedPath), barDirListing));
    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("foo"));
    String expectedMessage = "Some filesystem operations implied /workspace/foo/bar/baz/baz.sh was "
        + "a regular file with size of 0 and mtime of 0 and nodeId of " + bazFileNodeId + " and "
        + "mtime of 0 but others made us think it was a nonexistent path";
    EvaluationResult<PackageValue> result = SkyframeExecutorTestUtils.evaluate(
        getSkyframeExecutor(), skyKey, /*keepGoing=*/false, reporter);
    assertTrue(result.hasError());
    ErrorInfo errorInfo = result.getError(skyKey);
    String errorMessage = errorInfo.getException().getMessage();
    assertThat(errorMessage).contains("Inconsistent filesystem operations");
    assertThat(errorMessage).contains(expectedMessage);
  }

  /** Regression test for unexpected exception type from PackageValue. */
  @Test
  public void testDiscrepancyBetweenLegacyAndSkyframePackageLoadingErrors() throws Exception {
    reporter.removeHandler(failFastHandler);
    Path fooBuildFile = scratch.file("foo/BUILD",
        "sh_library(name = 'foo', srcs = glob(['bar/*.sh']))");
    Path fooDir = fooBuildFile.getParentDirectory();
    Path barDir = fooDir.getRelative("bar");
    scratch.file("foo/bar/baz.sh");
    fs.scheduleMakeUnreadableAfterReaddir(barDir);

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("foo"));
    String expectedMessage = "Encountered error 'Directory is not readable'";
    EvaluationResult<PackageValue> result = SkyframeExecutorTestUtils.evaluate(
        getSkyframeExecutor(), skyKey, /*keepGoing=*/false, reporter);
    assertTrue(result.hasError());
    ErrorInfo errorInfo = result.getError(skyKey);
    String errorMessage = errorInfo.getException().getMessage();
    assertThat(errorMessage).contains("Inconsistent filesystem operations");
    assertThat(errorMessage).contains(expectedMessage);
  }

  @Test
  public void testMultipleSubincludesFromSamePackage() throws Exception {
    scratch.file("foo/BUILD",
        "subinclude('//bar:a')",
        "subinclude('//bar:b')");
    scratch.file("bar/BUILD",
        "exports_files(['a', 'b'])");
    scratch.file("bar/a");
    scratch.file("bar/b");

    getSkyframeExecutor().preparePackageLoading(
        new PathPackageLocator(outputBase, ImmutableList.of(rootDirectory)),
        ConstantRuleVisibility.PUBLIC, true,
        7, "", UUID.randomUUID());

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("foo"));
    validPackage(skyKey);
  }

  @Test
  public void testTransitiveSubincludesStoredInPackage() throws Exception {
    scratch.file("foo/BUILD",
        "subinclude('//bar:a')");
    scratch.file("bar/BUILD",
        "exports_files(['a'])");
    scratch.file("bar/a",
        "subinclude('//baz:b')");
    scratch.file("baz/BUILD",
        "exports_files(['b', 'c'])");
    scratch.file("baz/b");
    scratch.file("baz/c");

    getSkyframeExecutor().preparePackageLoading(
        new PathPackageLocator(outputBase, ImmutableList.of(rootDirectory)),
        ConstantRuleVisibility.PUBLIC, true,
        7, "", UUID.randomUUID());

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("foo"));
    PackageValue value = validPackage(skyKey);
    assertThat(value.getPackage().getSubincludeLabels()).containsExactly(
        Label.parseAbsolute("//bar:a"), Label.parseAbsolute("//baz:b"));

    scratch.overwriteFile("bar/a",
        "subinclude('//baz:c')");
    getSkyframeExecutor().invalidateFilesUnderPathForTesting(reporter,
        ModifiedFileSet.builder().modify(new PathFragment("bar/a")).build(), rootDirectory);

    value = validPackage(skyKey);
    assertThat(value.getPackage().getSubincludeLabels()).containsExactly(
        Label.parseAbsolute("//bar:a"), Label.parseAbsolute("//baz:c"));
  }

  @Test
  public void testTransitiveSkylarkDepsStoredInPackage() throws Exception {
    scratch.file("foo/BUILD",
        "load('/bar/ext', 'a')");
    scratch.file("bar/BUILD");
    scratch.file("bar/ext.bzl",
        "load('/baz/ext', 'b')",
        "a = b");
    scratch.file("baz/BUILD");
    scratch.file("baz/ext.bzl",
        "b = 1");
    scratch.file("qux/BUILD");
    scratch.file("qux/ext.bzl",
        "c = 1");

    getSkyframeExecutor().preparePackageLoading(
        new PathPackageLocator(outputBase, ImmutableList.of(rootDirectory)),
        ConstantRuleVisibility.PUBLIC, true,
        7, "", UUID.randomUUID());

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("foo"));
    PackageValue value = validPackage(skyKey);
    assertThat(value.getPackage().getSkylarkFileDependencies()).containsExactly(
        Label.parseAbsolute("//bar:ext.bzl"), Label.parseAbsolute("//baz:ext.bzl"));

    scratch.overwriteFile("bar/ext.bzl",
        "load('/qux/ext', 'c')",
        "a = c");
    getSkyframeExecutor().invalidateFilesUnderPathForTesting(reporter,
        ModifiedFileSet.builder().modify(new PathFragment("bar/ext.bzl")).build(), rootDirectory);

    value = validPackage(skyKey);
    assertThat(value.getPackage().getSkylarkFileDependencies()).containsExactly(
        Label.parseAbsolute("//bar:ext.bzl"), Label.parseAbsolute("//qux:ext.bzl"));
  }

  @Test
  public void testNonExistingSkylarkExtension() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("test/skylark/BUILD",
        "load('/test/skylark/bad_extension', 'some_symbol')",
        "genrule(name = gr,",
        "    outs = ['out.txt'],",
        "    cmd = 'echo hello >@')");
    invalidatePackages();

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("test/skylark"));
    EvaluationResult<PackageValue> result = SkyframeExecutorTestUtils.evaluate(
        getSkyframeExecutor(), skyKey, /*keepGoing=*/false, reporter);
    assertTrue(result.hasError());
    ErrorInfo errorInfo = result.getError(skyKey);
    String expectedMsg = "error loading package 'test/skylark': "
        + "Extension file not found. Unable to load file '//test/skylark:bad_extension.bzl': "
        + "file doesn't exist or isn't a file";
    assertThat(errorInfo.getException())
        .hasMessage(expectedMsg);
  }

  @Test
  public void testNonExistingSkylarkExtensionWithPythonPreprocessing() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("foo/BUILD",
        "exports_files(['a'])");
    scratch.file("foo/a",
        "load('/test/skylark/bad_extension', 'some_symbol')");
    scratch.file("test/skylark/BUILD",
        "subinclude('//foo:a')");
    invalidatePackages();

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("test/skylark"));
    EvaluationResult<PackageValue> result = SkyframeExecutorTestUtils.evaluate(
        getSkyframeExecutor(), skyKey, /*keepGoing=*/false, reporter);
    assertTrue(result.hasError());
    ErrorInfo errorInfo = result.getError(skyKey);
    String expectedMsg = "error loading package 'test/skylark': "
        + "Extension file not found. Unable to load file '//test/skylark:bad_extension.bzl': "
        + "file doesn't exist or isn't a file";
    assertThat(errorInfo.getException())
        .hasMessage(expectedMsg);
  }

  @Test
  public void testNonExistingSkylarkExtensionFromExtension() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("test/skylark/extension.bzl",
        "load('/test/skylark/bad_extension', 'some_symbol')",
        "a = 'a'");
    scratch.file("test/skylark/BUILD",
        "load('/test/skylark/extension', 'a')",
        "genrule(name = gr,",
        "    outs = ['out.txt'],",
        "    cmd = 'echo hello >@')");
    invalidatePackages();

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("test/skylark"));
    EvaluationResult<PackageValue> result = SkyframeExecutorTestUtils.evaluate(
        getSkyframeExecutor(), skyKey, /*keepGoing=*/false, reporter);
    assertTrue(result.hasError());
    ErrorInfo errorInfo = result.getError(skyKey);
    assertThat(errorInfo.getException())
        .hasMessage("error loading package 'test/skylark': Extension file not found. "
            + "Unable to load file '//test/skylark:bad_extension.bzl': "
            + "file doesn't exist or isn't a file");
  }

  @Test
  public void testSymlinkCycleWithSkylarkExtension() throws Exception {
    reporter.removeHandler(failFastHandler);
    Path extensionFilePath = scratch.resolve("/workspace/test/skylark/extension.bzl");
    FileSystemUtils.ensureSymbolicLink(extensionFilePath, new PathFragment("extension.bzl"));
    scratch.file("test/skylark/BUILD",
        "load('/test/skylark/extension', 'a')",
        "genrule(name = gr,",
        "    outs = ['out.txt'],",
        "    cmd = 'echo hello >@')");
    invalidatePackages();

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("test/skylark"));
    EvaluationResult<PackageValue> result = SkyframeExecutorTestUtils.evaluate(
        getSkyframeExecutor(), skyKey, /*keepGoing=*/false, reporter);
    assertTrue(result.hasError());
    ErrorInfo errorInfo = result.getError(skyKey);
    assertEquals(skyKey, errorInfo.getRootCauseOfException());
    assertThat(errorInfo.getException())
        .hasMessage(
            "error loading package 'test/skylark': Encountered error while reading extension "
            + "file 'test/skylark/extension.bzl': Symlink cycle");
  }

  @Test
  public void testIOErrorLookingForSubpackageForLabelIsHandled() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("foo/BUILD",
        "sh_library(name = 'foo', srcs = ['bar/baz.sh'])");
    Path barBuildFile = scratch.file("foo/bar/BUILD");
    fs.stubStatError(barBuildFile, new IOException("nope"));
    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("foo"));
    EvaluationResult<PackageValue> result = SkyframeExecutorTestUtils.evaluate(
        getSkyframeExecutor(), skyKey, /*keepGoing=*/false, reporter);
    assertTrue(result.hasError());
    assertContainsEvent("nope");
  }

  @Test
  public void testLoadRelativePath() throws Exception {
    scratch.file("pkg/BUILD", "load('ext', 'a')");
    scratch.file("pkg/ext.bzl", "a = 1");
    validPackage(PackageValue.key(PackageIdentifier.parse("pkg")));
  }

  @Test
  public void testLoadAbsolutePath() throws Exception {
    scratch.file("pkg1/BUILD");
    scratch.file("pkg2/BUILD",
        "load('/pkg1/ext', 'a')");
    scratch.file("pkg1/ext.bzl", "a = 1");
    validPackage(PackageValue.key(PackageIdentifier.parse("pkg2")));
  }

  @Test
  public void testBadWorkspaceFile() throws Exception {
    Path workspacePath = scratch.overwriteFile("WORKSPACE", "junk");
    SkyKey skyKey = PackageValue.key(PackageIdentifier.createInDefaultRepo("external"));
    getSkyframeExecutor()
        .invalidate(
            Predicates.equalTo(
                FileStateValue.key(
                    RootedPath.toRootedPath(
                        workspacePath.getParentDirectory(),
                        new PathFragment(workspacePath.getBaseName())))));

    reporter.removeHandler(failFastHandler);
    EvaluationResult<PackageValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), skyKey, /*keepGoing=*/ false, reporter);
    assertFalse(result.hasError());
    assertTrue(result.get(skyKey).getPackage().containsErrors());
  }

  // Regression test for the two ugly consequences of a bug where GlobFunction incorrectly matched
  // dangling symlinks.
  @Test
  public void testIncrementalSkyframeHybridGlobbingOnDanglingSymlink() throws Exception {
    Path packageDirPath = scratch.file("foo/BUILD",
        "exports_files(glob(['*.txt']))").getParentDirectory();
    scratch.file("foo/existing.txt");
    FileSystemUtils.ensureSymbolicLink(packageDirPath.getChild("dangling.txt"), "nope");

    getSkyframeExecutor().preparePackageLoading(
        new PathPackageLocator(outputBase, ImmutableList.of(rootDirectory)),
        ConstantRuleVisibility.PUBLIC, true,
        7, "", UUID.randomUUID());

    SkyKey skyKey = PackageValue.key(PackageIdentifier.parse("foo"));
    PackageValue value = validPackage(skyKey);
    assertFalse(value.getPackage().containsErrors());
    assertThat(value.getPackage().getTarget("existing.txt").getName()).isEqualTo("existing.txt");
    try {
      value.getPackage().getTarget("dangling.txt");
      fail();
    } catch (NoSuchTargetException expected) {
    }

    scratch.overwriteFile("foo/BUILD",
        "exports_files(glob(['*.txt'])),",
        "#some-irrelevant-comment");

    getSkyframeExecutor().invalidateFilesUnderPathForTesting(reporter,
        ModifiedFileSet.builder().modify(new PathFragment("foo/BUILD")).build(), rootDirectory);

    value = validPackage(skyKey);
    assertFalse(value.getPackage().containsErrors());
    assertThat(value.getPackage().getTarget("existing.txt").getName()).isEqualTo("existing.txt");
    try {
      value.getPackage().getTarget("dangling.txt");
      fail();
    } catch (NoSuchTargetException expected) {
      // One consequence of the bug was that dangling symlinks were matched by globs evaluated by
      // Skyframe globbing, meaning there would incorrectly be corresponding targets in packages
      // that had skyframe cache hits during skyframe hybrid globbing.
    }

    scratch.file("foo/nope");
    getSkyframeExecutor().invalidateFilesUnderPathForTesting(reporter,
        ModifiedFileSet.builder().modify(new PathFragment("foo/nope")).build(), rootDirectory);

    PackageValue newValue = validPackage(skyKey);
    assertFalse(newValue.getPackage().containsErrors());
    assertThat(newValue.getPackage().getTarget("existing.txt").getName()).isEqualTo("existing.txt");
    // Another consequence of the bug is that change pruning would incorrectly cut off changes that
    // caused a dangling symlink potentially matched by a glob to come into existence.
    assertThat(newValue.getPackage().getTarget("dangling.txt").getName()).isEqualTo("dangling.txt");
    assertThat(newValue.getPackage()).isNotSameAs(value.getPackage());
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

        @Nullable
        private final FileStatus fileStatus;

        private  FileStatusImpl(@Nullable FileStatus fileStatus) {
          this.fileStatus = fileStatus;
        }

        @Override
        @Nullable
        FileStatus get() {
          return fileStatus;
        }
      }
    }

    private Map<Path, FileStatusOrException> stubbedStats = Maps.newHashMap();
    private Set<Path> makeUnreadableAfterReaddir = Sets.newHashSet();

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
    public FileStatus stat(Path path, boolean followSymlinks) throws IOException {
      if (stubbedStats.containsKey(path)) {
        return stubbedStats.get(path).get();
      }
      return super.stat(path, followSymlinks);
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
  }
}
