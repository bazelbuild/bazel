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
package com.google.devtools.build.lib.skyframe.packages;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertContainsEvent;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertNoEvents;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Before;
import org.junit.Test;

/** Abstract base class of a unit test for a {@link AbstractPackageLoader} implementation. */
public abstract class AbstractPackageLoaderTest {
  protected Path workspaceDir;
  protected StoredEventHandler handler;
  protected FileSystem fs;
  protected Root root;
  private Reporter reporter;

  @Before
  public final void init() throws Exception {
    fs = new InMemoryFileSystem();
    workspaceDir = fs.getPath("/workspace/");
    workspaceDir.createDirectoryAndParents();
    root = Root.fromPath(workspaceDir);
    reporter = new Reporter(new EventBus());
    handler = new StoredEventHandler();
    reporter.addHandler(handler);
  }

  protected abstract AbstractPackageLoader.Builder newPackageLoaderBuilder(Root workspaceDir);

  private AbstractPackageLoader.Builder newPackageLoaderBuilder() {
    return newPackageLoaderBuilder(root).useDefaultStarlarkSemantics().setReporter(reporter);
  }

  protected PackageLoader newPackageLoader() {
    return newPackageLoaderBuilder().build();
  }

  @Test
  public void simpleNoPackage() throws Exception {
    PackageLoader pkgLoader = newPackageLoader();
    PackageIdentifier pkgId = PackageIdentifier.createInMainRepo(PathFragment.create("nope"));
    NoSuchPackageException expected =
        assertThrows(NoSuchPackageException.class, () -> pkgLoader.loadPackage(pkgId));
    assertThat(expected)
        .hasMessageThat()
        .startsWith("no such package 'nope': BUILD file not found");
    assertNoEvents(handler.getEvents());
  }

  @Test
  public void simpleBadPackage() throws Exception {
    PackageLoader pkgLoader = newPackageLoader();
    file("bad/BUILD", "invalidBUILDsyntax");
    PackageIdentifier pkgId = PackageIdentifier.createInMainRepo(PathFragment.create("bad"));
    Package badPkg = pkgLoader.loadPackage(pkgId);
    assertThat(badPkg.containsErrors()).isTrue();
    assertContainsEvent(badPkg.getEvents(), "invalidBUILDsyntax");
    assertContainsEvent(handler.getEvents(), "invalidBUILDsyntax");
  }

  @Test
  public void simpleGoodPackage() throws Exception {
    PackageLoader pkgLoader = newPackageLoader();
    file("good/BUILD", "sh_library(name = 'good')");
    PackageIdentifier pkgId = PackageIdentifier.createInMainRepo(PathFragment.create("good"));
    Package goodPkg = pkgLoader.loadPackage(pkgId);
    assertThat(goodPkg.containsErrors()).isFalse();
    assertThat(goodPkg.getTarget("good").getAssociatedRule().getRuleClass())
        .isEqualTo("sh_library");
    assertNoEvents(goodPkg.getEvents());
    assertNoEvents(handler.getEvents());
  }

  @Test
  public void simpleMultipleGoodPackage() throws Exception {
    PackageLoader pkgLoader = newPackageLoader();
    file("good1/BUILD", "sh_library(name = 'good1')");
    file("good2/BUILD", "sh_library(name = 'good2')");
    PackageIdentifier pkgId1 = PackageIdentifier.createInMainRepo(PathFragment.create("good1"));
    PackageIdentifier pkgId2 = PackageIdentifier.createInMainRepo(PathFragment.create("good2"));
    ImmutableMap<PackageIdentifier, PackageLoader.PackageOrException> pkgs =
        pkgLoader.loadPackages(ImmutableList.of(pkgId1, pkgId2));
    assertThat(pkgs.get(pkgId1).get().containsErrors()).isFalse();
    assertThat(pkgs.get(pkgId2).get().containsErrors()).isFalse();
    assertThat(pkgs.get(pkgId1).get().getTarget("good1").getAssociatedRule().getRuleClass())
        .isEqualTo("sh_library");
    assertThat(pkgs.get(pkgId2).get().getTarget("good2").getAssociatedRule().getRuleClass())
        .isEqualTo("sh_library");
    assertNoEvents(pkgs.get(pkgId1).get().getEvents());
    assertNoEvents(pkgs.get(pkgId2).get().getEvents());
    assertNoEvents(handler.getEvents());
  }

  @Test
  public void loadPackagesToleratesDuplicates() throws Exception {
    PackageLoader pkgLoader = newPackageLoader();
    file("good1/BUILD", "sh_library(name = 'good1')");
    PackageIdentifier pkgId = PackageIdentifier.createInMainRepo(PathFragment.create("good1"));
    ImmutableMap<PackageIdentifier, PackageLoader.PackageOrException> pkgs =
        pkgLoader.loadPackages(ImmutableList.of(pkgId, pkgId));
    assertThat(pkgs.get(pkgId).get().containsErrors()).isFalse();
    assertThat(pkgs.get(pkgId).get().getTarget("good1").getAssociatedRule().getRuleClass())
        .isEqualTo("sh_library");
    assertNoEvents(pkgs.get(pkgId).get().getEvents());
    assertNoEvents(handler.getEvents());
  }

  @Test
  public void simpleGoodPackage_Skylark() throws Exception {
    PackageLoader pkgLoader = newPackageLoader();
    file("good/good.bzl", "def f(x):", "  native.sh_library(name = x)");
    file("good/BUILD", "load('//good:good.bzl', 'f')", "f('good')");
    PackageIdentifier pkgId = PackageIdentifier.createInMainRepo(PathFragment.create("good"));
    Package goodPkg = pkgLoader.loadPackage(pkgId);
    assertThat(goodPkg.containsErrors()).isFalse();
    assertThat(goodPkg.getTarget("good").getAssociatedRule().getRuleClass())
        .isEqualTo("sh_library");
    assertNoEvents(goodPkg.getEvents());
    assertNoEvents(handler.getEvents());
  }

  @Test
  public void externalFile_SupportedByDefault() throws Exception {
    Path externalPath = file(absolutePath("/external/BUILD"), "sh_library(name = 'foo')");
    symlink("foo/BUILD", externalPath);

    PackageLoader pkgLoader = newPackageLoader();

    PackageIdentifier pkgId = PackageIdentifier.createInMainRepo(PathFragment.create("foo"));
    Package fooPkg = pkgLoader.loadPackage(pkgId);
    assertThat(fooPkg.containsErrors()).isFalse();
    assertThat(fooPkg.getTarget("foo").getTargetKind()).isEqualTo("sh_library rule");
    assertNoEvents(fooPkg.getEvents());
    assertNoEvents(handler.getEvents());
  }

  @Test
  public void externalFile_AssumeNonExistentAndImmutable() throws Exception {
    Path externalPath = file(absolutePath("/external/BUILD"), "sh_library(name = 'foo')");
    symlink("foo/BUILD", externalPath);

    PackageLoader pkgLoader =
        newPackageLoaderBuilder()
            .setExternalFileAction(
                ExternalFileAction.ASSUME_NON_EXISTENT_AND_IMMUTABLE_FOR_EXTERNAL_PATHS)
            .build();

    PackageIdentifier pkgId = PackageIdentifier.createInMainRepo(PathFragment.create("foo"));
    NoSuchPackageException expected =
        assertThrows(NoSuchPackageException.class, () -> pkgLoader.loadPackage(pkgId));
    assertThat(expected).hasMessageThat().contains("no such package 'foo': BUILD file not found");
  }

  protected Path path(String rootRelativePath) {
    return workspaceDir.getRelative(PathFragment.create(rootRelativePath));
  }

  protected Path absolutePath(String absolutePath) {
    return fs.getPath(absolutePath);
  }

  protected Path file(String fileName, String... contents) throws Exception {
    return file(path(fileName), contents);
  }

  protected Path file(Path path, String... contents) throws Exception {
    path.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(path, Joiner.on("\n").join(contents));
    return path;
  }

  protected Path symlink(String linkPathString, Path linkTargetPath) throws Exception {
    Path path = path(linkPathString);
    FileSystemUtils.ensureSymbolicLink(path, linkTargetPath);
    return path;
  }
}
