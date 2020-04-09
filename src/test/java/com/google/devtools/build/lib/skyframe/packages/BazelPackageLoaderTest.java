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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertNoEvents;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Simple tests for {@link BazelPackageLoader}.
 *
 * <p>Bazel's unit and integration tests do sanity checks with {@link BazelPackageLoader} under the
 * covers, so we get pretty exhaustive correctness tests for free.
 */
@RunWith(JUnit4.class)
public final class BazelPackageLoaderTest extends AbstractPackageLoaderTest {

  private Path installBase;
  private Path outputBase;

  @Before
  public void setUp() throws Exception {
    installBase = fs.getPath("/installBase/");
    installBase.createDirectoryAndParents();
    outputBase = fs.getPath("/outputBase/");
    outputBase.createDirectoryAndParents();
    Path embeddedBinaries = ServerDirectories.getEmbeddedBinariesRoot(installBase);
    embeddedBinaries.createDirectoryAndParents();

    mockEmbeddedTools(embeddedBinaries);
    fetchExternalRepo(RepositoryName.create("@bazel_tools"));
  }

  private static void mockEmbeddedTools(Path embeddedBinaries) throws IOException {
    Path tools = embeddedBinaries.getRelative("embedded_tools");
    tools.getRelative("tools/cpp").createDirectoryAndParents();
    tools.getRelative("tools/osx").createDirectoryAndParents();
    FileSystemUtils.writeIsoLatin1(tools.getRelative("WORKSPACE"), "");
    FileSystemUtils.writeIsoLatin1(tools.getRelative("tools/cpp/BUILD"), "");
    FileSystemUtils.writeIsoLatin1(
        tools.getRelative("tools/cpp/cc_configure.bzl"),
        "def cc_configure(*args, **kwargs):",
        "    pass");
    FileSystemUtils.writeIsoLatin1(tools.getRelative("tools/osx/BUILD"), "");
    FileSystemUtils.writeIsoLatin1(
        tools.getRelative("tools/osx/xcode_configure.bzl"),
        "def xcode_configure(*args, **kwargs):",
        "    pass");
    FileSystemUtils.writeIsoLatin1(tools.getRelative("tools/sh/BUILD"), "");
    FileSystemUtils.writeIsoLatin1(
        tools.getRelative("tools/sh/sh_configure.bzl"),
        "def sh_configure(*args, **kwargs):",
        "    pass");
    FileSystemUtils.writeIsoLatin1(tools.getRelative("tools/build_defs/repo/BUILD"));
    FileSystemUtils.writeIsoLatin1(
        tools.getRelative("tools/build_defs/repo/http.bzl"),
        "def http_archive(**kwargs):",
        "  pass",
        "",
        "def http_file(**kwargs):",
        "  pass");
    FileSystemUtils.writeIsoLatin1(
        tools.getRelative("tools/build_defs/repo/utils.bzl"),
        "def maybe(repo_rule, name, **kwargs):",
        "  if name not in native.existing_rules():",
        "    repo_rule(name = name, **kwargs)");
  }

  private void fetchExternalRepo(RepositoryName externalRepo) {
    PackageLoader pkgLoaderForFetch =
        newPackageLoaderBuilder(root)
            .setFetchForTesting()
            .useDefaultSkylarkSemantics()
            .build();
    // Load the package '' in this repo. This package may or may not exist; we don't care since we
    // merely need the side-effects of the 'fetch' work.
    PackageIdentifier pkgId = PackageIdentifier.create(externalRepo, PathFragment.create(""));
    try {
      pkgLoaderForFetch.loadPackage(pkgId);
    } catch (NoSuchPackageException | InterruptedException e) {
      // Doesn't matter; see above comment.
    }
  }

  @Override
  protected BazelPackageLoader.Builder newPackageLoaderBuilder(Root workspaceDir) {
    return BazelPackageLoader.builder(workspaceDir, installBase, outputBase);
  }

  @Test
  public void simpleLocalRepositoryPackage() throws Exception {
    file("WORKSPACE", "local_repository(name = 'r', path='r')");
    file("r/WORKSPACE", "workspace(name = 'r')");
    file("r/good/BUILD", "sh_library(name = 'good')");
    RepositoryName rRepoName = RepositoryName.create("@r");
    fetchExternalRepo(rRepoName);

    PackageLoader pkgLoader = newPackageLoader();
    PackageIdentifier pkgId = PackageIdentifier.create(rRepoName, PathFragment.create("good"));
    Package goodPkg = pkgLoader.loadPackage(pkgId);
    assertThat(goodPkg.containsErrors()).isFalse();
    assertThat(goodPkg.getTarget("good").getAssociatedRule().getRuleClass())
        .isEqualTo("sh_library");
    assertNoEvents(goodPkg.getEvents());
    assertNoEvents(handler.getEvents());
  }

  @Test
  public void newLocalRepository() throws Exception {
    file(
        "WORKSPACE",
        "new_local_repository(name = 'r', path = '/r', "
            + "build_file_content = 'sh_library(name = \"good\")')");
    fs.getPath("/r").createDirectoryAndParents();
    RepositoryName rRepoName = RepositoryName.create("@r");
    fetchExternalRepo(rRepoName);

    PackageLoader pkgLoader = newPackageLoader();
    PackageIdentifier pkgId =
        PackageIdentifier.create(rRepoName, PathFragment.create(""));
    Package goodPkg = pkgLoader.loadPackage(pkgId);
    assertThat(goodPkg.containsErrors()).isFalse();
    assertThat(goodPkg.getTarget("good").getAssociatedRule().getRuleClass())
        .isEqualTo("sh_library");
    assertNoEvents(goodPkg.getEvents());
    assertNoEvents(handler.getEvents());
  }

  @Test
  public void buildDotBazelForSubpackageCheckDuringGlobbing() throws Exception {
    file("a/BUILD", "filegroup(name = 'fg', srcs = glob(['sub/a.txt']))");
    file("a/sub/a.txt");
    file("a/sub/BUILD.bazel");

    PackageLoader pkgLoader = newPackageLoader();
    PackageIdentifier pkgId = PackageIdentifier.createInMainRepo(PathFragment.create("a"));
    Package aPkg = pkgLoader.loadPackage(pkgId);
    assertThat(aPkg.containsErrors()).isFalse();
    assertThrows(NoSuchTargetException.class, () -> aPkg.getTarget("sub/a.txt"));
    assertNoEvents(aPkg.getEvents());
    assertNoEvents(handler.getEvents());
  }
}
