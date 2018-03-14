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

import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
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
  }

  private void mockEmbeddedTools(Path embeddedBinaries) throws IOException {
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
  }

  @Override
  protected AbstractPackageLoader.Builder newPackageLoaderBuilder(Path workspaceDir) {
    return BazelPackageLoader.builder(workspaceDir, installBase, outputBase);
  }

  @Test
  public void simpleLocalRepositoryPackage() throws Exception {
    PackageLoader pkgLoader = newPackageLoader();
    file("WORKSPACE", "local_repository(name = 'r', path='r')");
    file("r/WORKSPACE", "workspace(name = 'r')");
    file("r/good/BUILD", "sh_library(name = 'good')");
    PackageIdentifier pkgId =
        PackageIdentifier.create(RepositoryName.create("@r"), PathFragment.create("good"));
    Package goodPkg = pkgLoader.loadPackage(pkgId);
    assertThat(goodPkg.containsErrors()).isFalse();
    assertThat(goodPkg.getTarget("good").getAssociatedRule().getRuleClass())
        .isEqualTo("sh_library");
    assertNoEvents(goodPkg.getEvents());
    assertNoEvents(handler.getEvents());
  }

  @Test
  public void newLocalRepository() throws Exception {
    PackageLoader pkgLoader = newPackageLoader();
    file(
        "WORKSPACE",
        "new_local_repository(name = 'r', path = '/r', "
            + "build_file_content = 'sh_library(name = \"good\")')");
    fs.getPath("/r").createDirectoryAndParents();
    PackageIdentifier pkgId =
        PackageIdentifier.create(RepositoryName.create("@r"), PathFragment.create(""));
    Package goodPkg = pkgLoader.loadPackage(pkgId);
    assertThat(goodPkg.containsErrors()).isFalse();
    assertThat(goodPkg.getTarget("good").getAssociatedRule().getRuleClass())
        .isEqualTo("sh_library");
    assertNoEvents(goodPkg.getEvents());
    assertNoEvents(handler.getEvents());
  }
}
