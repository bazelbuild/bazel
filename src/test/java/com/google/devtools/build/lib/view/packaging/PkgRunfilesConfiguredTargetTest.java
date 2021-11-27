// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.view.packaging;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link PkgRunfiles}.
 */
@RunWith(JUnit4.class)
public class PkgRunfilesConfiguredTargetTest extends BuildViewTestCase {

  @Test
  public void testMultipleFiles() throws Exception {
    checkError("fruit", "grape", "must contain exactly one target",
        "sh_binary(name='charlie', srcs=['charlie.sh'])",
        "sh_binary(name='chaplin', srcs=['chaplin.sh'])",
        "pkg_runfiles(name='grape', srcs=[':charlie', ':chaplin'], package_path='')");
  }

  @Test
  public void testInvalidStripPrefix() throws Exception {
    checkError("octavius", "cherry", "strip prefix should be normalized",
        "sh_binary(name='indian', srcs=['indian.sh'])",
        "pkg_runfiles(name='cherry', srcs=[':indian'], package_path='', strip_prefix='a/../b')");

    checkError("lepidus", "cherry", "strip prefix should be a relative path",
        "sh_binary(name='wild', srcs=['wild.sh'])",
        "pkg_runfiles(name='cherry', srcs=[':wild'], package_path='', strip_prefix='/tu')");

    checkError("antony", "cherry", "not under specified strip prefix",
        "sh_binary(name='black', srcs=['black.sh'])",
        "pkg_runfiles(name='cherry', srcs=[':black'], package_path='', strip_prefix='melon')");

    checkError("cleopatra", "cherry", "not under specified strip prefix",
        "sh_binary(name='black', srcs=['black.sh'])",
        "pkg_runfiles(name='cherry', srcs=[':black'], package_path='', strip_prefix='goo')");

    checkError("augustus", "cherry", "not under specified strip prefix",
        "cc_binary(name='rainier', srcs=['rainier.cc'])",
        "pkg_runfiles(name='cherry', srcs=[':rainier'], package_path='', strip_prefix='goo')");
  }

  @Test
  public void testInvalidPackagePath() throws Exception {
    checkError("nero", "banana", "package path should be normalized",
        "sh_binary(name='chiquita', srcs=['chiquita.sh'])",
        "pkg_runfiles(name='banana', srcs=[':chiquita'], package_path='../nero')");
  }

  @Test
  public void testAbsolutePackagePath() throws Exception {
    checkError(
        "nero",
        "banana",
        "package path should be relative",
        "sh_binary(name='chiquita', srcs=['chiquita.sh'])",
        "pkg_runfiles(name='banana', srcs=[':chiquita'], package_path='/nero/the/fish')");
  }

  @Test
  public void testNotExecutable() throws Exception {
    checkError("fruit", "himbeer", "is not executable",
        "sh_library(name='berry')",
        "pkg_runfiles(name='himbeer', srcs=[':berry'], package_path='')");
  }

  @Test
  public void testValidMode() throws Exception {
    scratch.file("orange/BUILD",
        "sh_binary(name='lemon', srcs=['lemon.sh'])",
        "pkg_runfiles(name='orange', srcs=[':lemon'], package_path='', mode=0o7777)");
    getConfiguredTarget("//orange:orange");
  }

  @Test
  public void testInvalidMode() throws Exception {
    checkError("fruit", "orange", "should be a 12-bit Unix mode",
        "sh_binary(name='lemon', srcs=['lemon.sh'])",
        "pkg_runfiles(name='orange', srcs=[':lemon'], package_path='', mode=10000)");
  }

  @Test
  public void testAttributes() throws Exception {
    ConfiguredTarget target = scratchConfiguredTarget(
        "fruit", "cantaloupe",
        "cc_binary(name='honeydew', srcs=['honeydew.cc'])",
        "pkg_runfiles(name='cantaloupe',",
        "             srcs=[':honeydew'],",
        "             package_path='',",
        "             mode=0o777,",
        "             owner='echnaton',",
        "             group='pharao',",
        "             encrypted=1,",
        "             filegroups=['pyramid'])");

    PackagedFile packagedFile =
        target
            .get(PackageContentProvider.STARLARK_CONSTRUCTOR)
            .getTransitivePackagedFiles()
            .getSingleton();
    assertThat(packagedFile.getMode()).isEqualTo((Integer) 0777);
    assertThat(packagedFile.getOwner()).isEqualTo("echnaton");
    assertThat(packagedFile.getGroup()).isEqualTo("pharao");
    assertThat(packagedFile.getEncrypted()).isTrue();
    assertThat(packagedFile.getFilegroups()).containsExactly("pyramid").inOrder();
  }

  @Test
  public void testRootSymlinks() throws Exception {
    analysisMock.pySupport().setup(mockToolsConfig);
    ConfiguredTarget target = scratchConfiguredTarget(
        "fruit", "fig",
        "py_appengine_binary(name='kiwi', srcs=['kiwi.py'], configs=['kiwi.txt'])",
        "pkg_runfiles(name='fig',",
        "             srcs=[':kiwi'],",
        "             package_path='')");

    assertContainsPackagePaths(target, "kiwi.txt");
  }

  private static void assertContainsPackagePaths(
      ConfiguredTarget target, String... expected) {
    List<String> actual = new ArrayList<>();
    for (PackagedFile file :
        target
            .get(PackageContentProvider.STARLARK_CONSTRUCTOR)
            .getTransitivePackagedFiles()
            .toList()) {
      actual.add(file.getArchivePath());
    }

    assertThat(actual).containsAtLeastElementsIn(ImmutableList.copyOf(expected));
  }

}
