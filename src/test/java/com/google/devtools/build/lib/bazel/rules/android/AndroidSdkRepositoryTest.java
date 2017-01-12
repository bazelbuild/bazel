// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.android;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link AndroidSdkRepositoryFunction}. */
@RunWith(JUnit4.class)
public class AndroidSdkRepositoryTest extends BuildViewTestCase {

  @Before
  public void setup() throws Exception {
    scratch.setWorkingDir("/sdk");
    scratch.dir("platforms/android-25");
    scratch.file("extras/google/m2repository/com/google/android/foo/1.0.0/foo.pom",
        "<project>",
        "  <groupId>com.google.android</groupId>",
        "  <artifactId>foo</artifactId>",
        "  <version>1.0.0</version>",
        "  <packaging>aar</packaging>",
        "</project>");

    scratch.setWorkingDir("/workspace");
    FileSystemUtils.appendIsoLatin1(scratch.resolve("WORKSPACE"),
        "android_sdk_repository(",
        "    name = 'mysdk',",
        "    path = '/sdk',",
        "    build_tools_version = '25.0.0',",
        "    api_level = 25,",
        ")");
  }

  @Test
  public void testGeneratedAarImport() throws Exception {
    invalidatePackages();
    ConfiguredTarget aarImportTarget = getConfiguredTarget("@mysdk//com.google.android:foo-1.0.0");
    assertThat(aarImportTarget.getTarget().getAssociatedRule().getRuleClass())
        .isEqualTo("aar_import");
  }

  @Test
  public void testExportsFiles() throws Exception {
    invalidatePackages();
    ConfiguredTarget aarTarget = getConfiguredTarget(
        "@mysdk//:extras/google/m2repository/com/google/android/foo/1.0.0/foo.aar");
    assertThat(aarTarget).isNotNull();
  }
}
