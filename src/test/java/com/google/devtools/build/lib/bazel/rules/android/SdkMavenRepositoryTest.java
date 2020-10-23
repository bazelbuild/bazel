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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.vfs.Path;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SdkMavenRepository}. */
@RunWith(JUnit4.class)
public class SdkMavenRepositoryTest extends BuildViewTestCase {
  Path workspaceDir;
  Path repoPath;
  SdkMavenRepository sdkMavenRepository;

  @Before
  public void setup() throws Exception {
    repoPath = scratch.dir("repo");
    workspaceDir = scratch.dir("");

    scratch.file("repo/com/google/android/foo/1.0.0/foo.pom",
        "<project>",
        "  <groupId>com.google.android</groupId>",
        "  <artifactId>foo</artifactId>",
        "  <version>1.0.0</version>",
        "</project>");
    scratch.file("repo/com/google/android/bar/1.0.0/bar.pom",
        "<project>",
        "  <groupId>com.google.android</groupId>",
        "  <artifactId>bar</artifactId>",
        "  <version>1.0.0</version>",
        "  <packaging>aar</packaging>",
        "  <dependencies>",
        "    <dependency>",
        "      <groupId>com.google.android</groupId>",
        "      <artifactId>foo</artifactId>",
        "      <version>1.0.0</version>",
        "    </dependency>",
        "    <dependency>",
        "      <groupId>com.google.android</groupId>",
        "      <artifactId>baz</artifactId>",
        "      <version>1.0.0</version>",
        "    </dependency>",
        "  </dependencies>",
        "</project>");
    scratch.file("repo/com/google/android/baz/1.0.0/baz.pom",
        "<project>",
        "  <groupId>com.google.android</groupId>",
        "  <artifactId>baz</artifactId>",
        "  <version>1.0.0</version>",
        "  <packaging>par</packaging>",
        "</project>");
    sdkMavenRepository = SdkMavenRepository.create(ImmutableList.of(repoPath));
    scratch.file("BUILD",
        "exports_files([",
        sdkMavenRepository.getExportsFiles(workspaceDir),
        "])");
  }

  @Test
  public void testExportsFiles() throws Exception {
    assertThat(sdkMavenRepository.getExportsFiles(workspaceDir)).isEqualTo(
        "    'repo/com/google/android/bar/1.0.0/bar.aar',\n"
        + "    'repo/com/google/android/baz/1.0.0/baz.par',\n"
        + "    'repo/com/google/android/foo/1.0.0/foo.jar',\n"
    );
  }

  @Test
  public void testBuildFilesWritten() throws Exception {
    sdkMavenRepository.writeBuildFiles(workspaceDir);

    Path groupIdPath = scratch.resolve("com.google.android");
    assertThat(workspaceDir.getDirectoryEntries()).containsAtLeast(repoPath, groupIdPath);

    Path buildFilePath = groupIdPath.getRelative("BUILD");
    assertThat(groupIdPath.getDirectoryEntries()).containsExactly(buildFilePath);
  }

  @Test
  public void testGeneratedAarImport() throws Exception {
    sdkMavenRepository.writeBuildFiles(workspaceDir);
    Rule aarImport =
        getConfiguredTargetAndData("//com.google.android:bar-1.0.0")
            .getTarget()
            .getAssociatedRule();
    assertThat(aarImport.getRuleClass()).isEqualTo("aar_import");
    AttributeMap attributes = RawAttributeMapper.of(aarImport);
    assertThat(attributes.get("aar", BuildType.LABEL))
        .isEqualTo(Label.parseAbsoluteUnchecked("//:repo/com/google/android/bar/1.0.0/bar.aar"));
    assertThat(attributes.get("exports", BuildType.LABEL_LIST))
        .containsExactly(Label.parseAbsoluteUnchecked("//com.google.android:foo-1.0.0"));
  }

  @Test
  public void testGeneratedJavaImport() throws Exception {
    sdkMavenRepository.writeBuildFiles(workspaceDir);
    Rule javaImport =
        getConfiguredTargetAndData("//com.google.android:foo-1.0.0")
            .getTarget()
            .getAssociatedRule();
    assertThat(javaImport.getRuleClass()).isEqualTo("java_import");
    AttributeMap attributes = RawAttributeMapper.of(javaImport);
    assertThat(attributes.get("jars", BuildType.LABEL_LIST)).containsExactly(
        Label.parseAbsoluteUnchecked("//:repo/com/google/android/foo/1.0.0/foo.jar"));
    assertThat(attributes.get("exports", BuildType.LABEL_LIST)).isEmpty();
  }

  @Test
  public void testGeneratedRuleForInvalidPackaging() throws Exception {
    sdkMavenRepository.writeBuildFiles(workspaceDir);
    Rule invalidPackagingGenrule =
        getConfiguredTargetAndData("//com.google.android:baz-1.0.0")
            .getTarget()
            .getAssociatedRule();
    assertThat(invalidPackagingGenrule.getRuleClass()).isEqualTo("genrule");
    assertThat(RawAttributeMapper.of(invalidPackagingGenrule).get("cmd", Type.STRING))
        .isEqualTo("echo Bazel does not recognize the Maven packaging type for: "
            + "\"//:repo/com/google/android/baz/1.0.0/baz.par\"; exit 1");
  }
}
