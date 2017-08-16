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

package com.google.devtools.build.lib.rules.repository;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.bazel.repository.DecompressorDescriptor;
import com.google.devtools.build.lib.bazel.repository.DecompressorValue;
import com.google.devtools.build.lib.bazel.repository.JarDecompressor;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests expanding external jars into external repositories.
 */
@RunWith(JUnit4.class)
public class JarDecompressorTest {
  private static final String FOO_JAR = "/foo.jar";
  private static final String BUILD_FILE = "BUILD.bazel";
  private static final String FOO_SOURCES_JAR = "/foo-sources.jar";
  private DecompressorDescriptor.Builder descriptorBuilder;
  private Scratch fs;

  @Before
  public void setUpFs() throws Exception {
    fs = new Scratch();
    Path dir = fs.dir("/whatever/external/tester");
    Path jar = fs.file(FOO_JAR, "I'm a jar");
    FileSystemUtils.createDirectoryAndParents(dir);
    descriptorBuilder = DecompressorDescriptor.builder()
        .setDecompressor(JarDecompressor.INSTANCE)
        .setTargetName("tester")
        .setTargetKind("http_jar rule")
        .setRepositoryPath(dir)
        .setArchivePath(jar);
  }

  @Test
  public void testTargets() throws Exception {
    Path outputDir = DecompressorValue.decompress(descriptorBuilder.build());
    assertThat(outputDir.exists()).isTrue();
    String buildContent =
        new String(FileSystemUtils.readContentAsLatin1(jarDirectoryFile(BUILD_FILE, outputDir)));
    assertThat(jarDirectoryFile(FOO_JAR, outputDir).exists()).isTrue();
    assertThat(jarDirectoryFile(FOO_SOURCES_JAR, outputDir).exists()).isFalse();
    assertThat(buildContent).contains("java_import");
    assertThat(buildContent).contains("filegroup");
  }

  @Test
  public void testWorkspaceGen() throws Exception {
    Path outputDir = DecompressorValue.decompress(descriptorBuilder.build());
    assertThat(outputDir.exists()).isTrue();
    String workspaceContent = new String(
        FileSystemUtils.readContentAsLatin1(outputDir.getRelative("WORKSPACE")));
    assertThat(workspaceContent).contains("workspace(name = \"tester\")");
  }

  @Test
  public void testTargetsWithSources() throws Exception {
    Path sourceJar = fs.file(FOO_SOURCES_JAR, "I'm a sources jar");
    descriptorBuilder.setArchiveSourcesPath(sourceJar);

    Path outputDir = DecompressorValue.decompress(descriptorBuilder.build());
    assertThat(outputDir.exists()).isTrue();
    assertThat(jarDirectoryFile(sourceJar.getBaseName(), outputDir).exists()).isTrue();
    assertThat(jarDirectoryFile(FOO_JAR, outputDir).exists()).isTrue();

    String buildContent =
        new String(FileSystemUtils.readContentAsLatin1(jarDirectoryFile(BUILD_FILE, outputDir)));
    assertThat(buildContent).contains("java_import");
    assertThat(buildContent).contains("srcjar");
    assertThat(buildContent).contains("filegroup");
  }

  private Path jarDirectoryFile(String baseName, Path outputPath) {
    return outputPath.getRelative("jar/" + baseName);
  }
}
