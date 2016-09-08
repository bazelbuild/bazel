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
package com.google.devtools.build.lib.sandbox;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.nio.charset.Charset;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@code LinuxSandboxedStrategy}.
 */
@RunWith(JUnit4.class)
public class LinuxSandboxedStrategyTest extends LinuxSandboxedStrategyTestCase {
  @Test
  public void testParseManifestFile() throws Exception {
    PathFragment targetDir = new PathFragment("runfiles");

    Path testFile = workspaceDir.getRelative("testfile");
    FileSystemUtils.createEmptyFile(testFile);

    Path manifestFile = workspaceDir.getRelative("MANIFEST");
    FileSystemUtils.writeContent(
        manifestFile,
        Charset.defaultCharset(),
        String.format("x/testfile %s\nx/emptyfile \n", testFile.getPathString()));

    Map<PathFragment, Path> mounts = new TreeMap<>();
    SpawnHelpers.parseManifestFile(
        fileSystem, mounts, targetDir, manifestFile.getPathFile(), false, "");

    assertThat(mounts)
        .isEqualTo(
            ImmutableMap.of(
                new PathFragment("runfiles/x/testfile"),
                testFile,
                new PathFragment("runfiles/x/emptyfile"),
                fileSystem.getPath("/dev/null")));
  }

  @Test
  public void testParseFilesetManifestFile() throws Exception {
    PathFragment targetDir = new PathFragment("fileset");

    Path testFile = workspaceDir.getRelative("testfile");
    FileSystemUtils.createEmptyFile(testFile);

    Path manifestFile = workspaceDir.getRelative("MANIFEST");
    FileSystemUtils.writeContent(
        manifestFile,
        Charset.defaultCharset(),
        String.format("workspace/x/testfile %s\n0\n", testFile.getPathString()));

    Map<PathFragment, Path> mounts = new HashMap<>();
    SpawnHelpers.parseManifestFile(
        fileSystem, mounts, targetDir, manifestFile.getPathFile(), true, "workspace");

    assertThat(mounts).isEqualTo(ImmutableMap.of(new PathFragment("fileset/x/testfile"), testFile));
  }
}
