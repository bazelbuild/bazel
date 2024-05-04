// Copyright 2022 The Bazel Authors. All rights reserved.
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
import static com.google.common.truth.TruthJUnit.assume;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.Before;
import org.junit.Test;

/** Tests for linux-sandbox overlayfs functionality. */
public final class LinuxSandboxOverlayfsTest extends BuildIntegrationTestCase {

  @Before
  public void assumeRunningOnLinux() {
    assume().that(OS.getCurrent()).isEqualTo(OS.LINUX);
  }

  @Test
  public void nestedOverlayFs() throws Exception {
    var tmpMountNio =
        Files.createTempDirectory(java.nio.file.Path.of("/tmp"), "bazel-sandbox-test");
    tmpMountNio.toFile().deleteOnExit();
    var tmpMount = fileSystem.getPath(tmpMountNio.toAbsolutePath().toString());

    Path binDir = testRoot.getRelative("_bin");
    binDir.createDirectoryAndParents();
    Path linuxSandboxPath = SpawnRunnerTestUtil.copyLinuxSandboxIntoPath(binDir);

    // This path is writable with the sandboxed runner.
    Path writableRoot = testRoot.getRelative("root");
    writableRoot.createDirectoryAndParents();

    Path hermeticTmpOuter = writableRoot.getRelative("_hermetic_tmp_outer");
    Path upperDirOuter = hermeticTmpOuter.getRelative("_upper");
    upperDirOuter.createDirectoryAndParents();
    Path workDirOuter = hermeticTmpOuter.getRelative("_work");
    workDirOuter.createDirectoryAndParents();
    Path mountDirOuter = hermeticTmpOuter.getRelative("_mount");
    mountDirOuter.createDirectoryAndParents();
    FileSystemUtils.writeContent(mountDirOuter.getChild("file"), UTF_8, "outer_content");

    Path hermeticTmpInner = writableRoot.getRelative("_hermetic_tmp_inner");
    Path upperDirInner = hermeticTmpInner.getRelative("_upper");
    upperDirInner.createDirectoryAndParents();
    Path workDirInner = hermeticTmpInner.getRelative("_work");
    workDirInner.createDirectoryAndParents();
    Path mountDirInner = hermeticTmpInner.getRelative("_mount");
    mountDirInner.createDirectoryAndParents();
    FileSystemUtils.writeContent(mountDirInner.getChild("file"), UTF_8, "inner_content");

    var argsInner =
        LinuxSandboxCommandLineBuilder.commandLineBuilder(linuxSandboxPath)
            .mountOverlayfsOnTmp(upperDirInner.getPathString(), workDirInner.getPathString())
            .setWritableFilesAndDirectories(Set.of(writableRoot))
            .setBindMounts(Map.of(tmpMount, mountDirInner))
            .buildForCommand(List.of("cat", tmpMount.getChild("file").getPathString()));

    var argsOuter =
        LinuxSandboxCommandLineBuilder.commandLineBuilder(linuxSandboxPath)
            .mountOverlayfsOnTmp(upperDirOuter.getPathString(), workDirOuter.getPathString())
            .setWritableFilesAndDirectories(Set.of(writableRoot))
            .setBindMounts(Map.of(tmpMount, mountDirOuter))
            .buildForCommand(argsInner);

    Process process = new ProcessBuilder().command(argsOuter).start();
    byte[] stderr = process.getErrorStream().readAllBytes();
    assertThat(new String(stderr, StandardCharsets.UTF_8)).isEmpty();
    byte[] stdout = process.getInputStream().readAllBytes();
    assertThat(new String(stdout, StandardCharsets.UTF_8)).isEqualTo("inner_content");
    int exitCode = process.waitFor();
    assertThat(exitCode).isEqualTo(0);
  }
}
