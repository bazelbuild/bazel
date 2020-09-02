// Copyright 2018 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link FakeSandboxfsProcess}. */
@RunWith(JUnit4.class)
public class FakeSandboxfsProcessIntegrationTest extends BaseSandboxfsProcessIntegrationTest {

  @Override
  Path newTmpDir() throws IOException {
    FileSystem fileSystem = new InMemoryFileSystem();
    Path tmpDir = fileSystem.getPath("/tmp");
    tmpDir.createDirectory();
    return tmpDir;
  }

  @Override
  SandboxfsProcess mount(Path mountPoint) throws IOException {
    return new FakeSandboxfsProcess(mountPoint.getFileSystem(), mountPoint.asFragment());
  }

  @Test
  public void testMount_notADirectory() throws IOException {
    FileSystemUtils.createEmptyFile(tmpDir.getRelative("file"));
    IOException expected = assertThrows(
        IOException.class, () -> mount(tmpDir.getRelative("file")));
    assertThat(expected).hasMessageThat().matches(".*/file.*not a directory");
  }

  @Test
  public void testSandboxCreate_enforcesNonRepeatedNames() throws IOException {
    Path mountPoint = tmpDir.getRelative("mnt");
    mountPoint.createDirectory();
    SandboxfsProcess process = mount(mountPoint);

    process.createSandbox("first", (mapper) -> {});
    process.createSandbox("second", (mapper) -> {});
    assertThrows(
        IllegalArgumentException.class, () -> process.createSandbox("first", (mapper) -> {}));
  }

  @Test
  public void testSandboxDestroy_enforcesExistingName() throws IOException {
    Path mountPoint = tmpDir.getRelative("mnt");
    mountPoint.createDirectory();
    SandboxfsProcess process = mount(mountPoint);

    process.createSandbox("first", (mapper) -> {});
    process.destroySandbox("first");
    assertThrows(IllegalArgumentException.class, () -> process.destroySandbox("first"));
  }

  @Test
  public void testCreateAndDestroySandbox_reuseName() throws IOException {
    Path mountPoint = tmpDir.getRelative("mnt");
    mountPoint.createDirectory();
    SandboxfsProcess process = mount(mountPoint);

    process.createSandbox("first", (mapper) -> {});
    process.destroySandbox("first");
    process.createSandbox("first", (mapper) -> {});
    process.destroySandbox("first");
  }
}
