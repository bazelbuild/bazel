// Copyright 2026 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SandboxModule}. */
@RunWith(JUnit4.class)
public final class SandboxModuleTest {

  private Scratch scratch;
  private Path sandboxBase;

  @Before
  public void setUp() throws Exception {
    scratch = new Scratch();
    sandboxBase = scratch.dir("/sandbox_base");
  }

  @Test
  public void checkSandboxBaseTopOnlyContainsPersistentDirs_persistentDirsSuccess()
      throws Exception {
    scratch.dir("/sandbox_base/_moved_trash_dir");
    scratch.dir("/sandbox_base/sandbox_stash");
    scratch.file("/sandbox_base/.DS_Store");

    SandboxModule.checkSandboxBaseTopOnlyContainsPersistentDirs(sandboxBase);
  }

  @Test
  public void checkSandboxBaseTopOnlyContainsPersistentDirs_unexpectedDirThrows() throws Exception {
    scratch.dir("/sandbox_base/_moved_trash_dir");
    scratch.dir("/sandbox_base/sandbox_stash");
    scratch.dir("/sandbox_base/linux-sandbox");

    IllegalStateException e =
        assertThrows(
            IllegalStateException.class,
            () -> SandboxModule.checkSandboxBaseTopOnlyContainsPersistentDirs(sandboxBase));

    assertThat(e).hasMessageThat().contains("linux-sandbox");
    assertThat(e).hasMessageThat().doesNotContain("_moved_trash_dir");
    assertThat(e).hasMessageThat().doesNotContain("sandbox_stash");
  }

  @Test
  public void cleanSandboxBaseTopOnlyContainsPersistentDirs_cleansUnexpectedDirsAndFiles()
      throws Exception {
    scratch.dir("/sandbox_base/_moved_trash_dir");
    scratch.dir("/sandbox_base/sandbox_stash");
    scratch.dir("/sandbox_base/linux-sandbox");
    scratch.file("/sandbox_base/stale_file.txt", "stale content");

    SandboxModule.cleanSandboxBaseTopOnlyContainsPersistentDirs(
        sandboxBase, new SynchronousTreeDeleter());

    assertThat(sandboxBase.getChild("_moved_trash_dir").exists()).isTrue();
    assertThat(sandboxBase.getChild("sandbox_stash").exists()).isTrue();
    assertThat(sandboxBase.getChild("linux-sandbox").exists()).isFalse();
    assertThat(sandboxBase.getChild("stale_file.txt").exists()).isFalse();

    SandboxModule.checkSandboxBaseTopOnlyContainsPersistentDirs(sandboxBase);
  }
}
