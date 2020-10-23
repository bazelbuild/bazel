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

package com.google.devtools.build.lib.server;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link ShutdownHooks}. */
@RunWith(JUnit4.class)
public class ShutdownHooksTest {

  private FileSystem fileSystem;

  @Before
  public void setUp() {
    fileSystem = new InMemoryFileSystem(DigestHashFunction.SHA256);
  }

  @Test
  public void testDeletesRegisteredPaths() throws IOException {
    Path toDelete = fileSystem.getPath("/some-path-to-delete");
    toDelete.createDirectoryAndParents();

    Path toKeep = fileSystem.getPath("/some-path-to-keep");
    toKeep.createDirectoryAndParents();

    ShutdownHooks underTest = ShutdownHooks.createUnregistered();
    underTest.deleteAtExit(toDelete);
    underTest.runHooks();

    assertThat(toDelete.exists()).isFalse();
    assertThat(toKeep.exists()).isTrue();
  }

  @Test
  public void testSkipHooksIfDisabled() throws IOException {
    Path toDelete = fileSystem.getPath("/some-path-to-delete");
    toDelete.createDirectoryAndParents();

    ShutdownHooks underTest = ShutdownHooks.createUnregistered();
    underTest.deleteAtExit(toDelete);
    underTest.disable();
    underTest.runHooks();

    assertThat(toDelete.exists()).isTrue();
  }
}
