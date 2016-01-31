// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.unix.NativePosixFiles;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;

/**
 * Tests for the {@link UnixFileSystem} class.
 */
@RunWith(JUnit4.class)
public class UnixFileSystemTest extends SymlinkAwareFileSystemTest {

  @Override
  protected FileSystem getFreshFileSystem() {
    return new UnixFileSystem();
  }

  @Override
  public void destroyFileSystem(FileSystem fileSystem) {
    // Nothing.
  }

  @Override
  protected void expectNotFound(Path path) throws IOException {
    assertNull(path.statIfFound());
  }

  // Most tests are just inherited from FileSystemTest.

  @Test
  public void testCircularSymlinkFound() throws Exception {
    Path linkA = absolutize("link-a");
    Path linkB = absolutize("link-b");
    linkA.createSymbolicLink(linkB);
    linkB.createSymbolicLink(linkA);
    assertFalse(linkA.exists(Symlinks.FOLLOW));
    try {
      linkA.statIfFound(Symlinks.FOLLOW);
      fail();
    } catch (IOException expected) {
      // Expected.
    }
  }

  @Test
  public void testIsSpecialFile() throws Exception {
    Path regular = absolutize("regular");
    Path fifo = absolutize("fifo");
    FileSystemUtils.createEmptyFile(regular);
    NativePosixFiles.mkfifo(fifo.toString(), 0777);
    assertTrue(regular.isFile());
    assertFalse(regular.isSpecialFile());
    assertTrue(regular.stat().isFile());
    assertFalse(regular.stat().isSpecialFile());
    assertTrue(fifo.isFile());
    assertTrue(fifo.isSpecialFile());
    assertTrue(fifo.stat().isFile());
    assertTrue(fifo.stat().isSpecialFile());
  }
}
