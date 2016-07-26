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

import static org.junit.Assert.assertEquals;

import com.google.devtools.build.lib.testutil.ManualClock;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for the {@link JavaIoFileSystem}. That file system by itself is not
 * capable of creating symlinks; use the unix one to create them, so that the
 * test can check that the file system handles their existence correctly.
 */
@RunWith(JUnit4.class)
public class JavaIoFileSystemTest extends SymlinkAwareFileSystemTest {

  private ManualClock clock;

  @Override
  public FileSystem getFreshFileSystem() {
    clock = new ManualClock();
    return new JavaIoFileSystem(clock);
  }

  // Tests are inherited from the FileSystemTest

  // JavaIoFileSystem incorrectly throws a FileNotFoundException for all IO errors. This means that
  // statIfFound incorrectly suppresses those errors.
  @Override
  @Test
  public void testBadPermissionsThrowsExceptionOnStatIfFound() {}

  @Test
  public void testSetLastModifiedTime() throws Exception {
    Path file = xEmptyDirectory.getChild("new-file");
    FileSystemUtils.createEmptyFile(file);

    file.setLastModifiedTime(1000L);
    assertEquals(1000L, file.getLastModifiedTime());
    file.setLastModifiedTime(0L);
    assertEquals(0L, file.getLastModifiedTime());

    clock.advanceMillis(42000L);
    file.setLastModifiedTime(-1L);
    assertEquals(42000L, file.getLastModifiedTime());
  }
}
