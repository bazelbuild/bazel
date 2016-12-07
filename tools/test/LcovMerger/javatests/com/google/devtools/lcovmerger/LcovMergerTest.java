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

package com.google.devtools.lcovmerger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.FileSystems;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test for LcovMerger.
 */
@RunWith(JUnit4.class)
public class LcovMergerTest {

  private final Scratch scratch =
      new Scratch(FileSystemUtils.getWorkingDirectory(FileSystems.initDefaultAsJavaIo()));

  @Test
  public void testZeroDatFile() throws IOException {
    scratch.dir("dir0");

    File merged = new File("dir0Merged.dat");
    File dir = new File("dir0");
    LcovMerger merger = new LcovMerger(dir.getAbsolutePath(), merged.getAbsolutePath());
    boolean success = merger.merge();
    assertFalse(success);

    assertFalse(merged.exists());
  }

  @Test
  public void testOneDatFile() throws IOException {
    String content = "This is an lcov file.";
    scratch.file("dir1/jvcov.dat", content);

    File merged = new File("dir1Merged.dat");
    File dir = new File("dir1");
    LcovMerger merger = new LcovMerger(dir.getAbsolutePath(), merged.getAbsolutePath());
    boolean success = merger.merge();
    assertTrue(success);

    assertTrue(merged.exists());
    String readContent = new String(Files.readAllBytes(merged.toPath())).trim();
    assertEquals(content, readContent);
  }

  @Test
  public void testTwoDatFiles() throws IOException {
    scratch.file("dir2/jvcov1.dat", "This is an lcov file.");
    scratch.file("dir2/jvcov2.dat", "This is another lcov file.");

    File merged = new File("dir2Merged.dat");
    File dir = new File("dir2");
    LcovMerger merger = new LcovMerger(dir.getAbsolutePath(), merged.getAbsolutePath());
    boolean success = merger.merge();
    assertFalse(success);

    assertFalse(merged.exists());
  }
}
