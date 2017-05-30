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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.Lists;
import com.google.common.io.CharStreams;
import com.google.devtools.build.lib.testutil.BlazeTestUtils;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class ZipFileSystemTest {

  /**
   * Expected listing of sample zip files, in alpha sorted order
   */
  private static final String[] LISTING = {
    "/dir1",
    "/dir1/file1a",
    "/dir1/file1b",
    "/dir2",
    "/dir2/dir3",
    "/dir2/dir3/dir4",
    "/dir2/dir3/dir4/file4",
    "/dir2/file2",
    "/file0",
  };

  private FileSystem zipFS1;
  private FileSystem zipFS2;

  @Before
  public final void initializeFileSystems() throws Exception  {
    FileSystem unixFs = FileSystems.getNativeFileSystem();
    Path testdataDir = unixFs.getPath(BlazeTestUtils.runfilesDir()).getRelative(
        TestConstants.JAVATESTS_ROOT + "/com/google/devtools/build/lib/vfs");
    Path zPath1 = testdataDir.getChild("sample_with_dirs.zip");
    Path zPath2 = testdataDir.getChild("sample_without_dirs.zip");
    zipFS1 = new ZipFileSystem(zPath1);
    zipFS2 = new ZipFileSystem(zPath2);
  }

  private void checkExists(FileSystem fs) {
    assertThat(fs.getPath("/dir2/dir3/dir4").exists()).isTrue();
    assertThat(fs.getPath("/dir2/dir3/dir4/file4").exists()).isTrue();
    assertThat(fs.getPath("/dir2/dir3/dir4/bogus").exists()).isFalse();
  }

  @Test
  public void testExists() {
    checkExists(zipFS1);
    checkExists(zipFS2);
  }

  private void checkIsFile(FileSystem fs) {
    assertThat(fs.getPath("/dir2/dir3/dir4").isFile()).isFalse();
    assertThat(fs.getPath("/dir2/dir3/dir4/file4").isFile()).isTrue();
    assertThat(fs.getPath("/dir2/dir3/dir4/bogus").isFile()).isFalse();
  }

  @Test
  public void testIsFile() {
    checkIsFile(zipFS1);
    checkIsFile(zipFS2);
  }

  private void checkIsDir(FileSystem fs) {
    assertThat(fs.getPath("/dir2/dir3/dir4").isDirectory()).isTrue();
    assertThat(fs.getPath("/dir2/dir3/dir4/file4").isDirectory()).isFalse();
    assertThat(fs.getPath("/bogus/mobogus").isDirectory()).isFalse();
    assertThat(fs.getPath("/bogus").isDirectory()).isFalse();
  }

  @Test
  public void testIsDir() {
    checkIsDir(zipFS1);
    checkIsDir(zipFS2);
  }

  /**
   * Recursively add the contents of a given path, rendered as strings, into a
   * given list.
   */
  private static void listChildren(Path p, List<String> list)
      throws IOException {
    for (Path c : p.getDirectoryEntries()) {
      list.add(c.getPathString());
      if (c.isDirectory()) {
        listChildren(c, list);
      }
    }
  }

  private void checkListing(FileSystem fs) throws Exception {
    List<String> list = new ArrayList<>();
    listChildren(fs.getRootDirectory(), list);
    Collections.sort(list);
    assertThat(list).isEqualTo(Lists.newArrayList(LISTING));
  }

  @Test
  public void testListing() throws Exception {
    checkListing(zipFS1);
    checkListing(zipFS2);

    // Regression test for: creation of a path (i.e. a file *name*)
    // must not affect the result of getDirectoryEntries().
    zipFS1.getPath("/dir1/notthere");
    checkListing(zipFS1);
  }

  private void checkFileSize(FileSystem fs, String name, long expectedSize)
      throws IOException {
    assertThat(fs.getPath(name).getFileSize()).isEqualTo(expectedSize);
  }

  @Test
  public void testCanReadRoot() {
    Path rootDirectory = zipFS1.getRootDirectory();
    assertThat(rootDirectory.isDirectory()).isTrue();
  }

  @Test
  public void testFileSize() throws IOException {
    checkFileSize(zipFS1, "/dir1/file1a", 5);
    checkFileSize(zipFS2, "/dir1/file1a", 5);
    checkFileSize(zipFS1, "/dir2/dir3/dir4/file4", 5000);
    checkFileSize(zipFS2, "/dir2/dir3/dir4/file4", 5000);
  }

  private void checkCantGetFileSize(FileSystem fs, String name) {
    try {
      fs.getPath(name).getFileSize();
      fail();
    } catch (IOException expected) {
      // expected
    }
  }

  @Test
  public void testCantGetFileSize() {
    checkCantGetFileSize(zipFS1, "/dir2/dir3/dir4/bogus");
    checkCantGetFileSize(zipFS2, "/dir2/dir3/dir4/bogus");
  }

  private void checkOpenFile(FileSystem fs, String name, int expectedSize)
      throws Exception {
    InputStream is = fs.getPath(name).getInputStream();
    List<String> lines = CharStreams.readLines(new InputStreamReader(is, "ISO-8859-1"));
    assertThat(lines).hasSize(expectedSize);
    for (int i = 0; i < expectedSize; i++) {
      assertThat(lines.get(i)).isEqualTo("body");
    }
  }

  @Test
  public void testOpenSmallFile() throws Exception {
    checkOpenFile(zipFS1, "/dir1/file1a", 1);
    checkOpenFile(zipFS2, "/dir1/file1a", 1);
  }

  @Test
  public void testOpenBigFile() throws Exception {
    checkOpenFile(zipFS1, "/dir2/dir3/dir4/file4", 1000);
    checkOpenFile(zipFS2, "/dir2/dir3/dir4/file4", 1000);
  }

  private void checkCantOpenFile(FileSystem fs, String name) {
    try {
      fs.getPath(name).getInputStream();
      fail();
    } catch (IOException expected) {
      // expected
    }
  }

  @Test
  public void testCantOpenFile() throws Exception {
    checkCantOpenFile(zipFS1, "/dir2/dir3/dir4/bogus");
    checkCantOpenFile(zipFS2, "/dir2/dir3/dir4/bogus");
  }

  private void checkCantCreateAnything(FileSystem fs, String name)  {
    Path p = fs.getPath(name);
    try {
      p.createDirectory();
      fail();
    } catch (Exception expected) {}
    try {
      FileSystemUtils.createEmptyFile(p);
      fail();
    } catch (Exception expected) {}
    try {
      p.createSymbolicLink(p);
      fail();
    } catch (Exception expected) {}
  }

  @Test
  public void testCantCreateAnything() throws Exception {
    checkCantCreateAnything(zipFS1, "/dir2/dir3/dir4/new");
    checkCantCreateAnything(zipFS2, "/dir2/dir3/dir4/new");
  }

}
