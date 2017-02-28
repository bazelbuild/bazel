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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.Lists;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Path} in combination with the native file system for the current platform. */
@RunWith(JUnit4.class)
public class NativePathTest {

  private FileSystem fs;
  private File aDirectory;
  private File aFile;
  private File anotherFile;
  private File tmpDir;

  protected FileSystem getNativeFileSystem() {
    return FileSystems.getNativeFileSystem();
  }

  @Before
  public final void createFiles() throws Exception  {
    fs = getNativeFileSystem();
    tmpDir = new File(TestUtils.tmpDir(), "tmpDir");
    tmpDir.mkdirs();
    aDirectory = new File(tmpDir, "a_directory");
    aDirectory.mkdirs();
    aFile = new File(tmpDir, "a_file");
    new FileOutputStream(aFile).close();
    anotherFile = new File(aDirectory, "another_file.txt");
    new FileOutputStream(anotherFile).close();
  }

  @Test
  public void testExists() {
    assertTrue(fs.getPath(aDirectory.getPath()).exists());
    assertTrue(fs.getPath(aFile.getPath()).exists());
    assertFalse(fs.getPath("/does/not/exist").exists());
  }

  @Test
  public void testDirectoryEntriesForDirectory() throws IOException {
    Collection<Path> entries =
        fs.getPath(tmpDir.getPath()).getDirectoryEntries();
    List<Path> expectedEntries = Arrays.asList(
      fs.getPath(tmpDir.getPath() + "/a_file"),
      fs.getPath(tmpDir.getPath() + "/a_directory"));

    assertEquals(new HashSet<Object>(expectedEntries),
        new HashSet<Object>(entries));
  }

  @Test
  public void testDirectoryEntriesForFileThrowsException() {
    try {
      fs.getPath(aFile.getPath()).getDirectoryEntries();
      fail("No exception thrown.");
    } catch (IOException x) {
      // The expected result.
    }
  }

  @Test
  public void testIsFileIsTrueForFile() {
    assertTrue(fs.getPath(aFile.getPath()).isFile());
  }

  @Test
  public void testIsFileIsFalseForDirectory() {
    assertFalse(fs.getPath(aDirectory.getPath()).isFile());
  }

  @Test
  public void testBaseName() {
    assertEquals("base", fs.getPath("/foo/base").getBaseName());
  }

  @Test
  public void testBaseNameRunsAfterDotDotInterpretation() {
    assertEquals("base", fs.getPath("/base/foo/..").getBaseName());
  }

  @Test
  public void testParentOfRootIsRoot() {
    assertEquals(fs.getPath("/"), fs.getPath("/.."));
    assertEquals(fs.getPath("/"), fs.getPath("/../../../../../.."));
    assertEquals(fs.getPath("/foo"), fs.getPath("/../../../foo"));
  }

  @Test
  public void testIsDirectory() {
    assertTrue(fs.getPath(aDirectory.getPath()).isDirectory());
    assertFalse(fs.getPath(aFile.getPath()).isDirectory());
    assertFalse(fs.getPath("/does/not/exist").isDirectory());
  }

  @Test
  public void testListNonExistingDirectoryThrowsException() {
    try {
      fs.getPath("/does/not/exist").getDirectoryEntries();
      fail("No exception thrown.");
    } catch (IOException ex) {
      // success!
    }
  }

  private void assertPathSet(Collection<Path> actual, String... expected) {
    List<String> actualStrings = Lists.newArrayListWithCapacity(actual.size());

    for (Path path : actual) {
      actualStrings.add(path.getPathString());
    }

    assertThat(actualStrings).containsExactlyElementsIn(Arrays.asList(expected));
  }

  @Test
  public void testGlob() throws Exception {
    Collection<Path> textFiles = UnixGlob.forPath(fs.getPath(tmpDir.getPath()))
        .addPattern("*/*.txt")
        .globInterruptible();
    assertThat(textFiles).hasSize(1);
    Path onlyFile = textFiles.iterator().next();
    assertEquals(fs.getPath(anotherFile.getPath()), onlyFile);

    Collection<Path> onlyFiles =
        UnixGlob.forPath(fs.getPath(tmpDir.getPath()))
        .addPattern("*")
        .setExcludeDirectories(true)
        .globInterruptible();
    assertPathSet(onlyFiles, aFile.getPath());

    Collection<Path> directoriesToo =
        UnixGlob.forPath(fs.getPath(tmpDir.getPath()))
        .addPattern("*")
        .setExcludeDirectories(false)
        .globInterruptible();
    assertPathSet(directoriesToo, aFile.getPath(), aDirectory.getPath());
  }

  @Test
  public void testGetRelative() {
    Path relative = fs.getPath("/foo").getChild("bar");
    Path expected = fs.getPath("/foo/bar");
    assertEquals(expected, relative);
  }

  @Test
  public void testEqualsAndHash() {
    Path path = fs.getPath("/foo/bar");
    Path equalPath = fs.getPath("/foo/bar");
    Path differentPath = fs.getPath("/foo/bar/baz");
    Object differentType = new Object();

    new EqualsTester().addEqualityGroup(path, equalPath).testEquals();
    assertFalse(path.equals(differentPath));
    assertFalse(path.equals(differentType));
  }

  @Test
  public void testLatin1ReadAndWrite() throws IOException {
    char[] allLatin1Chars = new char[256];
    for (int i = 0; i < 256; i++) {
      allLatin1Chars[i] = (char) i;
    }
    Path path = fs.getPath(aFile.getPath());
    String latin1String = new String(allLatin1Chars);
    FileSystemUtils.writeContentAsLatin1(path, latin1String);
    String fileContent = new String(FileSystemUtils.readContentAsLatin1(path));
    assertEquals(fileContent, latin1String);
  }

  /**
   * Verify that the encoding implemented by {@link
   * com.google.devtools.build.lib.vfs.FileSystemUtils#writeContentAsLatin1(Path, String)} really is
   * 8859-1 (latin1).
   */
  @Test
  public void testVerifyLatin1() throws IOException {
    char[] allLatin1Chars = new char[256];
    for( int i = 0; i < 256; i++) {
      allLatin1Chars[i] = (char)i;
    }
    Path path = fs.getPath(aFile.getPath());
    String latin1String = new String(allLatin1Chars);
    FileSystemUtils.writeContentAsLatin1(path, latin1String);
    byte[] bytes = FileSystemUtils.readContent(path);
    assertEquals(new String(bytes, "ISO-8859-1"), latin1String);
  }

  @Test
  public void testBytesReadAndWrite() throws IOException {
    byte[] bytes = new byte[] { (byte) 0xdeadbeef, (byte) 0xdeadbeef>>8,
                                (byte) 0xdeadbeef>>16, (byte) 0xdeadbeef>>24 };
    Path path = fs.getPath(aFile.getPath());
    FileSystemUtils.writeContent(path, bytes);
    byte[] content = FileSystemUtils.readContent(path);
    assertEquals(bytes.length, content.length);
    for (int i = 0; i < bytes.length; i++) {
      assertEquals(bytes[i], content[i]);
    }
  }

  @Test
  public void testInputOutputStreams() throws IOException {
    Path path = fs.getPath(aFile.getPath());
    OutputStream out = path.getOutputStream();
    for (int i = 0; i < 256; i++) {
      out.write(i);
    }
    out.close();
    InputStream in = path.getInputStream();
    for (int i = 0; i < 256; i++) {
      assertEquals(i, in.read());
    }
    assertEquals(-1, in.read());
    in.close();
  }

  @Test
  public void testDerivedSegmentEquality() {
    Path absoluteSegment = fs.getRootDirectory();

    Path derivedNode = absoluteSegment.getChild("derivedSegment");
    Path otherDerivedNode = absoluteSegment.getChild("derivedSegment");

    assertSame(derivedNode, otherDerivedNode);
  }
}
