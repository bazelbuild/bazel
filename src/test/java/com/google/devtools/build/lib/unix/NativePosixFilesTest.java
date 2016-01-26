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
package com.google.devtools.build.lib.unix;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableMap;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.UnixFileSystem;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.File;
import java.io.FileNotFoundException;

/**
 * This class tests the FilesystemUtils class.
 */
@RunWith(JUnit4.class)
public class NativePosixFilesTest {
  private FileSystem testFS;
  private Path workingDir;
  private Path testFile;

  @Before
  public final void createFileSystem() throws Exception  {
    testFS = new UnixFileSystem();
    workingDir = testFS.getPath(new File(TestUtils.tmpDir()).getCanonicalPath());
    testFile = workingDir.getRelative("test");
  }

  /**
   * This test validates that the md5sum() method returns hashes that match the official test
   * vectors specified in RFC 1321, The MD5 Message-Digest Algorithm.
   *
   * @throws Exception
   */
  @Test
  public void testValidateMd5Sum() throws Exception {
    ImmutableMap<String, String> testVectors = ImmutableMap.<String, String>builder()
        .put("", "d41d8cd98f00b204e9800998ecf8427e")
        .put("a", "0cc175b9c0f1b6a831c399e269772661")
        .put("abc", "900150983cd24fb0d6963f7d28e17f72")
        .put("message digest", "f96b697d7cb7938d525a2f31aaf161d0")
        .put("abcdefghijklmnopqrstuvwxyz", "c3fcd3d76192e4007dfb496cca67e13b")
        .put("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        "d174ab98d277d9f5a5611c2c9f419d9f")
        .put(
        "12345678901234567890123456789012345678901234567890123456789012345678901234567890",
        "57edf4a22be3c955ac49da2e2107b67a")
        .build();

    for (String testInput : testVectors.keySet()) {
      FileSystemUtils.writeContentAsLatin1(testFile, testInput);
      HashCode result = NativePosixFiles.md5sum(testFile.getPathString());
      assertThat(testVectors).containsEntry(testInput, result.toString());
    }
  }

  @Test
  public void throwsFileAccessException() throws Exception {
    FileSystemUtils.createEmptyFile(testFile);
    NativePosixFiles.chmod(testFile.getPathString(), 0200);

    try {
      NativePosixFiles.md5sum(testFile.getPathString());
      fail("Expected FileAccessException, but wasn't thrown.");
    } catch (FileAccessException e) {
      assertThat(e).hasMessage(testFile + " (Permission denied)");
    }
  }

  @Test
  public void throwsFileNotFoundException() throws Exception {
    try {
      NativePosixFiles.md5sum(testFile.getPathString());
      fail("Expected FileNotFoundException, but wasn't thrown.");
    } catch (FileNotFoundException e) {
      assertThat(e).hasMessage(testFile + " (No such file or directory)");
    }
  }

  @Test
  public void throwsFilePermissionException() throws Exception {
    File foo = new File("/bin");
    try {
      NativePosixFiles.setWritable(foo);
      fail("Expected FilePermissionException, but wasn't thrown.");
    } catch (FilePermissionException e) {
      assertThat(e).hasMessage(foo + " (Operation not permitted)");
    }
  }
}
