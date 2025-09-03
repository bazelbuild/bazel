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
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;
import static org.junit.Assume.assumeTrue;

import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the {@link NativePosixFiles} class. */
@RunWith(JUnit4.class)
public class NativePosixFilesTest {

  private Path workingDir;

  @Before
  public final void createFileSystem() throws Exception  {
    FileSystem testFS = new UnixFileSystem(DigestHashFunction.SHA256, /*hashAttributeName=*/ "");
    workingDir = testFS.getPath(new File(TestUtils.tmpDir()).getCanonicalPath());
  }

  @Test
  public void nativeExceptionContainsFileAndLine() throws Exception {
    File foo = new File("/non-existent");
    IOException e = assertThrows(IOException.class, () -> NativePosixFiles.readlink(foo.getPath()));
    assertThat(e).hasMessageThat().startsWith("[unix_jni.cc:");
    assertThat(e).hasMessageThat().endsWith("/non-existent (No such file or directory)");
  }

  /** Skips the test if the file system does not support extended attributes. */
  private static void assumeXattrsSupported() throws Exception {
    // The standard file systems on macOS support extended attributes by default, so we can assume
    // that the test will work on that platform. For other systems, we currently don't have a
    // mechanism to validate this so the tests are skipped unconditionally.
    assumeTrue(OS.getCurrent() == OS.DARWIN);
  }

  @Test
  public void testGetxattr_attributeFound() throws Exception {
    assumeXattrsSupported();

    String myfile = Files.createTempFile("getxattrtest", null).toString();
    assertThat(new ProcessBuilder("xattr", "-w", "foo", "bar", myfile).start().waitFor())
        .isEqualTo(0);

    assertThat(new String(NativePosixFiles.getxattr(myfile, "foo"), UTF_8)).isEqualTo("bar");
    assertThat(new String(NativePosixFiles.lgetxattr(myfile, "foo"), UTF_8)).isEqualTo("bar");
  }

  @Test
  public void testGetxattr_attributeNotFoundReturnsNull() throws Exception {
    assumeXattrsSupported();

    String myfile = Files.createTempFile("getxattrtest", null).toString();

    assertThat(NativePosixFiles.getxattr(myfile, "foo")).isNull();
    assertThat(NativePosixFiles.lgetxattr(myfile, "foo")).isNull();
  }

  @Test
  public void testGetxattr_fileNotFound() throws Exception {
    String nonexistentFile = workingDir.getChild("nonexistent").toString();

    assertThrows(
        FileNotFoundException.class, () -> NativePosixFiles.getxattr(nonexistentFile, "foo"));
    assertThrows(
        FileNotFoundException.class, () -> NativePosixFiles.lgetxattr(nonexistentFile, "foo"));
  }
}
