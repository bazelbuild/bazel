// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.exec;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.DigestOfDirectoryException;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests SingleBuildFileCache. */
@RunWith(JUnit4.class)
public class SingleBuildFileCacheTest {
  private FileSystem fs;
  private Map<String, Integer> calls;
  private Map<String, byte[]> digestOverrides;

  private SingleBuildFileCache underTest;

  @Before
  public final void setUp() throws Exception {
    calls = new HashMap<>();
    digestOverrides = new HashMap<>();
    fs =
        new InMemoryFileSystem(DigestHashFunction.SHA256) {
          @Override
          protected InputStream getInputStream(PathFragment path) throws IOException {
            int c = calls.containsKey(path.toString()) ? calls.get(path.toString()) : 0;
            c++;
            calls.put(path.toString(), c);
            return super.getInputStream(path);
          }

          @Override
          protected byte[] getDigest(PathFragment path) throws IOException {
            byte[] override = digestOverrides.get(path.getPathString());
            return override != null ? override : super.getDigest(path);
          }

          @Override
          protected byte[] getFastDigest(PathFragment path) throws IOException {
            return null;
          }
        };
    underTest = new SingleBuildFileCache("/", fs, SyscallCache.NO_CACHE);
    FileSystemUtils.createEmptyFile(fs.getPath("/empty"));
  }

  @Test
  public void testNonExistentPath() throws Exception {
    ActionInput empty = ActionInputHelper.fromPath("/noexist");
    assertThrows(
        "non existent file should raise exception",
        IOException.class,
        () -> underTest.getInputMetadata(empty));
  }

  @Test
  public void testDirectory() throws Exception {
    Path file = fs.getPath("/directory");
    file.createDirectory();
    ActionInput input = ActionInputHelper.fromPath(file.getPathString());
    DigestOfDirectoryException expected =
        assertThrows(
            "directory should raise exception",
            DigestOfDirectoryException.class,
            () -> underTest.getInputMetadata(input));
    assertThat(expected).hasMessageThat().isEqualTo("Input is a directory: /directory");
  }

  @Test
  public void testCache() throws Exception {
    ActionInput empty = ActionInputHelper.fromPath("/empty");
    underTest.getInputMetadata(empty).getDigest();
    assertThat(calls).containsKey("/empty");
    assertThat((int) calls.get("/empty")).isEqualTo(1);
    underTest.getInputMetadata(empty).getDigest();
    assertThat((int) calls.get("/empty")).isEqualTo(1);
  }

  @Test
  public void testBasic() throws Exception {
    ActionInput empty = ActionInputHelper.fromPath("/empty");
    assertThat(underTest.getInputMetadata(empty).getSize()).isEqualTo(0);
    byte[] digest = underTest.getInputMetadata(empty).getDigest();
    byte[] expected = fs.getDigestFunction().getHashFunction().hashBytes(new byte[0]).asBytes();
    assertThat(digest).isEqualTo(expected);
  }

  @Test
  public void testUnreadableFileWhenFileSystemSupportsDigest() throws Exception {
    byte[] expectedDigest = "expected".getBytes(StandardCharsets.UTF_8);
    digestOverrides.put("/unreadable", expectedDigest);

    ActionInput input = ActionInputHelper.fromPath("/unreadable");
    Path file = fs.getPath("/unreadable");
    FileSystemUtils.createEmptyFile(file);
    file.chmod(0);
    byte[] actualDigest = underTest.getInputMetadata(input).getDigest();
    assertThat(actualDigest).isEqualTo(expectedDigest);
  }
}
