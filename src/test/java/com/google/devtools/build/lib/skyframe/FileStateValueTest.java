// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.when;

import com.google.devtools.build.lib.actions.FileContentsProxy;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Tests for {@link FileStateValue}. */
@RunWith(JUnit4.class)
public class FileStateValueTest {
  @Test
  public void testCodec() throws Exception {
    new SerializationTester(
            new FileStateValue.RegularFileStateValueWithDigest(
                /* size= */ 1, /* digest= */ new byte[] {1, 2, 3}, false),
            new FileStateValue.RegularFileStateValueWithDigest(
                /* size= */ 1, /* digest= */ new byte[0], true),
            new FileStateValue.RegularFileStateValueWithContentsProxy(
                /* size= */ 1, makeFileContentsProxy(/* ctime= */ 2, /* nodeId= */ 42), false),
            new FileStateValue.SpecialFileStateValue(
                makeFileContentsProxy(/* ctime= */ 4, /* nodeId= */ 84)),
            FileStateValue.DIRECTORY_FILE_STATE_NODE,
            new FileStateValue.SymlinkFileStateValue(PathFragment.create("somewhere/elses")),
            FileStateValue.NONEXISTENT_FILE_STATE_NODE)
        .runTests();
  }

  @Test
  public void executableChangeWithFastDigestIsDirty() throws Exception {
    var fs = new InMemoryFileSystem(DigestHashFunction.SHA256) {
      @Override
      public byte[] getFastDigest(PathFragment path) throws IOException {
        return getDigest(path);
      }
    };
    var file = fs.getPath("/script");
    FileSystemUtils.writeContentAsLatin1(file, "contents");
    file.setExecutable(false);
    var path = RootedPath.toRootedPath(Root.absoluteRoot(fs), file);
    var before = FileStateValue.create(path, SyscallCache.NO_CACHE, null);
    assertThat(before.getDigest()).isNotNull();
    file.setExecutable(true);
    var after = FileStateValue.create(path, SyscallCache.NO_CACHE, null);

    assertThat(after.getDigest()).isEqualTo(before.getDigest());
    assertThat(after).isNotEqualTo(before);
    assertThat(after.compareContents(before))
        .isEqualTo(FileStateValue.ContentsComparisonResult.DIFFERENT);
  }

  private static FileContentsProxy makeFileContentsProxy(long ctime, long nodeId)
      throws IOException {
    FileStatus status = Mockito.mock(FileStatus.class);
    when(status.getLastChangeTime()).thenReturn(ctime);
    when(status.getNodeId()).thenReturn(nodeId);
    return FileContentsProxy.create(status);
  }
}
