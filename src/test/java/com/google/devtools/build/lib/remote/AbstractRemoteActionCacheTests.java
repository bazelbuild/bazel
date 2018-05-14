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
package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.remote.AbstractRemoteActionCache.UploadManifest;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystem.HashFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.remoteexecution.v1test.ActionResult;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link AbstractRemoteActionCache}. */
@RunWith(JUnit4.class)
public class AbstractRemoteActionCacheTests {

  private FileSystem fs;
  private Path execRoot;
  private final DigestUtil digestUtil = new DigestUtil(HashFunction.SHA256);

  @Before
  public void setUp() throws Exception {
    fs = new InMemoryFileSystem(new JavaClock(), HashFunction.SHA256);
    execRoot = fs.getPath("/execroot");
    execRoot.createDirectory();
  }

  @Test
  public void uploadSymlinkAsFile() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path link = fs.getPath("/execroot/link");
    Path target = fs.getPath("/execroot/target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    link.createSymbolicLink(target);

    UploadManifest um = new UploadManifest(digestUtil, result, execRoot, true);
    um.addFiles(ImmutableList.of(link));
    assertThat(um.getDigestToFile()).containsExactly(digestUtil.compute(target), link);

    assertThat(
            assertThrows(
                ExecException.class,
                () ->
                    new UploadManifest(digestUtil, result, execRoot, false)
                        .addFiles(ImmutableList.of(link))))
        .hasMessageThat()
        .contains("Only regular files and directories may be uploaded to a remote cache.");
  }

  @Test
  public void uploadSymlinkInDirectory() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = fs.getPath("/execroot/dir");
    dir.createDirectory();
    Path target = fs.getPath("/execroot/target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    Path link = fs.getPath("/execroot/dir/link");
    link.createSymbolicLink(fs.getPath("/execroot/target"));

    UploadManifest um = new UploadManifest(digestUtil, result, fs.getPath("/execroot"), true);
    um.addFiles(ImmutableList.of(link));
    assertThat(um.getDigestToFile()).containsExactly(digestUtil.compute(target), link);

    assertThat(
            assertThrows(
                ExecException.class,
                () ->
                    new UploadManifest(digestUtil, result, execRoot, false)
                        .addFiles(ImmutableList.of(link))))
        .hasMessageThat()
        .contains("Only regular files and directories may be uploaded to a remote cache.");
  }
}
