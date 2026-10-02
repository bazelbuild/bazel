// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RepositoryUtils}. */
@RunWith(JUnit4.class)
public final class RepositoryUtilsTest {

  private static RepositoryUtils.ReplantSymlinksResult replantSymlinks(
      FileSystem fs, String symlinkTarget) throws Exception {
    Path workspace = fs.getPath("/workspace");
    workspace.createDirectoryAndParents();
    Path externalRoot = fs.getPath("/output_base/external");
    Path repoDir = externalRoot.getChild("repo");
    repoDir.getChild("dir").createDirectoryAndParents();
    FileSystemUtils.writeContent(repoDir.getRelative("dir/file"), UTF_8, "contents");
    repoDir.getChild("link").createSymbolicLink(PathFragment.create(symlinkTarget));
    return RepositoryUtils.replantSymlinks(
        repoDir,
        workspace,
        externalRoot,
        PathFragment.EMPTY_FRAGMENT,
        /* replantSymlinksIntoMainRepo= */ false);
  }

  @Test
  public void replantSymlinks_symlinkToFile_safeForRemoteCache() throws Exception {
    var result = replantSymlinks(new InMemoryFileSystem(DigestHashFunction.SHA256), "dir/file");

    assertThat(result.safeForRemoteCache()).isTrue();
  }

  @Test
  public void replantSymlinks_fileSymlinksUnsupported_symlinkToFile_notSafeForRemoteCache()
      throws Exception {
    var result = replantSymlinks(new FileSystemWithoutFileSymlinks(), "dir/file");

    assertThat(result.safeForRemoteCache()).isFalse();
  }

  @Test
  public void replantSymlinks_fileSymlinksUnsupported_symlinkToDirectory_safeForRemoteCache()
      throws Exception {
    var result = replantSymlinks(new FileSystemWithoutFileSymlinks(), "dir");

    assertThat(result.safeForRemoteCache()).isTrue();
  }

  @Test
  public void replantSymlinks_fileSymlinksUnsupported_danglingSymlink_notSafeForRemoteCache()
      throws Exception {
    var result = replantSymlinks(new FileSystemWithoutFileSymlinks(), "missing");

    assertThat(result.safeForRemoteCache()).isFalse();
  }

  @Test
  public void replantSymlinks_fileSymlinksUnsupported_symlinkLoop_notSafeForRemoteCache()
      throws Exception {
    var result = replantSymlinks(new FileSystemWithoutFileSymlinks(), "link");

    assertThat(result.safeForLocalCache()).isTrue();
    assertThat(result.safeForRemoteCache()).isFalse();
  }

  @Test
  public void replantSymlinks_repoRootIsSymlink_notSafeForRemoteCache() throws Exception {
    var fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    Path workspace = fs.getPath("/workspace");
    workspace.getChild("dir").createDirectoryAndParents();
    FileSystemUtils.writeContent(workspace.getRelative("dir/file"), UTF_8, "contents");
    Path externalRoot = fs.getPath("/output_base/external");
    externalRoot.createDirectoryAndParents();
    Path repoDir = externalRoot.getChild("repo");
    repoDir.createSymbolicLink(workspace.getChild("dir"));

    var result =
        RepositoryUtils.replantSymlinks(
            repoDir,
            workspace,
            externalRoot,
            PathFragment.EMPTY_FRAGMENT,
            /* replantSymlinksIntoMainRepo= */ false);

    assertThat(result.safeForRemoteCache()).isFalse();
  }

  /**
   * A file system that reports that it can't create symlinks to files, like the Windows file system
   * does by default.
   */
  private static final class FileSystemWithoutFileSymlinks extends InMemoryFileSystem {
    FileSystemWithoutFileSymlinks() {
      super(DigestHashFunction.SHA256);
    }

    @Override
    public boolean supportsSymbolicLinksNatively(PathFragment path) {
      return false;
    }
  }
}
