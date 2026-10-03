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

import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RepositoryUtils}. */
@RunWith(JUnit4.class)
public class RepositoryUtilsTest {
  // A byte that isn't valid UTF-8 in Bazel's internal string encoding.
  private static final String INVALID_UTF8 = "\u0080";

  private Path externalRoot;
  private Path repoDir;

  @Before
  public void setUp() throws Exception {
    var fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    externalRoot = fs.getPath("/output_base/external");
    repoDir = externalRoot.getChild("repo");
    repoDir.createDirectoryAndParents();
  }

  private RepositoryUtils.ReplantSymlinksResult replantSymlinks() throws Exception {
    return RepositoryUtils.replantSymlinks(
        repoDir,
        repoDir.getFileSystem().getPath("/workspace"),
        externalRoot,
        PathFragment.EMPTY_FRAGMENT,
        /* replantSymlinksIntoMainRepo= */ false);
  }

  @Test
  public void replantSymlinks_validNames_safeForRemoteCache() throws Exception {
    FileSystemUtils.writeContentAsLatin1(repoDir.getChild("f\u00c3\u00bcle.txt"), "");
    repoDir.getChild("link").createSymbolicLink(PathFragment.create("f\u00c3\u00bcle.txt"));

    assertThat(replantSymlinks().safeForRemoteCache()).isTrue();
  }

  @Test
  public void replantSymlinks_fileNameNotUtf8_notSafeForRemoteCache() throws Exception {
    repoDir.getChild("sub").createDirectory();
    FileSystemUtils.writeContentAsLatin1(repoDir.getRelative("sub/" + INVALID_UTF8 + ".txt"), "");

    var result = replantSymlinks();
    assertThat(result.safeForRemoteCache()).isFalse();
    assertThat(result.safeForLocalCache()).isTrue();
  }

  @Test
  public void replantSymlinks_symlinkTargetNotUtf8_notSafeForRemoteCache() throws Exception {
    FileSystemUtils.writeContentAsLatin1(repoDir.getChild("file.txt"), "");
    repoDir.getChild("link").createSymbolicLink(PathFragment.create(INVALID_UTF8 + ".txt"));

    var result = replantSymlinks();
    assertThat(result.safeForRemoteCache()).isFalse();
    assertThat(result.safeForLocalCache()).isTrue();
  }
}
