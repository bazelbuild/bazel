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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.SyscallCache.DirentTypeWithSkip;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.Collection;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DefaultSyscallCache}. */
@RunWith(JUnit4.class)
public final class DefaultSyscallCacheTest {

  private FileSystem fs;
  private DefaultSyscallCache cache;

  @Before
  public void setUp() {
    fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    cache = DefaultSyscallCache.newBuilder().build();
  }

  @Test
  public void statIfFound_regularFile() throws Exception {
    Path file = fs.getPath("/test/file.txt");
    file.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(file, "hello");

    FileStatus nofollow = cache.statIfFound(file, Symlinks.NOFOLLOW);
    assertThat(nofollow).isNotNull();
    assertThat(nofollow.isFile()).isTrue();
    assertThat(nofollow.isSymbolicLink()).isFalse();

    FileStatus follow = cache.statIfFound(file, Symlinks.FOLLOW);
    assertThat(follow).isNotNull();
    assertThat(follow.isFile()).isTrue();
  }

  @Test
  public void statIfFound_directory() throws Exception {
    Path dir = fs.getPath("/test/dir");
    dir.createDirectoryAndParents();

    FileStatus nofollow = cache.statIfFound(dir, Symlinks.NOFOLLOW);
    assertThat(nofollow).isNotNull();
    assertThat(nofollow.isDirectory()).isTrue();

    FileStatus follow = cache.statIfFound(dir, Symlinks.FOLLOW);
    assertThat(follow).isNotNull();
    assertThat(follow.isDirectory()).isTrue();
  }

  @Test
  public void statIfFound_nonExistentPath() throws Exception {
    Path missing = fs.getPath("/test/missing.txt");
    assertThat(cache.statIfFound(missing, Symlinks.NOFOLLOW)).isNull();
    assertThat(cache.statIfFound(missing, Symlinks.FOLLOW)).isNull();
  }

  @Test
  public void statIfFound_validSymlink() throws Exception {
    Path target = fs.getPath("/test/target.txt");
    target.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(target, "content");

    Path link = fs.getPath("/test/link.txt");
    link.createSymbolicLink(PathFragment.create("target.txt"));

    FileStatus nofollow = cache.statIfFound(link, Symlinks.NOFOLLOW);
    assertThat(nofollow).isNotNull();
    assertThat(nofollow.isSymbolicLink()).isTrue();

    FileStatus follow = cache.statIfFound(link, Symlinks.FOLLOW);
    assertThat(follow).isNotNull();
    assertThat(follow.isSymbolicLink()).isFalse();
    assertThat(follow.isFile()).isTrue();
  }

  @Test
  public void statIfFound_danglingSymlink() throws Exception {
    Path link = fs.getPath("/test/dangling.txt");
    link.getParentDirectory().createDirectoryAndParents();
    link.createSymbolicLink(PathFragment.create("nonexistent.txt"));

    FileStatus nofollow = cache.statIfFound(link, Symlinks.NOFOLLOW);
    assertThat(nofollow).isNotNull();
    assertThat(nofollow.isSymbolicLink()).isTrue();

    FileStatus follow = cache.statIfFound(link, Symlinks.FOLLOW);
    assertThat(follow).isNull();
  }

  @Test
  public void getType_resolvesCorrectly() throws Exception {
    Path dir = fs.getPath("/test/dir");
    dir.createDirectoryAndParents();
    Path file = dir.getChild("sub.txt");
    FileSystemUtils.writeContentAsLatin1(file, "content");

    // Populate readdir cache
    Collection<Dirent> dirents = cache.readdir(dir);
    assertThat(dirents).hasSize(1);

    DirentTypeWithSkip type = cache.getType(file, Symlinks.NOFOLLOW);
    assertThat(type).isNotNull();
    assertThat(type.getType()).isEqualTo(Dirent.Type.FILE);
  }

  @Test
  public void getType_withFollow_cachedEntries() throws Exception {
    Path file = fs.getPath("/test/target.txt");
    file.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(file, "target");

    Path link = fs.getPath("/test/link.txt");
    link.createSymbolicLink(PathFragment.create("target.txt"));

    // 1. Regular file cached in statCache
    assertThat(cache.statIfFound(file, Symlinks.FOLLOW)).isNotNull();
    DirentTypeWithSkip fileType = cache.getType(file, Symlinks.FOLLOW);
    assertThat(fileType).isNotNull();
    assertThat(fileType.getType()).isEqualTo(Dirent.Type.FILE);

    // 2. Symlink with followStatus cached in SymlinkEntry
    assertThat(cache.statIfFound(link, Symlinks.FOLLOW)).isNotNull();
    DirentTypeWithSkip linkTypeFollow = cache.getType(link, Symlinks.FOLLOW);
    assertThat(linkTypeFollow).isNotNull();
    assertThat(linkTypeFollow.getType()).isEqualTo(Dirent.Type.FILE);

    // 3. NO_STATUS cached for missing file
    Path missing = fs.getPath("/test/missing.txt");
    assertThat(cache.statIfFound(missing, Symlinks.FOLLOW)).isNull();
    assertThat(cache.getType(missing, Symlinks.FOLLOW)).isNull();
  }

  @Test
  public void clear_resetsCache() throws Exception {
    Path file = fs.getPath("/test/file.txt");
    file.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(file, "v1");

    FileStatus status1 = cache.statIfFound(file, Symlinks.NOFOLLOW);
    assertThat(status1).isNotNull();

    cache.clear();

    FileStatus status2 = cache.statIfFound(file, Symlinks.NOFOLLOW);
    assertThat(status2).isNotNull();
  }
}
