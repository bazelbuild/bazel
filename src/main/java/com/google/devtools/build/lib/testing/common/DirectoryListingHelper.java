// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.testing.common;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;

/** Namespace for helpers to test recursive directory listings. */
public final class DirectoryListingHelper {

  private DirectoryListingHelper() {}

  /** Shorthand for {@link Dirent} of {@link Dirent.Type#FILE} type with a given name. */
  public static Dirent file(String name) {
    return new Dirent(name, Dirent.Type.FILE);
  }

  /** Shorthand for {@link Dirent} of {@link Dirent.Type#SYMLINK} type with a given name. */
  public static Dirent symlink(String name) {
    return new Dirent(name, Dirent.Type.SYMLINK);
  }

  /** Shorthand for {@link Dirent} of {@link Dirent.Type#DIRECTORY} type with a given name. */
  public static Dirent directory(String name) {
    return new Dirent(name, Dirent.Type.DIRECTORY);
  }

  /**
   * Returns all of the leaf {@linkplain Dirent dirents} under a given directory.
   *
   * <p>For directory structure of:
   *
   * <pre>
   *   dir/dir2
   *   dir/file1
   *   dir/subdir/file2
   * </pre>
   *
   * will return: {@code FILE(dir/file1), FILE(dir/subdir/file2), DIRECTORY(dir/dir2)}.
   */
  public static ImmutableList<Dirent> leafDirectoryEntries(Path path) throws IOException {
    ImmutableList.Builder<Dirent> entries = ImmutableList.builder();
    leafDirectoryEntriesInternal(path, "", entries);
    return entries.build();
  }

  private static void leafDirectoryEntriesInternal(
      Path path, String prefix, ImmutableList.Builder<Dirent> entries) throws IOException {
    boolean isEmpty = true;
    for (Dirent dirent : path.readdir(Symlinks.NOFOLLOW)) {
      isEmpty = false;
      String entryName = prefix.isEmpty() ? dirent.getName() : prefix + "/" + dirent.getName();

      if (dirent.getType() == Dirent.Type.DIRECTORY) {
        leafDirectoryEntriesInternal(path.getChild(dirent.getName()), entryName, entries);
        continue;
      }

      entries.add(new Dirent(entryName, dirent.getType()));
    }

    // Skip adding the root if it's empty.
    if (isEmpty && !prefix.isEmpty()) {
      entries.add(new Dirent(prefix, Dirent.Type.DIRECTORY));
    }
  }
}
