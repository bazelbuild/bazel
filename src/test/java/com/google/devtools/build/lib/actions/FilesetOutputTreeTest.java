// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link FilesetOutputTree}. */
@RunWith(JUnit4.class)
public final class FilesetOutputTreeTest {

  private static final PathFragment EXEC_ROOT = PathFragment.create("/root");

  private static FilesetOutputSymlink symlink(String from, String to) {
    return FilesetOutputSymlink.createForTesting(
        PathFragment.create(from), PathFragment.create(to), EXEC_ROOT);
  }

  private static VisitedSymlink visitedSymlink(String from, String to) {
    return new VisitedSymlink(PathFragment.create(from), PathFragment.create(to));
  }

  private record VisitedSymlink(PathFragment name, PathFragment target) {}

  private static FilesetOutputTree createTree(FilesetOutputSymlink... symlinks) {
    return FilesetOutputTree.create(ImmutableList.copyOf(symlinks));
  }

  private static ImmutableList<VisitedSymlink> visit(FilesetOutputTree filesetOutput) {
    var visited = ImmutableList.<VisitedSymlink>builder();
    filesetOutput.visitSymlinks(
        (name, target, metadata) -> visited.add(new VisitedSymlink(name, target)));
    return visited.build();
  }

  @Test
  public void empty() throws Exception {
    assertThat(visit(FilesetOutputTree.EMPTY)).isEmpty();
  }

  @Test
  public void canonicalEmptyInstance() {
    assertThat(createTree()).isSameInstanceAs(FilesetOutputTree.EMPTY);
  }

  @Test
  public void singleFile() {
    FilesetOutputTree filesetOutput = createTree(symlink("bar", "/dir/file"));

    assertThat(visit(filesetOutput)).containsExactly(visitedSymlink("bar", "/dir/file"));
  }

  @Test
  public void twoFiles() {
    FilesetOutputTree filesetOutput =
        createTree(symlink("bar", "/dir/file"), symlink("baz", "/dir/file"));

    assertThat(visit(filesetOutput))
        .containsExactly(visitedSymlink("bar", "/dir/file"), visitedSymlink("baz", "/dir/file"));
  }

  @Test
  public void symlinkWithExecRootRelativePath() {
    FilesetOutputTree filesetOutput =
        createTree(symlink("bar", EXEC_ROOT.getRelative("foo/bar").getPathString()));

    assertThat(visit(filesetOutput)).containsExactly(visitedSymlink("bar", "foo/bar"));
  }
}
