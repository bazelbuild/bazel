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
package com.google.devtools.build.lib.vfs;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class LocalBulkDeleterTest {
  private final FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
  private Path execRoot;
  private Path outsideTarget;
  private Path trashRoot;
  private LocalBulkDeleter deleter;

  @Before
  public void setUp() throws Exception {
    execRoot = fs.getPath("/output_base/execroot/_main");
    execRoot.createDirectoryAndParents();
    outsideTarget = fs.getPath("/elsewhere/target.txt");
    outsideTarget.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContent(outsideTarget, UTF_8, "keep me");

    Path stale = execRoot.getRelative("bazel-out/k8-fastbuild/bin/pkg/stale");
    stale.getRelative("nested/deeper").createDirectoryAndParents();
    FileSystemUtils.writeContent(stale.getRelative("a.o"), UTF_8, "a");
    FileSystemUtils.writeContent(stale.getRelative("nested/deeper/b.o"), UTF_8, "b");
    stale.getRelative("nested/outbound-link").createSymbolicLink(outsideTarget);
    FileSystemUtils.writeContent(
        execRoot.getRelative("bazel-out/k8-fastbuild/bin/pkg/sibling.txt"), UTF_8, "sibling");

    trashRoot = fs.getPath("/output_base/_bulk_delete_trash");
    deleter = new LocalBulkDeleter(execRoot, trashRoot, "this-build");
  }

  @After
  public void drainDeleter() throws Exception {
    deleter.awaitPendingDeletes();
  }

  @Test
  public void deletesTopLevelSymlinkToDirectoryWithoutFollowingIt() throws Exception {
    Path outsideDir = fs.getPath("/elsewhere/linked-dir");
    outsideDir.createDirectoryAndParents();
    FileSystemUtils.writeContent(outsideDir.getRelative("kept.txt"), UTF_8, "kept");
    Path link = execRoot.getRelative("bazel-out/k8-fastbuild/bin/pkg/dir-link");
    link.createSymbolicLink(outsideDir);

    deleter.bulkDelete(
        ImmutableList.of(PathFragment.create("bazel-out/k8-fastbuild/bin/pkg/dir-link")));
    deleter.awaitPendingDeletes();

    assertThat(link.exists(Symlinks.NOFOLLOW)).isFalse();
    assertThat(FileSystemUtils.readContent(outsideDir.getRelative("kept.txt"), UTF_8))
        .isEqualTo("kept");
  }

  @Test
  public void deletesEveryTreeAfterManyAreQueued() throws Exception {
    ImmutableList.Builder<PathFragment> paths = ImmutableList.builder();
    for (int i = 0; i < 20; i++) {
      Path tree = execRoot.getRelative("bazel-out/k8-fastbuild/bin/many/t" + i + "/inner");
      tree.createDirectoryAndParents();
      FileSystemUtils.writeContent(tree.getRelative("f"), UTF_8, "x");
      paths.add(PathFragment.create("bazel-out/k8-fastbuild/bin/many/t" + i));
    }

    deleter.bulkDelete(paths.build());
    deleter.awaitPendingDeletes();

    assertThat(execRoot.getRelative("bazel-out/k8-fastbuild/bin/many").getDirectoryEntries())
        .isEmpty();
    assertThat(trashRoot.getRelative("this-build").getDirectoryEntries()).isEmpty();
  }

  @Test
  public void trashIsEmptyAfterPendingDeletesFinish() throws Exception {
    deleter.bulkDelete(
        ImmutableList.of(PathFragment.create("bazel-out/k8-fastbuild/bin/pkg/stale")));

    deleter.awaitPendingDeletes();

    assertThat(trashRoot.getRelative("this-build").getDirectoryEntries()).isEmpty();
    assertThat(FileSystemUtils.readContent(outsideTarget, UTF_8)).isEqualTo("keep me");
  }

  @Test
  public void deletesTrashLeftByEarlierBuild() throws Exception {
    deleter.awaitPendingDeletes();
    Path leftover = trashRoot.getRelative("earlier-build/0/nested");
    leftover.createDirectoryAndParents();
    FileSystemUtils.writeContent(leftover.getRelative("f"), UTF_8, "x");

    LocalBulkDeleter next = new LocalBulkDeleter(execRoot, trashRoot, "next-build");
    next.awaitPendingDeletes();

    assertThat(trashRoot.getDirectoryEntries()).isEmpty();
  }

  @Test
  public void deletesTreeAndLeavesSiblingAndLinkTarget() throws Exception {
    deleter.bulkDelete(
        ImmutableList.of(PathFragment.create("bazel-out/k8-fastbuild/bin/pkg/stale")));

    Path pkg = execRoot.getRelative("bazel-out/k8-fastbuild/bin/pkg");
    assertThat(pkg.getRelative("stale").exists(Symlinks.NOFOLLOW)).isFalse();
    assertThat(pkg.getDirectoryEntries()).containsExactly(pkg.getRelative("sibling.txt"));
    assertThat(FileSystemUtils.readContent(pkg.getRelative("sibling.txt"), UTF_8))
        .isEqualTo("sibling");
    assertThat(FileSystemUtils.readContent(outsideTarget, UTF_8)).isEqualTo("keep me");
  }

  @Test
  public void deletesPlainFile() throws Exception {
    deleter.bulkDelete(
        ImmutableList.of(PathFragment.create("bazel-out/k8-fastbuild/bin/pkg/sibling.txt")));

    Path pkg = execRoot.getRelative("bazel-out/k8-fastbuild/bin/pkg");
    assertThat(pkg.getDirectoryEntries()).containsExactly(pkg.getRelative("stale"));
  }

  @Test
  public void missingPathSucceeds() throws Exception {
    deleter.bulkDelete(ImmutableList.of(PathFragment.create("bazel-out/never/existed")));

    assertThat(execRoot.getRelative("bazel-out/never/existed").exists()).isFalse();
    assertThat(execRoot.getRelative("bazel-out/k8-fastbuild/bin/pkg/stale/a.o").exists()).isTrue();
  }

  @Test
  public void secondDeleteConvergesToSameState() throws Exception {
    ImmutableList<PathFragment> paths =
        ImmutableList.of(
            PathFragment.create("bazel-out/k8-fastbuild/bin/pkg/stale"),
            PathFragment.create("bazel-out/k8-fastbuild/bin/pkg/stale/nested"));

    deleter.bulkDelete(paths);
    deleter.bulkDelete(paths);

    Path pkg = execRoot.getRelative("bazel-out/k8-fastbuild/bin/pkg");
    assertThat(pkg.getDirectoryEntries()).containsExactly(pkg.getRelative("sibling.txt"));
    assertThat(FileSystemUtils.readContent(outsideTarget, UTF_8)).isEqualTo("keep me");
  }

  @Test
  public void deletesTreeWithManyChildDirectories() throws Exception {
    Path wide = execRoot.getRelative("bazel-out/k8-fastbuild/bin/pkg/wide");
    for (int i = 0; i < 50; i++) {
      Path dir = wide.getRelative("d" + i + "/inner");
      dir.createDirectoryAndParents();
      FileSystemUtils.writeContent(dir.getRelative("f"), UTF_8, "x");
    }
    FileSystemUtils.writeContent(wide.getRelative("top.txt"), UTF_8, "top");

    deleter.bulkDelete(ImmutableList.of(PathFragment.create("bazel-out/k8-fastbuild/bin/pkg/wide")));

    Path pkg = execRoot.getRelative("bazel-out/k8-fastbuild/bin/pkg");
    assertThat(pkg.getDirectoryEntries())
        .containsExactly(pkg.getRelative("stale"), pkg.getRelative("sibling.txt"));
  }

  @Test
  public void deletesReadOnlyTree() throws Exception {
    Path stale = execRoot.getRelative("bazel-out/k8-fastbuild/bin/pkg/stale");
    stale.getRelative("nested/deeper").setWritable(false);
    stale.getRelative("nested").setWritable(false);
    stale.setWritable(false);

    deleter.bulkDelete(ImmutableList.of(PathFragment.create("bazel-out/k8-fastbuild/bin/pkg/stale")));

    Path pkg = execRoot.getRelative("bazel-out/k8-fastbuild/bin/pkg");
    assertThat(pkg.getDirectoryEntries()).containsExactly(pkg.getRelative("sibling.txt"));
  }

  @Test
  public void doesNotFollowSymlinkToDirectory() throws Exception {
    Path outsideDir = fs.getPath("/elsewhere/dir");
    outsideDir.createDirectoryAndParents();
    FileSystemUtils.writeContent(outsideDir.getRelative("kept.txt"), UTF_8, "kept");
    Path stale = execRoot.getRelative("bazel-out/k8-fastbuild/bin/pkg/stale");
    stale.getRelative("dir-link").createSymbolicLink(outsideDir);

    deleter.bulkDelete(ImmutableList.of(PathFragment.create("bazel-out/k8-fastbuild/bin/pkg/stale")));

    assertThat(stale.exists(Symlinks.NOFOLLOW)).isFalse();
    assertThat(FileSystemUtils.readContent(outsideDir.getRelative("kept.txt"), UTF_8))
        .isEqualTo("kept");
  }

  @Test
  public void pendingInterruptStillDeletesAndKeepsTheFlag() throws Exception {
    Thread.currentThread().interrupt();

    deleter.bulkDelete(
        ImmutableList.of(PathFragment.create("bazel-out/k8-fastbuild/bin/pkg/stale")));

    assertThat(Thread.currentThread().isInterrupted()).isTrue();
    Thread.interrupted();
    assertThat(execRoot.getRelative("bazel-out/k8-fastbuild/bin/pkg/stale").exists()).isFalse();
    assertThat(execRoot.getRelative("bazel-out/k8-fastbuild/bin/pkg/sibling.txt").exists()).isTrue();
  }

  @Test
  public void deletesFileWhoseParentIsNotWritable() throws Exception {
    Path parent = execRoot.getRelative("bazel-out/k8-fastbuild/bin/pkg/locked");
    parent.createDirectoryAndParents();
    FileSystemUtils.writeContent(parent.getRelative("somefile.out"), UTF_8, "x");
    parent.setWritable(false);

    deleter.bulkDelete(
        ImmutableList.of(PathFragment.create("bazel-out/k8-fastbuild/bin/pkg/locked/somefile.out")));

    assertThat(parent.getRelative("somefile.out").exists()).isFalse();
    assertThat(execRoot.getRelative("bazel-out/k8-fastbuild/bin/pkg/sibling.txt").exists()).isTrue();
  }

  @Test
  public void deletesTreeWhoseParentIsNotWritable() throws Exception {
    Path parent = execRoot.getRelative("bazel-out/k8-fastbuild/bin/pkg/locked");
    Path tree = parent.getRelative("somedir");
    tree.createDirectoryAndParents();
    FileSystemUtils.writeContent(tree.getRelative("somefile"), UTF_8, "x");
    parent.setWritable(false);

    deleter.bulkDelete(
        ImmutableList.of(PathFragment.create("bazel-out/k8-fastbuild/bin/pkg/locked/somedir")));

    assertThat(tree.exists()).isFalse();
    assertThat(execRoot.getRelative("bazel-out/k8-fastbuild/bin/pkg/sibling.txt").exists()).isTrue();
  }
}
