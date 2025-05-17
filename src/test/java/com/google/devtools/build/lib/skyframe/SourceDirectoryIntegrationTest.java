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
package com.google.devtools.build.lib.skyframe;

import static com.google.devtools.build.lib.vfs.FileSystemUtils.createEmptyFile;
import static com.google.devtools.build.lib.vfs.FileSystemUtils.ensureSymbolicLink;
import static com.google.devtools.build.lib.vfs.FileSystemUtils.touchFile;
import static com.google.devtools.build.lib.vfs.FileSystemUtils.writeContent;
import static com.google.devtools.build.lib.vfs.FileSystemUtils.writeIsoLatin1;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Integration test for invalidation of actions that consume source directories. */
@RunWith(JUnit4.class)
public final class SourceDirectoryIntegrationTest extends BuildIntegrationTestCase {

  private Path sourceDir;

  @Override
  protected ImmutableSet<EventKind> additionalEventsToCollect() {
    return ImmutableSet.of(EventKind.FINISH);
  }

  @Before
  public void setUpGenrule() throws Exception {
    write(
        "foo/BUILD",
        """
        genrule(
            name = "foo",
            srcs = ["dir"],
            outs = ["foo.out"],
            cmd = "touch $@",
        )
        """);

    sourceDir = getWorkspace().getRelative("foo/dir");
    sourceDir.createDirectoryAndParents();
    writeIsoLatin1(sourceDir.getRelative("file1"), "content");
    writeIsoLatin1(sourceDir.getRelative("file2"), "content");
    writeIsoLatin1(sourceDir.getRelative("file3"), "other content");
    sourceDir.getRelative("symlink").createSymbolicLink(PathFragment.create("file3"));
    sourceDir
        .getRelative("dangling_symlink")
        .createSymbolicLink(PathFragment.create("does_not_exist"));

    Path subDir = sourceDir.getRelative("subdir");
    subDir.createDirectory();
    writeIsoLatin1(subDir.getRelative("file1"), "content");
    writeIsoLatin1(subDir.getRelative("file2"), "content");
    writeIsoLatin1(subDir.getRelative("file3"), "other content");
    subDir.getRelative("symlink").createSymbolicLink(PathFragment.create("file3"));
    subDir
        .getRelative("dangling_symlink")
        .createSymbolicLink(PathFragment.create("does_not_exist"));

    subDir.getRelative("nested").createDirectory();
    subDir.getRelative("nested2").createDirectory();
    subDir.getRelative("nested_non_empty").createDirectory();
    writeIsoLatin1(subDir.getRelative("nested_non_empty/file1"), "content");

    buildTarget("//foo");
    assertContainsEvent(events.collector(), "Executing genrule //foo:foo");

    events.collector().clear();
  }

  @Test
  public void nothingModified_doesNotInvalidateAction() throws Exception {
    assertNotInvalidatedByBuild();
  }

  @Test
  public void touched_doesNotInvalidateAction() throws Exception {
    touchFile(sourceDir);
    assertNotInvalidatedByBuild();
  }

  @Test
  public void topLevelFileTouched_doesNotInvalidateAction() throws Exception {
    touchFile(sourceDir.getRelative("file1"));
    assertNotInvalidatedByBuild();
  }

  @Test
  public void topLevelDirTouched_doesNotInvalidateAction() throws Exception {
    touchFile(sourceDir.getRelative("subdir"));
    assertNotInvalidatedByBuild();
  }

  @Test
  public void nestedFileTouched_doesNotInvalidateAction() throws Exception {
    touchFile(sourceDir.getRelative("subdir/file1"));
    assertNotInvalidatedByBuild();
  }

  @Test
  public void nestedDirTouched_doesNotInvalidateAction() throws Exception {
    sourceDir.getRelative("subdir/nested").setLastModifiedTime(Path.NOW_SENTINEL_TIME);
    assertNotInvalidatedByBuild();
  }

  @Test
  public void topLevelFileDeleted_invalidatesAction() throws Exception {
    sourceDir.getRelative("file1").delete();
    assertInvalidatedByBuild();
  }

  @Test
  public void nestedFileDeleted_invalidatesAction() throws Exception {
    sourceDir.getRelative("subdir/file1").delete();
    assertInvalidatedByBuild();
  }

  @Test
  public void topLevelFileModified_invalidatesAction() throws Exception {
    writeIsoLatin1(sourceDir.getRelative("file1"), "modified content");
    assertInvalidatedByBuild();
  }

  @Test
  public void nestedFileModified_invalidatesAction() throws Exception {
    writeIsoLatin1(sourceDir.getRelative("subdir/file1"), "modified content");
    assertInvalidatedByBuild();
  }

  @Test
  public void topLevelFileAdded_invalidatesAction() throws Exception {
    writeIsoLatin1(sourceDir.getRelative("new_file"), "modified content");
    assertInvalidatedByBuild();
  }

  @Test
  public void nestedFileAdded_invalidatesAction() throws Exception {
    writeIsoLatin1(sourceDir.getRelative("subdir/new_file"), "modified content");
    assertInvalidatedByBuild();
  }

  @Test
  public void emptyDirAdded_invalidatesAction() throws Exception {
    sourceDir.getRelative("subdir/nested3").createDirectory();
    assertInvalidatedByBuild();
  }

  @Test
  public void emptyDirDeleted_invalidatesAction() throws Exception {
    sourceDir.getRelative("subdir/nested").delete();
    assertInvalidatedByBuild();
  }

  @Test
  public void emptyDirReplacedWithEmptyFile_invalidatesAction() throws Exception {
    Path dir = sourceDir.getRelative("subdir/nested");
    dir.delete();
    createEmptyFile(dir);
    assertInvalidatedByBuild();
  }

  @Test
  public void fileAddedToEmptyDir_invalidatesAction() throws Exception {
    createEmptyFile(sourceDir.getRelative("subdir/nested/file1"));
    assertInvalidatedByBuild();
  }

  @Test
  public void fileReplacedByIdenticalSymlink_doesNotInvalidateAction() throws Exception {
    Path file = sourceDir.getRelative("file1");
    file.delete();
    file.createSymbolicLink(sourceDir.getRelative("file2"));
    assertNotInvalidatedByBuild();
  }

  @Test
  public void fileReplacedByDifferentSymlink_invalidatesAction() throws Exception {
    Path file = sourceDir.getRelative("file1");
    file.delete();
    file.createSymbolicLink(sourceDir.getRelative("file3"));
    assertInvalidatedByBuild();
  }

  @Test
  @Ignore("TODO(#25834)")
  public void emptyDirReplacedWithIdenticalSymlink_doesNotInvalidateAction() throws Exception {
    Path dir = sourceDir.getRelative("subdir/nested2");
    dir.delete();
    dir.createSymbolicLink(PathFragment.create("nested"));
    assertNotInvalidatedByBuild();
  }

  @Test
  public void emptyDirReplacedWithDifferentSymlink_invalidatesAction() throws Exception {
    Path dir = sourceDir.getRelative("subdir/nested2");
    dir.delete();
    dir.createSymbolicLink(PathFragment.create("nested_non_empty"));
    assertInvalidatedByBuild();
  }

  @Test
  public void danglingSymlinkModified_invalidatesAction() throws Exception {
    ensureSymbolicLink(
        sourceDir.getRelative("dangling_symlink"), PathFragment.create("still_does_not_exist"));
    assertInvalidatedByBuild();
  }

  @Test
  public void danglingSymlinkReplacedWithFile_invalidatesAction() throws Exception {
    Path danglingSymlink = sourceDir.getRelative("dangling_symlink");
    String target = danglingSymlink.readSymbolicLink().getPathString();
    danglingSymlink.delete();
    writeContent(danglingSymlink, ISO_8859_1, target);
    assertInvalidatedByBuild();
  }

  @Test
  public void crossingPackageBoundary_fails() throws Exception {
    createEmptyFile(sourceDir.getRelative("subdir/BUILD"));
    assertThrows(BuildFailedException.class, () -> buildTarget("//foo"));
    assertContainsEvent(
        "Directory artifact foo/dir crosses package boundary into package rooted at"
            + " foo/dir/subdir");
  }

  @Test
  public void infiniteSymlinkExpansion_fails() throws Exception {
    Path dir = sourceDir.getRelative("subdir/nested2");
    dir.delete();
    dir.createSymbolicLink(PathFragment.create(".."));
    assertThrows(BuildFailedException.class, () -> buildTarget("//foo"));
    assertContainsEvent("infinite symlink expansion detected");
    assertContainsEvent("foo/dir/subdir/nested2");
  }

  private static final String GENRULE_EVENT = "Executing genrule //foo:foo";

  private void assertInvalidatedByBuild() throws Exception {
    buildTarget("//foo");
    assertContainsEvent(events.collector(), GENRULE_EVENT);
  }

  private void assertNotInvalidatedByBuild() throws Exception {
    buildTarget("//foo");
    assertDoesNotContainEvent(events.collector(), GENRULE_EVENT);
  }
}
