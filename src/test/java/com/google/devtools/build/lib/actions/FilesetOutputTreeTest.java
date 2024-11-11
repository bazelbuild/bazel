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
import static com.google.devtools.build.lib.actions.FilesetOutputTree.RelativeSymlinkBehavior.ERROR;
import static com.google.devtools.build.lib.actions.FilesetOutputTree.RelativeSymlinkBehaviorWithoutError.IGNORE;
import static com.google.devtools.build.lib.actions.FilesetOutputTree.RelativeSymlinkBehaviorWithoutError.RESOLVE_FULLY;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.FilesetOutputTree.ForbiddenRelativeSymlinkException;
import com.google.devtools.build.lib.actions.FilesetOutputTree.RelativeSymlinkBehavior;
import com.google.devtools.build.lib.actions.FilesetOutputTree.RelativeSymlinkBehaviorWithoutError;
import com.google.devtools.build.lib.actions.FilesetOutputTreeTest.CommonTests;
import com.google.devtools.build.lib.actions.FilesetOutputTreeTest.OneOffTests;
import com.google.devtools.build.lib.actions.FilesetOutputTreeTest.ResolvingTests;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.runners.Suite;

/** Tests for {@link FilesetOutputTree}. */
@RunWith(Suite.class)
@Suite.SuiteClasses({CommonTests.class, OneOffTests.class, ResolvingTests.class})
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

  private static ImmutableList<VisitedSymlink> visit(
      FilesetOutputTree filesetOutput, RelativeSymlinkBehavior relSymlinkBehavior)
      throws ForbiddenRelativeSymlinkException {
    var visited = ImmutableList.<VisitedSymlink>builder();
    filesetOutput.visitSymlinks(
        relSymlinkBehavior,
        (name, target, metadata) -> visited.add(new VisitedSymlink(name, target)));
    return visited.build();
  }

  private static ImmutableList<VisitedSymlink> visit(
      FilesetOutputTree filesetOutput, RelativeSymlinkBehaviorWithoutError relSymlinkBehavior) {
    var visited = ImmutableList.<VisitedSymlink>builder();
    filesetOutput.visitSymlinks(
        relSymlinkBehavior,
        (name, target, metadata) -> visited.add(new VisitedSymlink(name, target)));
    return visited.build();
  }

  /** Tests that apply to all relative symlink behaviors. */
  @RunWith(TestParameterInjector.class)
  public static final class CommonTests {

    @TestParameter private RelativeSymlinkBehavior behavior;

    @Test
    public void empty() throws Exception {
      assertThat(visit(FilesetOutputTree.EMPTY, behavior)).isEmpty();
    }

    @Test
    public void singleFile() throws Exception {
      FilesetOutputTree filesetOutput = createTree(symlink("bar", "/dir/file"));

      assertThat(visit(filesetOutput, behavior))
          .containsExactly(visitedSymlink("bar", "/dir/file"));
    }

    @Test
    public void twoFiles() throws Exception {
      FilesetOutputTree filesetOutput =
          createTree(symlink("bar", "/dir/file"), symlink("baz", "/dir/file"));

      assertThat(visit(filesetOutput, behavior))
          .containsExactly(visitedSymlink("bar", "/dir/file"), visitedSymlink("baz", "/dir/file"));
    }

    @Test
    public void symlinkWithExecRootRelativePath() throws Exception {
      FilesetOutputTree filesetOutput =
          createTree(symlink("bar", EXEC_ROOT.getRelative("foo/bar").getPathString()));

      assertThat(visit(filesetOutput, behavior)).containsExactly(visitedSymlink("bar", "foo/bar"));
    }
  }

  /** Tests that apply to a specific relative symlink behavior. */
  @RunWith(JUnit4.class)
  public static final class OneOffTests {

    @Test
    public void canonicalEmptyInstance() {
      assertThat(createTree()).isSameInstanceAs(FilesetOutputTree.EMPTY);
    }

    @Test
    public void errorOnRelativeSymlink() {
      FilesetOutputTree filesetOutput =
          createTree(symlink("bar", "foo"), symlink("foo", "/foo/bar"));

      var e =
          assertThrows(ForbiddenRelativeSymlinkException.class, () -> visit(filesetOutput, ERROR));
      assertThat(e).hasMessageThat().contains("Fileset symlink bar -> foo is not absolute");
    }

    @Test
    public void ignoredRelativeSymlink() {
      FilesetOutputTree filesetOutput =
          createTree(symlink("bar", "foo"), symlink("foo", "/foo/bar"));

      assertThat(visit(filesetOutput, IGNORE)).containsExactly(visitedSymlink("foo", "/foo/bar"));
    }

    @Test
    public void resolvedRelativeDirectorySymlink() {
      FilesetOutputTree filesetOutput =
          createTree(symlink("foo/subdir/f1", "/foo/subdir/f1"), symlink("foo/bar", "subdir"));

      assertThat(visit(filesetOutput, RESOLVE_FULLY))
          .containsExactly(
              visitedSymlink("foo/subdir/f1", "/foo/subdir/f1"),
              visitedSymlink("foo/bar/f1", "/foo/subdir/f1"));
    }
  }

  /** Tests that apply resolving relative symlink behaviors. */
  @RunWith(TestParameterInjector.class)
  public static final class ResolvingTests {

    @TestParameter({"RESOLVE", "RESOLVE_FULLY"})
    private RelativeSymlinkBehaviorWithoutError behavior;

    @Test
    public void resolvedRelativeSymlink() {
      FilesetOutputTree filesetOutput =
          createTree(symlink("bar", "foo"), symlink("foo", "/foo/bar"));

      assertThat(visit(filesetOutput, behavior))
          .containsExactly(visitedSymlink("bar", "/foo/bar"), visitedSymlink("foo", "/foo/bar"));
    }

    @Test
    public void resolvedRelativeSymlinkWithDotSlash() {
      FilesetOutputTree filesetOutput =
          createTree(symlink("bar", "./foo"), symlink("foo", "/foo/bar"));

      assertThat(visit(filesetOutput, behavior))
          .containsExactly(visitedSymlink("bar", "/foo/bar"), visitedSymlink("foo", "/foo/bar"));
    }

    @Test
    public void symlinkCycle() {
      FilesetOutputTree filesetOutput =
          createTree(
              symlink("bar", "foo"),
              symlink("foo", "biz"),
              symlink("biz", "bar"),
              symlink("reg", "/file"));

      assertThat(visit(filesetOutput, behavior)).containsExactly(visitedSymlink("reg", "/file"));
    }

    @Test
    public void unboundedSymlinkDescendant() {
      FilesetOutputTree filesetOutput =
          createTree(symlink("p", "a/b"), symlink("a/b", "../b/c"), symlink("reg", "/file"));

      assertThat(visit(filesetOutput, behavior)).containsExactly(visitedSymlink("reg", "/file"));
    }

    @Test
    public void unboundedSymlinkAncestor() {
      FilesetOutputTree filesetOutput =
          createTree(symlink("a/b", "c/d"), symlink("a/c/d", ".././a"), symlink("reg", "/file"));

      assertThat(visit(filesetOutput, behavior)).containsExactly(visitedSymlink("reg", "/file"));
    }

    @Test
    public void resolvedRelativeSymlinkWithDotDotSlash() {
      FilesetOutputTree filesetOutput =
          createTree(symlink("bar/bar", "../foo/foo"), symlink("foo/foo", "/foo/bar"));

      assertThat(visit(filesetOutput, behavior))
          .containsExactly(
              visitedSymlink("bar/bar", "/foo/bar"), visitedSymlink("foo/foo", "/foo/bar"));
    }

    @Test
    public void unresolvableRelativeSymlink() {
      FilesetOutputTree filesetOutput = createTree(symlink("bar", "foo"));

      assertThat(visit(filesetOutput, behavior)).isEmpty();
    }

    @Test
    public void unresolvableRelativeSymlinkToRelativeSymlink() {
      FilesetOutputTree filesetOutput = createTree(symlink("bar", "foo"), symlink("foo", "baz"));

      assertThat(visit(filesetOutput, behavior)).isEmpty();
    }
  }
}
