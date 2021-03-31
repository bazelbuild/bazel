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
package com.google.devtools.build.lib.exec;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.FilesetManifest.RelativeSymlinkBehavior.ERROR;
import static com.google.devtools.build.lib.actions.FilesetManifest.RelativeSymlinkBehavior.IGNORE;
import static com.google.devtools.build.lib.actions.FilesetManifest.RelativeSymlinkBehavior.RESOLVE;
import static com.google.devtools.build.lib.actions.FilesetManifest.RelativeSymlinkBehavior.RESOLVE_FULLY;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.FilesetManifest;
import com.google.devtools.build.lib.actions.FilesetManifest.RelativeSymlinkBehavior;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.exec.FilesetManifestTest.ManifestCommonTests;
import com.google.devtools.build.lib.exec.FilesetManifestTest.OneOffManifestTests;
import com.google.devtools.build.lib.exec.FilesetManifestTest.ResolvingManifestTests;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.runners.Parameterized;
import org.junit.runners.Suite;

/** Tests for {@link FilesetManifest}. */
@RunWith(Suite.class)
@Suite.SuiteClasses({
  ManifestCommonTests.class,
  OneOffManifestTests.class,
  ResolvingManifestTests.class
})
public final class FilesetManifestTest {

  private static final PathFragment EXEC_ROOT = PathFragment.create("/root");

  private static FilesetOutputSymlink filesetSymlink(String from, String to) {
    return FilesetOutputSymlink.createForTesting(
        PathFragment.create(from), PathFragment.create(to), EXEC_ROOT);
  }

  /** Manifest tests that apply to all relative symlink behavior. */
  @RunWith(Parameterized.class)
  public static final class ManifestCommonTests {
    private final RelativeSymlinkBehavior behavior;

    @Parameterized.Parameters
    public static ImmutableCollection<Object[]> behaviors() {
      return ImmutableList.of(
          new Object[] {ERROR},
          new Object[] {RESOLVE},
          new Object[] {IGNORE},
          new Object[] {RESOLVE_FULLY});
    }

    public ManifestCommonTests(RelativeSymlinkBehavior behavior) {
      this.behavior = behavior;
    }

    @Test
    public void testEmptyManifest() throws Exception {
      List<FilesetOutputSymlink> symlinks = ImmutableList.of();

      FilesetManifest manifest =
          FilesetManifest.constructFilesetManifest(
              symlinks, PathFragment.create("out/foo"), behavior);

      assertThat(manifest.getEntries()).isEmpty();
    }

    @Test
    public void testManifestWithSingleFile() throws Exception {
      List<FilesetOutputSymlink> symlinks = ImmutableList.of(filesetSymlink("bar", "/dir/file"));

      FilesetManifest manifest =
          FilesetManifest.constructFilesetManifest(
              symlinks, PathFragment.create("out/foo"), behavior);

      assertThat(manifest.getEntries())
          .containsExactly(PathFragment.create("out/foo/bar"), "/dir/file");
    }

    @Test
    public void testManifestWithTwoFiles() throws Exception {
      List<FilesetOutputSymlink> symlinks =
          ImmutableList.of(filesetSymlink("bar", "/dir/file"), filesetSymlink("baz", "/dir/file"));

      FilesetManifest manifest =
          FilesetManifest.constructFilesetManifest(
              symlinks, PathFragment.create("out/foo"), behavior);

      assertThat(manifest.getEntries())
          .containsExactly(
              PathFragment.create("out/foo/bar"), "/dir/file",
              PathFragment.create("out/foo/baz"), "/dir/file");
    }

    @Test
    public void testManifestWithDirectory() throws Exception {
      List<FilesetOutputSymlink> symlinks = ImmutableList.of(filesetSymlink("bar", "/some"));

      FilesetManifest manifest =
          FilesetManifest.constructFilesetManifest(
              symlinks, PathFragment.create("out/foo"), behavior);

      assertThat(manifest.getEntries())
          .containsExactly(PathFragment.create("out/foo/bar"), "/some");
    }

    /** Regression test: code was previously crashing in this case. */
    @Test
    public void testManifestWithEmptyPath() throws Exception {
      List<FilesetOutputSymlink> symlinks = ImmutableList.of(filesetSymlink("bar", ""));

      FilesetManifest manifest =
          FilesetManifest.constructFilesetManifest(
              symlinks, PathFragment.create("out/foo"), behavior);

      assertThat(manifest.getEntries()).containsExactly(PathFragment.create("out/foo/bar"), null);
    }

    @Test
    public void testManifestWithExecRootRelativePath() throws Exception {
      List<FilesetOutputSymlink> symlinks =
          ImmutableList.of(filesetSymlink("bar", EXEC_ROOT.getRelative("foo/bar").getPathString()));

      FilesetManifest manifest =
          FilesetManifest.constructFilesetManifest(
              symlinks, PathFragment.create("out/foo"), behavior);

      assertThat(manifest.getEntries())
          .containsExactly(PathFragment.create("out/foo/bar"), "foo/bar");
    }

    /** Current behavior is first one wins. */
    @Test
    public void testDefactoBehaviorWithDuplicateEntries() throws Exception {
      List<FilesetOutputSymlink> symlinks =
          ImmutableList.of(filesetSymlink("bar", "/foo/bar"), filesetSymlink("bar", "/baz"));

      FilesetManifest manifest =
          FilesetManifest.constructFilesetManifest(
              symlinks, PathFragment.create("out/foo"), behavior);

      assertThat(manifest.getEntries())
          .containsExactly(PathFragment.create("out/foo/bar"), "/foo/bar");
    }
  }

  /** Manifest tests that apply to a specific relative symlink behavior. */
  @RunWith(JUnit4.class)
  public static final class OneOffManifestTests {

    @Test
    public void testManifestWithErrorOnRelativeSymlink() {
      List<FilesetOutputSymlink> symlinks =
          ImmutableList.of(filesetSymlink("bar", "foo"), filesetSymlink("foo", "/foo/bar"));

      // Because BugReport throws in tests, we catch the wrapped exception.
      IllegalStateException e =
          assertThrows(
              IllegalStateException.class,
              () ->
                  FilesetManifest.constructFilesetManifest(
                      symlinks, PathFragment.create("out/foo"), ERROR));
      assertThat(e).hasCauseThat().isInstanceOf(IOException.class);
      assertThat(e)
          .hasCauseThat()
          .hasMessageThat()
          .contains("runfiles target is not absolute: foo");
    }

    @Test
    public void testManifestWithIgnoredRelativeSymlink() throws Exception {
      List<FilesetOutputSymlink> symlinks =
          ImmutableList.of(filesetSymlink("bar", "foo"), filesetSymlink("foo", "/foo/bar"));

      FilesetManifest manifest =
          FilesetManifest.constructFilesetManifest(
              symlinks, PathFragment.create("out/foo"), IGNORE);

      assertThat(manifest.getEntries())
          .containsExactly(PathFragment.create("out/foo/foo"), "/foo/bar");
    }

    @Test
    public void testManifestWithResolvedRelativeDirectorySymlink() throws Exception {
      List<FilesetOutputSymlink> symlinks =
          ImmutableList.of(
              filesetSymlink("foo/subdir/f1", "/foo/subdir/f1"),
              filesetSymlink("foo/bar", "subdir"));

      FilesetManifest manifest =
          FilesetManifest.constructFilesetManifest(
              symlinks, PathFragment.create("out"), RESOLVE_FULLY);

      assertThat(manifest.getEntries())
          .containsExactly(
              PathFragment.create("out/foo/subdir/f1"), "/foo/subdir/f1",
              PathFragment.create("out/foo/bar/f1"), "/foo/subdir/f1");
    }
  }

  /** Manifest tests that apply resolving relative symlink behavior. */
  @RunWith(Parameterized.class)
  public static final class ResolvingManifestTests {
    private final RelativeSymlinkBehavior behavior;

    @Parameterized.Parameters
    public static ImmutableCollection<Object[]> behaviors() {
      return ImmutableList.of(new Object[] {RESOLVE}, new Object[] {RESOLVE_FULLY});
    }

    public ResolvingManifestTests(RelativeSymlinkBehavior behavior) {
      this.behavior = behavior;
    }

    @Test
    public void testManifestWithResolvedRelativeSymlink() throws Exception {
      List<FilesetOutputSymlink> symlinks =
          ImmutableList.of(filesetSymlink("bar", "foo"), filesetSymlink("foo", "/foo/bar"));

      FilesetManifest manifest =
          FilesetManifest.constructFilesetManifest(
              symlinks, PathFragment.create("out/foo"), behavior);

      assertThat(manifest.getEntries())
          .containsExactly(
              PathFragment.create("out/foo/bar"), "/foo/bar",
              PathFragment.create("out/foo/foo"), "/foo/bar");
    }

    @Test
    public void testManifestWithResolvedRelativeSymlinkWithDotSlash() throws Exception {
      List<FilesetOutputSymlink> symlinks =
          ImmutableList.of(filesetSymlink("bar", "./foo"), filesetSymlink("foo", "/foo/bar"));

      FilesetManifest manifest =
          FilesetManifest.constructFilesetManifest(
              symlinks, PathFragment.create("out/foo"), behavior);

      assertThat(manifest.getEntries())
          .containsExactly(
              PathFragment.create("out/foo/bar"), "/foo/bar",
              PathFragment.create("out/foo/foo"), "/foo/bar");
    }

    @Test
    public void testManifestWithSymlinkCycle() throws Exception {
      List<FilesetOutputSymlink> symlinks =
          ImmutableList.of(
              filesetSymlink("bar", "foo"),
              filesetSymlink("foo", "biz"),
              filesetSymlink("biz", "bar"),
              filesetSymlink("reg", "/file"));

      FilesetManifest manifest =
          FilesetManifest.constructFilesetManifest(symlinks, PathFragment.create("out"), behavior);

      assertThat(manifest.getEntries()).containsExactly(PathFragment.create("out/reg"), "/file");
    }

    @Test
    public void testUnboundedSymlinkDescendant() throws Exception {
      List<FilesetOutputSymlink> symlinks =
          ImmutableList.of(
              filesetSymlink("p", "a/b"),
              filesetSymlink("a/b", "../b/c"),
              filesetSymlink("reg", "/file"));

      FilesetManifest manifest =
          FilesetManifest.constructFilesetManifest(symlinks, PathFragment.create("out"), behavior);

      assertThat(manifest.getEntries()).containsExactly(PathFragment.create("out/reg"), "/file");
    }

    @Test
    public void testUnboundedSymlinkAncestor() throws Exception {
      List<FilesetOutputSymlink> symlinks =
          ImmutableList.of(
              filesetSymlink("a/b", "c/d"),
              filesetSymlink("a/c/d", ".././a"),
              filesetSymlink("reg", "/file"));

      FilesetManifest manifest =
          FilesetManifest.constructFilesetManifest(symlinks, PathFragment.create("out"), behavior);

      assertThat(manifest.getEntries()).containsExactly(PathFragment.create("out/reg"), "/file");
    }

    @Test
    public void testManifestWithResolvedRelativeSymlinkWithDotDotSlash() throws Exception {
      List<FilesetOutputSymlink> symlinks =
          ImmutableList.of(
              filesetSymlink("bar/bar", "../foo/foo"), filesetSymlink("foo/foo", "/foo/bar"));

      FilesetManifest manifest =
          FilesetManifest.constructFilesetManifest(
              symlinks, PathFragment.create("out/foo"), behavior);

      assertThat(manifest.getEntries())
          .containsExactly(
              PathFragment.create("out/foo/bar/bar"), "/foo/bar",
              PathFragment.create("out/foo/foo/foo"), "/foo/bar");
    }

    @Test
    public void testManifestWithUnresolvableRelativeSymlink() throws Exception {
      List<FilesetOutputSymlink> symlinks = ImmutableList.of(filesetSymlink("bar", "foo"));

      FilesetManifest manifest =
          FilesetManifest.constructFilesetManifest(
              symlinks, PathFragment.create("out/foo"), behavior);

      assertThat(manifest.getEntries()).isEmpty();
      assertThat(manifest.getArtifactValues()).isEmpty();
    }

    @Test
    public void testManifestWithUnresolvableRelativeSymlinkToRelativeSymlink() throws Exception {
      List<FilesetOutputSymlink> symlinks =
          ImmutableList.of(filesetSymlink("bar", "foo"), filesetSymlink("foo", "baz"));

      FilesetManifest manifest =
          FilesetManifest.constructFilesetManifest(
              symlinks, PathFragment.create("out/foo"), behavior);

      assertThat(manifest.getEntries()).isEmpty();
      assertThat(manifest.getArtifactValues()).isEmpty();
    }
  }
}
