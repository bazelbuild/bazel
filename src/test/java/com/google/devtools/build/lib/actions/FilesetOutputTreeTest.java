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
import static com.google.devtools.build.lib.actions.FilesetOutputTree.RelativeSymlinkBehavior.IGNORE;
import static com.google.devtools.build.lib.actions.FilesetOutputTree.RelativeSymlinkBehavior.RESOLVE_FULLY;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.FilesetOutputTree.FilesetManifest;
import com.google.devtools.build.lib.actions.FilesetOutputTree.ForbiddenRelativeSymlinkException;
import com.google.devtools.build.lib.actions.FilesetOutputTree.RelativeSymlinkBehavior;
import com.google.devtools.build.lib.actions.FilesetOutputTreeTest.ManifestCommonTests;
import com.google.devtools.build.lib.actions.FilesetOutputTreeTest.OneOffManifestTests;
import com.google.devtools.build.lib.actions.FilesetOutputTreeTest.ResolvingManifestTests;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.runners.Suite;

/** Tests for {@link FilesetOutputTree}. */
@RunWith(Suite.class)
@Suite.SuiteClasses({
  ManifestCommonTests.class,
  OneOffManifestTests.class,
  ResolvingManifestTests.class
})
public final class FilesetOutputTreeTest {

  private static final PathFragment EXEC_ROOT = PathFragment.create("/root");

  private static FilesetOutputSymlink symlink(String from, String to) {
    return FilesetOutputSymlink.createForTesting(
        PathFragment.create(from), PathFragment.create(to), EXEC_ROOT);
  }

  private static FilesetOutputTree createTree(FilesetOutputSymlink... symlinks) {
    return FilesetOutputTree.create(ImmutableList.copyOf(symlinks));
  }

  /** Manifest tests that apply to all relative symlink behavior. */
  @RunWith(TestParameterInjector.class)
  public static final class ManifestCommonTests {

    @TestParameter private RelativeSymlinkBehavior behavior;

    @Test
    public void emptyManifest() throws Exception {
      FilesetManifest manifest =
          FilesetOutputTree.EMPTY.constructFilesetManifest(
              PathFragment.create("out/foo"), behavior);

      assertThat(manifest.getEntries()).isEmpty();
      assertThat(manifest.getArtifactValues()).isEmpty();
    }

    @Test
    public void manifestWithSingleFile() throws Exception {
      FilesetOutputTree filesetOutput = createTree(symlink("bar", "/dir/file"));

      FilesetManifest manifest =
          filesetOutput.constructFilesetManifest(PathFragment.create("out/foo"), behavior);

      assertThat(manifest.getEntries())
          .containsExactly(PathFragment.create("out/foo/bar"), "/dir/file");
    }

    @Test
    public void manifestWithTwoFiles() throws Exception {
      FilesetOutputTree filesetOutput =
          createTree(symlink("bar", "/dir/file"), symlink("baz", "/dir/file"));

      FilesetManifest manifest =
          filesetOutput.constructFilesetManifest(PathFragment.create("out/foo"), behavior);

      assertThat(manifest.getEntries())
          .containsExactly(
              PathFragment.create("out/foo/bar"), "/dir/file",
              PathFragment.create("out/foo/baz"), "/dir/file");
    }

    @Test
    public void manifestWithDirectory() throws Exception {
      FilesetOutputTree filesetOutput = createTree(symlink("bar", "/some"));

      FilesetManifest manifest =
          filesetOutput.constructFilesetManifest(PathFragment.create("out/foo"), behavior);

      assertThat(manifest.getEntries())
          .containsExactly(PathFragment.create("out/foo/bar"), "/some");
    }

    @Test
    public void manifestWithExecRootRelativePath() throws Exception {
      FilesetOutputTree filesetOutput =
          createTree(symlink("bar", EXEC_ROOT.getRelative("foo/bar").getPathString()));

      FilesetManifest manifest =
          filesetOutput.constructFilesetManifest(PathFragment.create("out/foo"), behavior);

      assertThat(manifest.getEntries())
          .containsExactly(PathFragment.create("out/foo/bar"), "foo/bar");
    }
  }

  /** Manifest tests that apply to a specific relative symlink behavior. */
  @RunWith(JUnit4.class)
  public static final class OneOffManifestTests {

    @Test
    public void canonicalEmptyInstance() {
      assertThat(createTree()).isSameInstanceAs(FilesetOutputTree.EMPTY);
    }

    @Test
    public void manifestWithErrorOnRelativeSymlink() {
      FilesetOutputTree filesetOutput =
          createTree(symlink("bar", "foo"), symlink("foo", "/foo/bar"));

      var e =
          assertThrows(
              ForbiddenRelativeSymlinkException.class,
              () -> filesetOutput.constructFilesetManifest(PathFragment.create("out/foo"), ERROR));
      assertThat(e).hasMessageThat().contains("Fileset symlink foo is not absolute");
    }

    @Test
    public void manifestWithIgnoredRelativeSymlink() throws Exception {
      FilesetOutputTree filesetOutput =
          createTree(symlink("bar", "foo"), symlink("foo", "/foo/bar"));

      FilesetManifest manifest =
          filesetOutput.constructFilesetManifest(PathFragment.create("out/foo"), IGNORE);

      assertThat(manifest.getEntries())
          .containsExactly(PathFragment.create("out/foo/foo"), "/foo/bar");
    }

    @Test
    public void manifestWithResolvedRelativeDirectorySymlink() throws Exception {
      FilesetOutputTree filesetOutput =
          createTree(symlink("foo/subdir/f1", "/foo/subdir/f1"), symlink("foo/bar", "subdir"));

      FilesetManifest manifest =
          filesetOutput.constructFilesetManifest(PathFragment.create("out"), RESOLVE_FULLY);

      assertThat(manifest.getEntries())
          .containsExactly(
              PathFragment.create("out/foo/subdir/f1"), "/foo/subdir/f1",
              PathFragment.create("out/foo/bar/f1"), "/foo/subdir/f1");
    }
  }

  /** Manifest tests that apply resolving relative symlink behavior. */
  @RunWith(TestParameterInjector.class)
  public static final class ResolvingManifestTests {

    @TestParameter({"RESOLVE", "RESOLVE_FULLY"})
    private RelativeSymlinkBehavior behavior;

    @Test
    public void manifestWithResolvedRelativeSymlink() throws Exception {
      FilesetOutputTree filesetOutput =
          createTree(symlink("bar", "foo"), symlink("foo", "/foo/bar"));

      FilesetManifest manifest =
          filesetOutput.constructFilesetManifest(PathFragment.create("out/foo"), behavior);

      assertThat(manifest.getEntries())
          .containsExactly(
              PathFragment.create("out/foo/bar"), "/foo/bar",
              PathFragment.create("out/foo/foo"), "/foo/bar");
    }

    @Test
    public void manifestWithResolvedRelativeSymlinkWithDotSlash() throws Exception {
      FilesetOutputTree filesetOutput =
          createTree(symlink("bar", "./foo"), symlink("foo", "/foo/bar"));

      FilesetManifest manifest =
          filesetOutput.constructFilesetManifest(PathFragment.create("out/foo"), behavior);

      assertThat(manifest.getEntries())
          .containsExactly(
              PathFragment.create("out/foo/bar"), "/foo/bar",
              PathFragment.create("out/foo/foo"), "/foo/bar");
    }

    @Test
    public void manifestWithSymlinkCycle() throws Exception {
      FilesetOutputTree filesetOutput =
          createTree(
              symlink("bar", "foo"),
              symlink("foo", "biz"),
              symlink("biz", "bar"),
              symlink("reg", "/file"));

      FilesetManifest manifest =
          filesetOutput.constructFilesetManifest(PathFragment.create("out"), behavior);

      assertThat(manifest.getEntries()).containsExactly(PathFragment.create("out/reg"), "/file");
    }

    @Test
    public void unboundedSymlinkDescendant() throws Exception {
      FilesetOutputTree filesetOutput =
          createTree(symlink("p", "a/b"), symlink("a/b", "../b/c"), symlink("reg", "/file"));

      FilesetManifest manifest =
          filesetOutput.constructFilesetManifest(PathFragment.create("out"), behavior);

      assertThat(manifest.getEntries()).containsExactly(PathFragment.create("out/reg"), "/file");
    }

    @Test
    public void unboundedSymlinkAncestor() throws Exception {
      FilesetOutputTree filesetOutput =
          createTree(symlink("a/b", "c/d"), symlink("a/c/d", ".././a"), symlink("reg", "/file"));

      FilesetManifest manifest =
          filesetOutput.constructFilesetManifest(PathFragment.create("out"), behavior);

      assertThat(manifest.getEntries()).containsExactly(PathFragment.create("out/reg"), "/file");
    }

    @Test
    public void manifestWithResolvedRelativeSymlinkWithDotDotSlash() throws Exception {
      FilesetOutputTree filesetOutput =
          createTree(symlink("bar/bar", "../foo/foo"), symlink("foo/foo", "/foo/bar"));

      FilesetManifest manifest =
          filesetOutput.constructFilesetManifest(PathFragment.create("out/foo"), behavior);

      assertThat(manifest.getEntries())
          .containsExactly(
              PathFragment.create("out/foo/bar/bar"), "/foo/bar",
              PathFragment.create("out/foo/foo/foo"), "/foo/bar");
    }

    @Test
    public void manifestWithUnresolvableRelativeSymlink() throws Exception {
      FilesetOutputTree filesetOutput = createTree(symlink("bar", "foo"));

      FilesetManifest manifest =
          filesetOutput.constructFilesetManifest(PathFragment.create("out/foo"), behavior);

      assertThat(manifest.getEntries()).isEmpty();
      assertThat(manifest.getArtifactValues()).isEmpty();
    }

    @Test
    public void manifestWithUnresolvableRelativeSymlinkToRelativeSymlink() throws Exception {
      FilesetOutputTree filesetOutput = createTree(symlink("bar", "foo"), symlink("foo", "baz"));

      FilesetManifest manifest =
          filesetOutput.constructFilesetManifest(PathFragment.create("out/foo"), behavior);

      assertThat(manifest.getEntries()).isEmpty();
      assertThat(manifest.getArtifactValues()).isEmpty();
    }
  }
}
