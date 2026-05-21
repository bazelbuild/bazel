package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DirectoryTreeDigestValue}. */
@RunWith(JUnit4.class)
public class DirectoryTreeDigestValueTest {

  private Path root;

  @Before
  public final void initializeFileSystem() throws Exception {
    FileSystem filesystem =
        new InMemoryFileSystem(BlazeClock.instance(), DigestHashFunction.SHA256);
    root = filesystem.getPath("/");
  }

  @Test
  public void keyBasicExcludes() {
    Path pkg = root.getRelative("pkg");
    RootedPath rootedPath =
        RootedPath.toRootedPath(Root.fromPath(pkg), PathFragment.create("foo/bar"));
    DirectoryTreeDigestValue.Key key =
        DirectoryTreeDigestValue.key(
            rootedPath, rootedPath, ImmutableList.of("ignoredFile", "**/*.tmp"));

    assertThat(key.excludes("foo/bar/ignoredFile", null)).isTrue();
    assertThat(key.excludes("foo/bar/anything.ending.in.tmp", null)).isTrue();
    assertThat(key.excludes("foo/bar/anything/ending/in/file.tmp", null)).isTrue();
    assertThat(key.excludes("foo/bar/notIgnored", null)).isFalse();
  }

  @Test
  public void keyDifferentRoots() {
    Path pkg1 = root.getRelative("pkg");
    RootedPath rootedPath =
        RootedPath.toRootedPath(Root.fromPath(pkg1), PathFragment.create("foo/bar"));

    Path pkg2 = root.getRelative("pkg2");
    RootedPath differentRoot =
        RootedPath.toRootedPath(Root.fromPath(pkg2), PathFragment.create("foo/bar/ignoredFile"));

    DirectoryTreeDigestValue.Key key =
        DirectoryTreeDigestValue.key(rootedPath, rootedPath, ImmutableList.of("ignoredFile"));

    assertThat(key.excludes(differentRoot, null)).isFalse();
  }

  @Test
  public void keySameRoots() {
    Path pkg = root.getRelative("pkg");
    RootedPath rootedPath =
        RootedPath.toRootedPath(Root.fromPath(pkg), PathFragment.create("foo/bar"));
    RootedPath sameRootIgnoredFile =
        RootedPath.toRootedPath(Root.fromPath(pkg), PathFragment.create("foo/bar/ignoredFile"));

    DirectoryTreeDigestValue.Key key =
        DirectoryTreeDigestValue.key(rootedPath, rootedPath, ImmutableList.of("ignoredFile"));
    assertThat(key.excludes(sameRootIgnoredFile, null)).isTrue();
  }

  @Test
  public void keyEmptyExcludes() {
    Path pkg = root.getRelative("pkg");
    RootedPath rootedPath =
        RootedPath.toRootedPath(Root.fromPath(pkg), PathFragment.create("foo/bar"));

    DirectoryTreeDigestValue.Key key =
        DirectoryTreeDigestValue.key(rootedPath, rootedPath, ImmutableList.of());
    assertThat(key.excludes("/pkg/foo/bar", null)).isFalse();
  }
}
