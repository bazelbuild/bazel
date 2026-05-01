package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import java.util.List;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import org.junit.Before;
import org.junit.Test;

/**
 * Tests for {@link FilteredRootedPath}.
 */
@RunWith(JUnit4.class)
public class FilteredRootedPathTest {
  private FileSystem filesystem;
  private Path root;

  @Before
  public final void initializeFileSystem() throws Exception  {
    filesystem = new InMemoryFileSystem(BlazeClock.instance(), DigestHashFunction.SHA256);
    root = filesystem.getPath("/");
  }

  @Test
  public void basicExcludes() {
    Path pkg = root.getRelative("pkg");
    RootedPath rootedPath =
        RootedPath.toRootedPath(Root.fromPath(pkg), PathFragment.create("foo/bar"));
    FilteredRootedPath p =
        new FilteredRootedPath(rootedPath, rootedPath, List.of("ignoredFile", "**/*.tmp"));

    assertThat(p.excludes("foo/bar/ignoredFile", null)).isTrue();
    assertThat(p.excludes("foo/bar/anything.ending.in.tmp", null)).isTrue();
    assertThat(p.excludes("foo/bar/anything/ending/in/file.tmp", null)).isTrue();
    assertThat(p.excludes("foo/bar/notIgnored", null)).isFalse();
  }

  @Test
  public void differentRoots() {
    Path pkg1 = root.getRelative("pkg");
    RootedPath rootedPath =
        RootedPath.toRootedPath(Root.fromPath(pkg1), PathFragment.create("foo/bar"));

    Path pkg2 = root.getRelative("pkg2");
    RootedPath differentRoot =
        RootedPath.toRootedPath(Root.fromPath(pkg2), PathFragment.create("foo/bar/ignoredFile"));

    FilteredRootedPath p = new FilteredRootedPath(rootedPath, rootedPath, List.of("ignoredFile"));

    assertThat(p.excludes(differentRoot, null)).isFalse();
  }

  @Test
  public void sameRoots() {
    Path pkg = root.getRelative("pkg");
    RootedPath rootedPath =
        RootedPath.toRootedPath(Root.fromPath(pkg), PathFragment.create("foo/bar"));
    RootedPath sameRootIgnoredFile =
        RootedPath.toRootedPath(Root.fromPath(pkg), PathFragment.create("foo/bar/ignoredFile"));

    FilteredRootedPath p = new FilteredRootedPath(rootedPath, rootedPath, List.of("ignoredFile"));
    assertThat(p.excludes(sameRootIgnoredFile, null)).isTrue();
  }

  @Test
  public void nullOrEmptyExcludes() {
    Path pkg = root.getRelative("pkg");
    RootedPath rootedPath =
        RootedPath.toRootedPath(Root.fromPath(pkg), PathFragment.create("foo/bar"));

    FilteredRootedPath pNull = new FilteredRootedPath(rootedPath, rootedPath, null);
    assertThat(pNull.excludes("/pkg/foo/bar", null)).isFalse();

    FilteredRootedPath pEmpty = new FilteredRootedPath(rootedPath, rootedPath, List.of());
    assertThat(pEmpty.excludes("/pkg/foo/bar", null)).isFalse();
  }
}
