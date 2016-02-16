// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.Lists;
import com.google.common.util.concurrent.Futures;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.Globber.BadGlobException;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

/**
 * Tests for {@link GlobCache}
 */
@RunWith(JUnit4.class)
public class GlobCacheTest {

  private static final List<String> NONE = Collections.emptyList();

  private Scratch scratch = new Scratch("/workspace");

  private Path packageDirectory;
  private Path buildFile;
  private GlobCache cache;

  @Before
  public final void createFiles() throws Exception  {
    buildFile = scratch.file("isolated/BUILD",
        "# contents don't matter in this test");
    scratch.file("isolated/sub/BUILD",
        "# contents don't matter in this test");

    packageDirectory = buildFile.getParentDirectory();

    scratch.file("isolated/first.txt",
        "# this is first.txt");

    scratch.file("isolated/second.txt",
        "# this is second.txt");

    scratch.file("isolated/first.js",
        "# this is first.js");

    scratch.file("isolated/second.js",
        "# this is second.js");

    // Files in subdirectories

    scratch.file("isolated/foo/first.js",
        "# this is foo/first.js");

    scratch.file("isolated/foo/second.js",
        "# this is foo/second.js");

    scratch.file("isolated/bar/first.js",
        "# this is bar/first.js");

    scratch.file("isolated/bar/second.js",
        "# this is bar/second.js");

    scratch.file("isolated/sub/sub.js",
        "# this is sub/sub.js");

    cache = new GlobCache(packageDirectory, PackageIdentifier.createInDefaultRepo("isolated"),
        new CachingPackageLocator() {
      @Override
      public Path getBuildFileForPackage(PackageIdentifier packageId) {
        String packageName = packageId.getPackageFragment().getPathString();
        if (packageName.equals("isolated")) {
          return scratch.resolve("isolated/BUILD");
        } else if (packageName.equals("isolated/sub")) {
          return scratch.resolve("isolated/sub/BUILD");
        } else {
          return null;
        }
      }
    }, null, TestUtils.getPool());
  }

  @After
  public final void deleteFiles() throws Exception  {
    FileSystemUtils.deleteTreesBelow(scratch.getFileSystem().getRootDirectory());
  }

  @Test
  public void testSafeGlob() throws Exception {
    List<Path> paths = cache.safeGlob("*.js", false).get();
    assertPathsAre(paths,
        "/workspace/isolated/first.js", "/workspace/isolated/second.js");
  }

  @Test
  public void testSafeGlobInvalidPatterns() throws Exception {
    for (String pattern : new String[] {
        "Foo?.txt", "List{Test}.py", "List(Test).py" }) {
      try {
        cache.safeGlob(pattern, false);
        fail("Expected pattern " + pattern + " to fail");
      } catch (BadGlobException expected) {
      }
    }
  }

  @Test
  public void testGetGlob() throws Exception {
    List<String> glob = cache.getGlob("*.js");
    assertThat(glob).containsExactly("first.js", "second.js");
  }

  @Test
  public void testGetGlob_subdirectory() throws Exception {
    List<String> glob = cache.getGlob("foo/*.js");
    assertThat(glob).containsExactly("foo/first.js", "foo/second.js");
  }

  @Test
  public void testGetKeySet() throws Exception {
    assertThat(cache.getKeySet()).isEmpty();

    cache.getGlob("*.java");
    assertThat(cache.getKeySet()).containsExactly(Pair.of("*.java", false));

    cache.getGlob("*.java");
    assertThat(cache.getKeySet()).containsExactly(Pair.of("*.java", false));

    cache.getGlob("*.js");
    assertThat(cache.getKeySet()).containsExactly(Pair.of("*.java", false), Pair.of("*.js", false));

    cache.getGlob("*.java", true);
    assertThat(cache.getKeySet()).containsExactly(Pair.of("*.java", false), Pair.of("*.js", false),
        Pair.of("*.java", true));

    try {
      cache.getGlob("invalid?");
      fail("Expected an invalid regex exception");
    } catch (BadGlobException expected) {
    }
    assertThat(cache.getKeySet()).containsExactly(Pair.of("*.java", false), Pair.of("*.js", false),
        Pair.of("*.java", true));

    cache.getGlob("foo/first.*");
    assertThat(cache.getKeySet()).containsExactly(Pair.of("*.java", false), Pair.of("*.java", true),
        Pair.of("*.js", false), Pair.of("foo/first.*", false));
  }

  @Test
  public void testGlob() throws Exception {
    assertEmpty(cache.glob(list("*.java"), NONE, false));

    assertThat(cache.glob(list("*.*"), NONE, false)).containsExactly("first.js", "first.txt",
        "second.js", "second.txt").inOrder();

    assertThat(cache.glob(list("*.*"), list("first.js"), false)).containsExactly("first.txt",
        "second.js", "second.txt").inOrder();

    assertThat(cache.glob(list("*.txt", "first.*"), NONE, false)).containsExactly("first.txt",
        "second.txt", "first.js").inOrder();
  }

  @Test
  public void testSetGlobPaths() throws Exception {
    // This pattern matches no files.
    String pattern = "fake*.java";
    assertThat(cache.getKeySet()).doesNotContain(pattern);

    List<String> results = cache.getGlob(pattern, false);

    assertThat(cache.getKeySet()).contains(Pair.of(pattern, false));
    assertThat(results).isEmpty();

    cache.setGlobPaths(pattern, false, Futures.<List<Path>>immediateFuture(Lists.newArrayList(
        scratch.resolve("isolated/fake.txt"),
        scratch.resolve("isolated/fake.py"))));

    assertThat(cache.getGlob(pattern, false)).containsExactly("fake.py", "fake.txt");
  }

  @Test
  public void testGlobsUpToDate() throws Exception {
    assertTrue(cache.globsUpToDate());

    // Initialize the cache
    cache.getGlob("*.txt");
    assertTrue(cache.globsUpToDate());

    cache.getGlob("*.js");
    assertTrue(cache.globsUpToDate());

    // Change the filesystem
    scratch.file("isolated/third.txt",
        "# this is third.txt");
    assertFalse(cache.globsUpToDate());

    // Fool the cache to observe the method's behavior.
    cache.setGlobPaths("*.txt", false, Futures.<List<Path>>immediateFuture(Lists.newArrayList(
        scratch.resolve("isolated/first.txt"),
        scratch.resolve("isolated/second.txt"),
        scratch.resolve("isolated/third.txt"))));
    assertTrue(cache.globsUpToDate());
  }

  @Test
  public void testRecursiveGlobDoesNotMatchSubpackage() throws Exception {
    List<String> glob = cache.getGlob("**/*.js");
    assertThat(glob).containsExactly("first.js", "second.js", "foo/first.js", "bar/first.js",
        "foo/second.js", "bar/second.js");
  }

  private void assertEmpty(Collection<?> glob) {
    assertThat(glob).isEmpty();
  }

  private void assertPathsAre(List<Path> paths, String... strings) {
    List<String> pathStrings = new ArrayList<>();
    for (Path path : paths) {
      pathStrings.add(path.getPathString());
    }
    assertThat(pathStrings).containsExactlyElementsIn(Arrays.asList(strings));
  }

  /* syntactic shorthand for Lists.newArrayList(strings) */
  private List<String> list(String... strings) {
    return Lists.newArrayList(strings);
  }
}
