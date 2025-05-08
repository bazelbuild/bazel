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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.ThreadStateReceiver;
import com.google.devtools.build.lib.cmdline.IgnoredSubdirectories;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.Globber.BadGlobException;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SyscallCache;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link GlobCache}
 */
@RunWith(JUnit4.class)
public class GlobCacheTest {

  private static final List<String> NONE = Collections.emptyList();

  private Scratch scratch = new Scratch("/workspace");

  private Path packageDirectory;
  private Path buildFile;
  private ExecutorService cacheThreadPool;
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

    createCache();
  }

  @After
  public void shutDownThreadPoolIfExists() {
    if (cacheThreadPool != null) {
      cacheThreadPool.shutdownNow();
    }
  }

  private void createCache(PathFragment... ignoredDirectories) {
    shutDownThreadPoolIfExists();
    cacheThreadPool = Executors.newFixedThreadPool(10);
    cache =
        new GlobCache(
            packageDirectory,
            PackageIdentifier.createInMainRepo("isolated"),
            IgnoredSubdirectories.of(ImmutableSet.copyOf(ignoredDirectories)),
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

              @Override
              public String getBaseNameForLoadedPackage(PackageIdentifier packageName) {
                Path buildFileForPackage = getBuildFileForPackage(packageName);
                return buildFileForPackage == null ? null : buildFileForPackage.getBaseName();
              }
            },
            SyscallCache.NO_CACHE,
            cacheThreadPool,
            -1,
            ThreadStateReceiver.NULL_INSTANCE);
  }

  @After
  public final void deleteFiles() throws Exception  {
    scratch.getFileSystem().getPath("/").deleteTreesBelow();
  }

  @Test
  public void testIgnoredDirectory() throws Exception {
    createCache(PathFragment.create("isolated/foo"));
    List<Path> paths = cache.safeGlobUnsorted("**/*.js", Globber.Operation.FILES).get();
    assertPathsAre(
        paths,
        "/workspace/isolated/first.js",
        "/workspace/isolated/second.js",
        "/workspace/isolated/bar/first.js",
        "/workspace/isolated/bar/second.js");
  }

  @Test
  public void testSafeGlob() throws Exception {
    List<Path> paths = cache.safeGlobUnsorted("*.js", Globber.Operation.FILES_AND_DIRS).get();
    assertPathsAre(paths,
        "/workspace/isolated/first.js", "/workspace/isolated/second.js");
  }

  @Test
  public void testSafeGlobInvalidPattern() throws Exception {
    String invalidPattern = "Foo?.txt";
    assertThrows(
        BadGlobException.class,
        () -> cache.safeGlobUnsorted(invalidPattern, Globber.Operation.FILES_AND_DIRS).get());
  }

  @Test
  public void testGetGlob() throws Exception {
    List<String> glob = cache.getGlobUnsorted("*.js");
    assertThat(glob).containsExactly("first.js", "second.js");
  }

  @Test
  public void testGetGlob_subdirectory() throws Exception {
    List<String> glob = cache.getGlobUnsorted("foo/*.js");
    assertThat(glob).containsExactly("foo/first.js", "foo/second.js");
  }

  @Test
  public void testGetKeySet() throws Exception {
    assertThat(cache.getKeySet()).isEmpty();

    cache.getGlobUnsorted("*.java");
    assertThat(cache.getKeySet())
        .containsExactly(Pair.of("*.java", Globber.Operation.FILES_AND_DIRS));

    cache.getGlobUnsorted("*.java");
    assertThat(cache.getKeySet())
        .containsExactly(Pair.of("*.java", Globber.Operation.FILES_AND_DIRS));

    cache.getGlobUnsorted("*.js");
    assertThat(cache.getKeySet())
        .containsExactly(
            Pair.of("*.java", Globber.Operation.FILES_AND_DIRS),
            Pair.of("*.js", Globber.Operation.FILES_AND_DIRS));

    cache.getGlobUnsorted("*.java", Globber.Operation.FILES);
    assertThat(cache.getKeySet())
        .containsExactly(
            Pair.of("*.java", Globber.Operation.FILES_AND_DIRS),
            Pair.of("*.js", Globber.Operation.FILES_AND_DIRS),
            Pair.of("*.java", Globber.Operation.FILES));

    assertThrows(BadGlobException.class, () -> cache.getGlobUnsorted("invalid?"));
    assertThat(cache.getKeySet())
        .containsExactly(
            Pair.of("*.java", Globber.Operation.FILES_AND_DIRS),
            Pair.of("*.js", Globber.Operation.FILES_AND_DIRS),
            Pair.of("*.java", Globber.Operation.FILES));

    cache.getGlobUnsorted("foo/first.*");
    assertThat(cache.getKeySet())
        .containsExactly(
            Pair.of("*.java", Globber.Operation.FILES_AND_DIRS),
            Pair.of("*.java", Globber.Operation.FILES),
            Pair.of("*.js", Globber.Operation.FILES_AND_DIRS),
            Pair.of("foo/first.*", Globber.Operation.FILES_AND_DIRS));
  }

  @Test
  public void testGlob() throws Exception {
    assertEmpty(cache.globUnsorted(list("*.java"), NONE, Globber.Operation.FILES, true));

    assertThat(cache.globUnsorted(list("*.*"), NONE, Globber.Operation.FILES, true))
        .containsExactly("first.js", "first.txt", "second.js", "second.txt");

    assertThat(cache.globUnsorted(list("*.*"), list("first.js"), Globber.Operation.FILES, true))
        .containsExactly("first.txt", "second.js", "second.txt");

    assertThat(cache.globUnsorted(list("*.txt", "first.*"), NONE, Globber.Operation.FILES, true))
        .containsExactly("first.txt", "second.txt", "first.js");
  }

  @Test
  public void testRecursiveGlobDoesNotMatchSubpackage() throws Exception {
    List<String> glob = cache.getGlobUnsorted("**/*.js");
    assertThat(glob).containsExactly("first.js", "second.js", "foo/first.js", "bar/first.js",
        "foo/second.js", "bar/second.js");
  }

  @Test
  public void testSingleFileExclude_star() throws Exception {
    assertThat(
            cache.globUnsorted(
                list("*"), list("first.txt"), Globber.Operation.FILES_AND_DIRS, true))
        .containsExactly("BUILD", "bar", "first.js", "foo", "second.js", "second.txt");
  }

  @Test
  public void testSingleFileExclude_starStar() throws Exception {
    assertThat(
            cache.globUnsorted(
                list("**"), list("first.txt"), Globber.Operation.FILES_AND_DIRS, true))
        .containsExactly(
            "BUILD",
            "bar",
            "bar/first.js",
            "bar/second.js",
            "first.js",
            "foo",
            "foo/first.js",
            "foo/second.js",
            "second.js",
            "second.txt");
  }

  @Test
  public void testExcludeAll_star() throws Exception {
    assertThat(cache.globUnsorted(list("*"), list("*"), Globber.Operation.FILES_AND_DIRS, true))
        .isEmpty();
  }

  @Test
  public void testExcludeAll_star_noMatchesAnyway() throws Exception {
    assertThat(cache.globUnsorted(list("nope"), list("*"), Globber.Operation.FILES_AND_DIRS, true))
        .isEmpty();
  }

  @Test
  public void testExcludeAll_starStar() throws Exception {
    assertThat(cache.globUnsorted(list("**"), list("**"), Globber.Operation.FILES_AND_DIRS, true))
        .isEmpty();
  }

  @Test
  public void testExcludeAll_manual() throws Exception {
    assertThat(
            cache.globUnsorted(
                list("**"), list("*", "*/*", "*/*/*"), Globber.Operation.FILES_AND_DIRS, true))
        .isEmpty();
  }

  @Test
  public void testSingleFileExcludeDoesntMatch() throws Exception {
    assertThat(
            cache.globUnsorted(
                list("first.txt"), list("nope.txt"), Globber.Operation.FILES_AND_DIRS, true))
        .containsExactly("first.txt");
  }

  @Test
  public void testExcludeDirectory() throws Exception {
    assertThat(cache.globUnsorted(list("foo/*"), NONE, Globber.Operation.FILES, true))
        .containsExactly("foo/first.js", "foo/second.js");
    assertThat(
            cache.globUnsorted(list("foo/*"), list("foo"), Globber.Operation.FILES_AND_DIRS, true))
        .containsExactly("foo/first.js", "foo/second.js");
  }

  @Test
  public void testChildGlobWithChildExclude() throws Exception {
    assertThat(
            cache.globUnsorted(
                list("foo/*"), list("foo/*"), Globber.Operation.FILES_AND_DIRS, true))
        .isEmpty();
    assertThat(
            cache.globUnsorted(
                list("foo/first.js", "foo/second.js"),
                list("foo/*"),
                Globber.Operation.FILES_AND_DIRS,
                true))
        .isEmpty();
    assertThat(
            cache.globUnsorted(
                list("foo/first.js"), list("foo/first.js"), Globber.Operation.FILES_AND_DIRS, true))
        .isEmpty();
    assertThat(
            cache.globUnsorted(
                list("foo/first.js"), list("*/first.js"), Globber.Operation.FILES_AND_DIRS, true))
        .isEmpty();
    assertThat(
            cache.globUnsorted(
                list("foo/first.js"), list("*/*"), Globber.Operation.FILES_AND_DIRS, true))
        .isEmpty();
  }

  @Test
  public void testSubpackages_noWildcard() throws Exception {
    assertThat(cache.globUnsorted(list("sub/sub.js"), list(), Globber.Operation.SUBPACKAGES, true))
        .isEmpty();
  }

  @Test
  public void testSubpackages_simpleDoubleStar() throws Exception {
    assertThat(cache.globUnsorted(list("**"), list(), Globber.Operation.SUBPACKAGES, true))
        .containsExactly("sub");
  }

  @Test
  public void testSubpackages_onlySub() throws Exception {
    assertThat(cache.globUnsorted(list("sub"), list(), Globber.Operation.SUBPACKAGES, true))
        .containsExactly("sub");
  }

  @Test
  public void testSubpackages_singleStarsAfterSub() throws Exception {
    assertThat(cache.globUnsorted(list("sub/*"), list(), Globber.Operation.SUBPACKAGES, true))
        .isEmpty();
  }

  @Test
  public void testSubpackages_doubleStarsAfterSub() throws Exception {
    assertThat(cache.globUnsorted(list("sub/**"), list(), Globber.Operation.SUBPACKAGES, true))
        .containsExactly("sub");
  }

  @Test
  public void testSubpackages_twoDoubleStarsAfterSub() throws Exception {
    // Both `**`s are considered to match no path fragments.
    assertThat(cache.globUnsorted(list("sub/**/**"), list(), Globber.Operation.SUBPACKAGES, true))
        .containsExactly("sub");
  }

  @Test
  public void testSubpackages_doubleStarsAndOtherPathAfterSub() throws Exception {
    assertThat(cache.globUnsorted(list("sub/**/foo"), list(), Globber.Operation.SUBPACKAGES, true))
        .isEmpty();
  }

  @Test
  public void testSubpackages_doubleStarWithTrailingPattern() throws Exception {
    assertThat(cache.globUnsorted(list("**/bar"), list(), Globber.Operation.SUBPACKAGES, true))
        .isEmpty();
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
