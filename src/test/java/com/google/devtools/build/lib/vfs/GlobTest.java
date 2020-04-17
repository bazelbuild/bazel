// Copyright 2014 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link UnixGlob}
 */
@RunWith(JUnit4.class)
public class GlobTest {

  private Path tmpPath;
  private FileSystem fs;
  private Path throwOnReaddir = null;
  private Path throwOnStat = null;

  @Before
  public final void initializeFileSystem() throws Exception  {
    fs = new InMemoryFileSystem() {
      @Override
      public Collection<Dirent> readdir(Path path, boolean followSymlinks) throws IOException {
        if (path.equals(throwOnReaddir)) {
          throw new FileNotFoundException(path.getPathString());
        }
        return super.readdir(path, followSymlinks);
      }

      @Override
      public FileStatus statIfFound(Path path, boolean followSymlinks) throws IOException {
        if (path.equals(throwOnStat)) {
          throw new FileNotFoundException(path.getPathString());
        }
        return super.statIfFound(path, followSymlinks);
      }
    };
    tmpPath = fs.getPath("/globtmp");
    for (String dir : ImmutableList.of("foo/bar/wiz",
                         "foo/barnacle/wiz",
                         "food/barnacle/wiz",
                         "fool/barnacle/wiz")) {
      FileSystemUtils.createDirectoryAndParents(tmpPath.getRelative(dir));
    }
    FileSystemUtils.createEmptyFile(tmpPath.getRelative("foo/bar/wiz/file"));
  }

  @After
  public void resetInteruppt() {
    Thread.interrupted();
  }

  @Test
  public void testQuestionMarkMatch() throws Exception {
    assertGlobMatches("foo?", /* => */"food", "fool");
  }

  @Test
  public void testQuestionMarkNoMatch() throws Exception {
    assertGlobMatches("food/bar?" /* => nothing */);
  }

  @Test
  public void testStartsWithStar() throws Exception {
    assertGlobMatches("*oo", /* => */"foo");
  }

  @Test
  public void testStartsWithStarWithMiddleStar() throws Exception {
    assertGlobMatches("*f*o", /* => */"foo");
  }

  @Test
  public void testEndsWithStar() throws Exception {
    assertGlobMatches("foo*", /* => */"foo", "food", "fool");
  }

  @Test
  public void testEndsWithStarWithMiddleStar() throws Exception {
    assertGlobMatches("f*oo*", /* => */"foo", "food", "fool");
  }

  @Test
  public void testMiddleStar() throws Exception {
    assertGlobMatches("f*o", /* => */"foo");
  }

  @Test
  public void testTwoMiddleStars() throws Exception {
    assertGlobMatches("f*o*o", /* => */"foo");
  }

  @Test
  public void testSingleStarPatternWithNamedChild() throws Exception {
    assertGlobMatches("*/bar", /* => */"foo/bar");
  }

  @Test
  public void testSingleStarPatternWithChildGlob() throws Exception {
    assertGlobMatches("*/bar*", /* => */
        "foo/bar", "foo/barnacle", "food/barnacle", "fool/barnacle");
  }

  @Test
  public void testSingleStarAsChildGlob() throws Exception {
    assertGlobMatches("foo/*/wiz", /* => */"foo/bar/wiz", "foo/barnacle/wiz");
  }

  @Test
  public void testNoAsteriskAndFilesDontExist() throws Exception {
    // Note un-UNIX like semantics:
    assertGlobMatches("ceci/n'est/pas/une/globbe" /* => nothing */);
  }

  @Test
  public void testSingleAsteriskUnderNonexistentDirectory() throws Exception {
    // Note un-UNIX like semantics:
    assertGlobMatches("not-there/*" /* => nothing */);
  }

  @Test
  public void testGlobWithNonExistentBase() throws Exception {
    Collection<Path> globResult = UnixGlob.forPath(fs.getPath("/does/not/exist"))
        .addPattern("*.txt")
        .globInterruptible();
    assertThat(globResult).isEmpty();
  }

  @Test
  public void testGlobUnderFile() throws Exception {
    assertGlobMatches("foo/bar/wiz/file/*" /* => nothing */);
  }

  private void assertGlobMatches(String pattern, String... expecteds)
      throws Exception {
    assertGlobMatches(Collections.singleton(pattern), expecteds);
  }

  private void assertGlobMatches(Collection<String> pattern,
                                 String... expecteds)
      throws Exception {
    assertThat(
        new UnixGlob.Builder(tmpPath)
            .addPatterns(pattern)
            .globInterruptible())
    .containsExactlyElementsIn(resolvePaths(expecteds));
  }

  private Set<Path> resolvePaths(String... relativePaths) {
    Set<Path> expectedFiles = new HashSet<>();
    for (String expected : relativePaths) {
      Path file = expected.equals(".")
          ? tmpPath
          : tmpPath.getRelative(expected);
      expectedFiles.add(file);
    }
    return expectedFiles;
  }

  @Test
  public void testIOFailureOnStat() throws Exception {
    UnixGlob.FilesystemCalls syscalls =
        new UnixGlob.FilesystemCalls() {
          @Override
          public FileStatus statIfFound(Path path, Symlinks symlinks) throws IOException {
            throw new IOException("EIO");
          }

          @Override
          public Collection<Dirent> readdir(Path path) {
            throw new IllegalStateException();
          }

          @Override
          public Dirent.Type getType(Path path, Symlinks symlinks) {
            throw new IllegalStateException();
          }
        };

    IOException e =
        assertThrows(
            IOException.class,
            () ->
                new UnixGlob.Builder(tmpPath)
                    .addPattern("foo/bar/wiz/file")
                    .setFilesystemCalls(new AtomicReference<>(syscalls))
                    .glob());
    assertThat(e).hasMessageThat().isEqualTo("EIO");
  }

  @Test
  public void testGlobWithoutWildcardsDoesNotCallReaddir() throws Exception {
    UnixGlob.FilesystemCalls syscalls =
        new UnixGlob.FilesystemCalls() {
          @Override
          public FileStatus statIfFound(Path path, Symlinks symlinks) throws IOException {
            return UnixGlob.DEFAULT_SYSCALLS.statIfFound(path, symlinks);
          }

          @Override
          public Collection<Dirent> readdir(Path path) {
            throw new IllegalStateException();
          }

          @Override
          public Dirent.Type getType(Path path, Symlinks symlinks) {
            throw new IllegalStateException();
          }
        };

    assertThat(
            new UnixGlob.Builder(tmpPath)
                .addPattern("foo/bar/wiz/file")
                .setFilesystemCalls(new AtomicReference<>(syscalls))
                .glob())
        .containsExactly(tmpPath.getRelative("foo/bar/wiz/file"));
  }

  @Test
  public void testIllegalPatterns() throws Exception {
    assertIllegalPattern("foo**bar");
    assertIllegalPattern("");
    assertIllegalPattern(".");
    assertIllegalPattern("/foo");
    assertIllegalPattern("./foo");
    assertIllegalPattern("foo/");
    assertIllegalPattern("foo/./bar");
    assertIllegalPattern("../foo/bar");
    assertIllegalPattern("foo//bar");
  }

  /**
   * Tests that globs can contain Java regular expression special characters
   */
  @Test
  public void testSpecialRegexCharacter() throws Exception {
    Path tmpPath2 = fs.getPath("/globtmp2");
    FileSystemUtils.createDirectoryAndParents(tmpPath2);
    Path aDotB = tmpPath2.getChild("a.b");
    FileSystemUtils.createEmptyFile(aDotB);
    Path aPlusB = tmpPath2.getChild("a+b");
    FileSystemUtils.createEmptyFile(aPlusB);
    Path aWordCharacterB = tmpPath2.getChild("a\\wb");
    FileSystemUtils.createEmptyFile(aWordCharacterB);
    Path groupsAndBrackets = tmpPath2.getChild("(a|b){1,2}[ab]");
    FileSystemUtils.createEmptyFile(groupsAndBrackets);
    Path lineNoise = tmpPath2.getChild("\\|}{[(])(.+");
    FileSystemUtils.createEmptyFile(lineNoise);
    FileSystemUtils.createEmptyFile(tmpPath2.getChild("aab"));
    // Note: these contain two asterisks because otherwise a RE is not built,
    // as an optimization.
    assertThat(UnixGlob.forPath(tmpPath2).addPattern("*a.b*").globInterruptible())
        .containsExactly(aDotB);
    assertThat(UnixGlob.forPath(tmpPath2).addPattern("*a+b*").globInterruptible())
        .containsExactly(aPlusB);
    assertThat(UnixGlob.forPath(tmpPath2).addPattern("*a\\wb*").globInterruptible())
        .containsExactly(aWordCharacterB);
    assertThat(UnixGlob.forPath(tmpPath2).addPattern("*(a|b){1,2}[ab]*").globInterruptible())
        .containsExactly(groupsAndBrackets);
    assertThat(UnixGlob.forPath(tmpPath2).addPattern("*\\|}{[(])(.+*").globInterruptible())
        .containsExactly(lineNoise);
  }

  @Test
  public void testMatchesCallWithNoCache() {
    assertThat(UnixGlob.matches("*a*b", "CaCb", null)).isTrue();
  }

  @Test
  public void testMultiplePatterns() throws Exception {
    assertGlobMatches(Lists.newArrayList("foo", "fool"), "foo", "fool");
  }

  @Test
  public void testMatcherMethodRecursiveBelowDir() throws Exception {
    FileSystemUtils.createEmptyFile(tmpPath.getRelative("foo/file"));
    String pattern = "foo/**/*";
    assertThat(UnixGlob.matches(pattern, "foo/bar")).isTrue();
    assertThat(UnixGlob.matches(pattern, "foo/bar/baz")).isTrue();
    assertThat(UnixGlob.matches(pattern, "foo")).isFalse();
    assertThat(UnixGlob.matches(pattern, "foob")).isFalse();
    assertThat(UnixGlob.matches("**/foo", "foo")).isTrue();
  }

  @Test
  public void testMultiplePatternsWithOverlap() throws Exception {
    assertGlobMatchesAnyOrder(Lists.newArrayList("food", "foo?"),
                              "food", "fool");
    assertGlobMatchesAnyOrder(Lists.newArrayList("food", "?ood", "f??d"),
                              "food");
    assertThat(resolvePaths("food", "fool", "foo")).containsExactlyElementsIn(
        new UnixGlob.Builder(tmpPath).addPatterns("food", "xxx", "*").glob());

  }

  private void assertGlobMatchesAnyOrder(ArrayList<String> patterns,
                                         String... paths) throws Exception {
    assertThat(resolvePaths(paths)).containsExactlyElementsIn(
        new UnixGlob.Builder(tmpPath).addPatterns(patterns).globInterruptible());
  }

  private void assertIllegalPattern(String pattern) throws Exception {
    UnixGlob.BadPattern e =
        assertThrows(
            UnixGlob.BadPattern.class,
            () -> new UnixGlob.Builder(tmpPath).addPattern(pattern).globInterruptible());
    assertThat(e).hasMessageThat().containsMatch("in glob pattern");
  }

  @Test
  public void testHiddenFiles() throws Exception {
    for (String dir : ImmutableList.of(".hidden", "..also.hidden", "not.hidden")) {
      FileSystemUtils.createDirectoryAndParents(tmpPath.getRelative(dir));
    }

    // Note that these are not in the result: ".", ".."
    assertGlobMatches("*", "not.hidden", "foo", "fool", "food", ".hidden", "..also.hidden");

    assertGlobMatches("*.hidden", "not.hidden");

    assertGlobMatches(".*also*", "..also.hidden");
  }

  @Test
  public void testIOException() throws Exception {
    throwOnReaddir = fs.getPath("/throw_on_readdir");
    throwOnReaddir.createDirectory();
    assertThrows(
        IOException.class, () -> new UnixGlob.Builder(throwOnReaddir).addPattern("**").glob());
  }

  @Test
  public void testFastFailureithInterrupt() throws Exception {
    Thread.currentThread().interrupt();
    throwOnStat = tmpPath;
    FileNotFoundException e =
        assertThrows(FileNotFoundException.class, () -> new UnixGlob.Builder(tmpPath).glob());
    assertThat(e).hasMessageThat().contains("globtmp");
  }

  @Test
  public void testCheckCanBeInterrupted() throws Exception {
    final Thread mainThread = Thread.currentThread();
    final ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(10);

    Predicate<Path> interrupterPredicate =
        new Predicate<Path>() {
          @Override
          public boolean apply(Path input) {
            mainThread.interrupt();
            return true;
          }
        };

    Future<?> globResult =
        new UnixGlob.Builder(tmpPath)
            .addPattern("**")
            .setDirectoryFilter(interrupterPredicate)
            .setExecutor(executor)
            .globAsync();
    assertThrows(InterruptedException.class, () -> globResult.get());

    globResult.cancel(true);
    assertThrows(
        CancellationException.class, () -> Uninterruptibles.getUninterruptibly(globResult));

    Thread.interrupted();
    assertThat(executor.isShutdown()).isFalse();
    executor.shutdown();
    assertThat(executor.awaitTermination(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS))
        .isTrue();
  }

  @Test
  public void testCheckCannotBeInterrupted() throws Exception {
    final Thread mainThread = Thread.currentThread();
    final ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(10);
    final AtomicBoolean sentInterrupt = new AtomicBoolean(false);

    Predicate<Path> interrupterPredicate = new Predicate<Path>() {
      @Override
      public boolean apply(Path input) {
        if (!sentInterrupt.getAndSet(true)) {
          mainThread.interrupt();
        }
        return true;
      }
    };

    List<Path> result =
        new UnixGlob.Builder(tmpPath)
            .addPatterns("**", "*")
            .setDirectoryFilter(interrupterPredicate)
            .setExecutor(executor)
            .glob();

    // In the non-interruptible case, the interrupt bit should be set, but the
    // glob should return the correct set of full results.
    assertThat(Thread.interrupted()).isTrue();
    assertThat(result)
        .containsExactlyElementsIn(
            resolvePaths(
                ".",
                "foo",
                "foo/bar",
                "foo/bar/wiz",
                "foo/bar/wiz/file",
                "foo/barnacle",
                "foo/barnacle/wiz",
                "food",
                "food/barnacle",
                "food/barnacle/wiz",
                "fool",
                "fool/barnacle",
                "fool/barnacle/wiz"));

    assertThat(executor.isShutdown()).isFalse();
    executor.shutdown();
    assertThat(executor.awaitTermination(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS))
        .isTrue();
  }

  private static Collection<String> removeExcludes(ImmutableList<String> paths, String... excludes)
      throws UnixGlob.BadPattern {
    HashSet<String> pathSet = new HashSet<>(paths);
    UnixGlob.removeExcludes(pathSet, ImmutableList.copyOf(excludes));
    return pathSet;
  }

  @Test
  public void testExcludeFiltering() throws UnixGlob.BadPattern {
    ImmutableList<String> paths = ImmutableList.of("a/A.java", "a/B.java", "a/b/C.java", "c.cc");
    assertThat(removeExcludes(paths, "**/*.java")).containsExactly("c.cc");
    assertThat(removeExcludes(paths, "a/**/*.java")).containsExactly("c.cc");
    assertThat(removeExcludes(paths, "**/nomatch.*")).containsAtLeastElementsIn(paths);
    assertThat(removeExcludes(paths, "a/A.java")).containsExactly("a/B.java", "a/b/C.java", "c.cc");
    assertThat(removeExcludes(paths, "a/?.java")).containsExactly("a/b/C.java", "c.cc");
    assertThat(removeExcludes(paths, "a/*/C.java")).containsExactly("a/A.java", "a/B.java", "c.cc");
    assertThat(removeExcludes(paths, "**")).isEmpty();
    assertThat(removeExcludes(paths, "**/**")).isEmpty();

    // Test filenames that look like code patterns.
    paths = ImmutableList.of("a/A.java", "a/B.java", "a/b/*.java", "a/b/C.java", "c.cc");
    assertThat(removeExcludes(paths, "**/*.java")).containsExactly("c.cc");
    assertThat(removeExcludes(paths, "**/A.java", "**/B.java", "**/C.java"))
        .containsExactly("a/b/*.java", "c.cc");
  }
}
