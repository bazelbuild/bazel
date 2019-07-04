// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests glob() on Windows (when glob is case-insensitive). */
@RunWith(JUnit4.class)
public class WindowsGlobTest {

  private static void assertMatches(
      UnixGlob.FilesystemCalls fsCalls, Path root, String pattern, String... expecteds)
      throws IOException {
    AtomicReference<UnixGlob.FilesystemCalls> sysCalls = new AtomicReference<>(fsCalls);
    assertThat(
            Iterables.transform(
                new UnixGlob.Builder(root)
                    .addPattern(pattern)
                    .setFilesystemCalls(sysCalls)
                    .setExcludeDirectories(true)
                    .glob(),
                // Convert Paths to Strings, otherwise they'd be compared with the host system's
                // Path
                // comparison semantics.
                a -> a.relativeTo(root).toString()))
        .containsExactlyElementsIn(expecteds);
  }

  private static void assertExcludes(
      Collection<String> unfiltered,
      String exclusionPattern,
      boolean caseSensitive,
      Collection<String> expected) {
    Set<String> matched = new HashSet<>(unfiltered);
    UnixGlob.removeExcludes(matched, ImmutableList.of(exclusionPattern), caseSensitive);
    assertThat(matched).containsExactlyElementsIn(expected);
  }

  @Test
  public void testMatches() throws Exception {
    assertThat(UnixGlob.matches("Foo/**", "Foo/Bar/a.txt", null, true)).isTrue();
    assertThat(UnixGlob.matches("Foo/**", "Foo/Bar/a.txt", null, false)).isTrue();

    assertThat(UnixGlob.matches("foo/**", "Foo/Bar/a.txt", null, true)).isFalse();
    assertThat(UnixGlob.matches("foo/**", "Foo/Bar/a.txt", null, false)).isTrue();

    assertThat(UnixGlob.matches("F*o*o/**", "Foo/Bar/a.txt", null, true)).isTrue();
    assertThat(UnixGlob.matches("F*o*o/**", "Foo/Bar/a.txt", null, false)).isTrue();

    assertThat(UnixGlob.matches("f*o*o/**", "Foo/Bar/a.txt", null, true)).isFalse();
    assertThat(UnixGlob.matches("f*o*o/**", "Foo/Bar/a.txt", null, false)).isTrue();

    assertThat(UnixGlob.matches("Foo/**", "Foo/Bar/a.txt", null, true)).isTrue();
    assertThat(UnixGlob.matches("Foo/**", "Foo/Bar/a.txt", null, false)).isTrue();
  }

  @Test
  public void testExcludes() throws Exception {
    assertExcludes(
        Arrays.asList("Foo/Bar/a.txt", "Foo/Bar/b.dat"),
        "Foo/**/*.dat",
        true,
        Arrays.asList("Foo/Bar/a.txt"));
    assertExcludes(
        Arrays.asList("Foo/Bar/a.txt", "Foo/Bar/b.dat"),
        "Foo/**/*.dat",
        false,
        Arrays.asList("Foo/Bar/a.txt"));

    assertExcludes(
        Arrays.asList("Foo/Bar/a.txt", "Foo/Bar/b.dat"),
        "foo/**/*.dat",
        true,
        Arrays.asList("Foo/Bar/a.txt", "Foo/Bar/b.dat"));
    assertExcludes(
        Arrays.asList("Foo/Bar/a.txt", "Foo/Bar/b.dat"),
        "foo/**/*.dat",
        false,
        Arrays.asList("Foo/Bar/a.txt"));
  }

  private enum MockStat implements FileStatus {
    FILE(true, false),
    DIR(false, true),
    UNKNOWN(false, false);

    private final boolean isFile;
    private final boolean isDir;

    private MockStat(boolean isFile, boolean isDir) {
      this.isFile = isFile;
      this.isDir = isDir;
    }

    @Override
    public boolean isFile() {
      return isFile;
    }

    @Override
    public boolean isDirectory() {
      return isDir;
    }

    @Override
    public boolean isSymbolicLink() {
      return false;
    }

    @Override
    public boolean isSpecialFile() {
      return false;
    }

    @Override
    public long getSize() throws IOException {
      return 0;
    }

    @Override
    public long getLastModifiedTime() throws IOException {
      throw new UnsupportedOperationException();
    }

    @Override
    public long getLastChangeTime() throws IOException {
      throw new UnsupportedOperationException();
    }

    @Override
    public long getNodeId() throws IOException {
      throw new UnsupportedOperationException();
    }
  }

  private static FileSystem mockFs(boolean caseSensitive) throws IOException {
    FileSystem fs =
        new InMemoryFileSystem(BlazeClock.instance()) {
          @Override
          public boolean isFilePathCaseSensitive() {
            return caseSensitive;
          }

          @Override
          public boolean isGlobCaseSensitive() {
            return isFilePathCaseSensitive();
          }
        };
    fs.getPath("/globtmp/Foo/Bar").createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(fs.getPath("/globtmp/Foo/Bar/a.txt"), "foo");
    FileSystemUtils.writeContentAsLatin1(fs.getPath("/globtmp/Foo/Bar/b.dat"), "bar");
    return fs;
  }

  private static Map<String, Collection<Dirent>> mockDirents(Path root) {
    // Map keys must be Strings not Paths, lest they follow the host OS' case-sensitivity policy.
    Map<String, Collection<Dirent>> d =
        root.getFileSystem().isGlobCaseSensitive()
            ? new HashMap<>()
            : new TreeMap<>((String x, String y) -> x.compareToIgnoreCase(y));
    d.put(
        root.getParentDirectory().toString(),
        ImmutableList.of(new Dirent(root.getBaseName(), Dirent.Type.DIRECTORY)));
    d.put(root.toString(), ImmutableList.of(new Dirent("Foo", Dirent.Type.DIRECTORY)));
    d.put(
        root.getRelative("Foo").toString(),
        ImmutableList.of(new Dirent("Bar", Dirent.Type.DIRECTORY)));
    d.put(
        root.getRelative("Foo/Bar").toString(),
        ImmutableList.of(
            new Dirent("a.txt", Dirent.Type.FILE), new Dirent("b.dat", Dirent.Type.FILE)));
    return d;
  }

  private static Map<String, FileStatus> mockStats(Path root) {
    // Map keys must be Strings not Paths, lest they follow the host OS' case-sensitivity policy.
    Map<String, FileStatus> d =
        root.getFileSystem().isGlobCaseSensitive()
            ? new HashMap<>()
            : new TreeMap<>((String x, String y) -> x.compareToIgnoreCase(y));
    d.put(root.getParentDirectory().toString(), MockStat.DIR);
    d.put(root.toString(), MockStat.DIR);
    d.put(root.getRelative("Foo").toString(), MockStat.DIR);
    d.put(root.getRelative("Foo/Bar").toString(), MockStat.DIR);
    d.put(root.getRelative("Foo/Bar/a.txt").toString(), MockStat.FILE);
    d.put(root.getRelative("Foo/Bar/b.dat").toString(), MockStat.FILE);
    return d;
  }

  private static UnixGlob.FilesystemCalls mockFsCalls(Path root) {
    return new UnixGlob.FilesystemCalls() {
      // These maps use case-sensitive or case-insensitive key comparison depending on
      // root.getFileSystem().isGlobCaseSensitive()
      private final Map<String, Collection<Dirent>> dirents = mockDirents(root);
      private final Map<String, FileStatus> stats = mockStats(root);

      @Override
      public Collection<Dirent> readdir(Path path) throws IOException {
        String p = path.toString();
        if (dirents.containsKey(p)) {
          return dirents.get(p);
        }
        throw new IOException(p);
      }

      @Override
      public FileStatus statIfFound(Path path, Symlinks symlinks) throws IOException {
        String p = path.toString();
        if (stats.containsKey(p)) {
          return stats.get(p);
        }
        return MockStat.UNKNOWN;
      }

      @Override
      public Dirent.Type getType(Path path, Symlinks symlinks) throws IOException {
        String p = path.toString();
        if (dirents.containsKey(p)) {
          for (Dirent d : dirents.get(p)) {
            if (d.getName().equals(path.getBaseName())) {
              return d.getType();
            }
          }
        }
        throw new IOException(p);
      }
    };
  }

  @Test
  public void testFoo() throws Exception {
    FileSystem unixFs = mockFs(/* caseSensitive */ true);
    FileSystem winFs = mockFs(/* caseSensitive */ false);

    Path unixRoot = unixFs.getPath("/globtmp");
    Path winRoot = winFs.getPath("/globtmp");

    Path unixRoot2 = unixFs.getPath("/globTMP");
    Path winRoot2 = winFs.getPath("/globTMP");

    UnixGlob.FilesystemCalls unixFsCalls = mockFsCalls(unixRoot);
    UnixGlob.FilesystemCalls winFsCalls = mockFsCalls(winRoot);

    // Try a simple, non-recursive glob that matches no files. (Directories are exluded.)
    assertMatches(unixFsCalls, unixRoot, "Foo/*");
    assertMatches(winFsCalls, winRoot, "Foo/*");

    // Try a simple, non-recursive glob that should match some files.
    assertMatches(unixFsCalls, unixRoot, "Foo/*/*", "Foo/Bar/a.txt", "Foo/Bar/b.dat");
    assertMatches(winFsCalls, winRoot, "Foo/*/*", "Foo/Bar/a.txt", "Foo/Bar/b.dat");

    // Try a recursive glob.
    assertMatches(unixFsCalls, unixRoot, "Foo/**", "Foo/Bar/a.txt", "Foo/Bar/b.dat");
    assertMatches(winFsCalls, winRoot, "Foo/**", "Foo/Bar/a.txt", "Foo/Bar/b.dat");

    // Try a recursive glob, but use incorrect casing in the pattern.
    // The case-insensitive glob should match, but the results retain the casing of the pattern
    // ("foO") because the glob logic checks its existence with 'statIfFound', and uses the name
    // from the pattern and not from the filesystem.
    // The **-matched parts use the actual casing ("Bar") because the glob does a 'readdir' to get
    // these.
    assertMatches(unixFsCalls, unixRoot, "foO/**");
    assertMatches(winFsCalls, winRoot, "foO/**", "foO/Bar/a.txt", "foO/Bar/b.dat");

    // Try the same with another path component in the glob pattern. The casing of that pattern
    // should be retained in the results.
    assertMatches(unixFsCalls, unixRoot, "foO/baR/*");
    assertMatches(winFsCalls, winRoot, "foO/baR/*", "foO/baR/a.txt", "foO/baR/b.dat");

    // Even if the root's casing is incorrect, the case-insensitive glob should match it.
    assertMatches(unixFsCalls, unixRoot2, "**");
    assertMatches(winFsCalls, winRoot2, "**", "Foo/Bar/a.txt", "Foo/Bar/b.dat");

    // Try again the "wrong" root with a "wrong" first component. The result should retain the
    // casing.
    assertMatches(unixFsCalls, unixRoot2, "foO/**");
    assertMatches(winFsCalls, winRoot2, "foO/**", "foO/Bar/a.txt", "foO/Bar/b.dat");

    // Try the same with more "wrong" path components in the pattern. The results should retain all
    // the casing.
    assertMatches(unixFsCalls, unixRoot2, "foO/baR/A.TXT");
    assertMatches(winFsCalls, winRoot2, "foO/baR/A.TXT", "foO/baR/A.TXT");

    // Try a so-called "complex" pattern in the directory name. The glob logic creates a regex and
    // matches it against the result of a 'readdir', so the result retains the filesystem casing.
    assertMatches(unixFsCalls, unixRoot, "Foo/*R/*");
    assertMatches(winFsCalls, winRoot, "Foo/*R/*", "Foo/Bar/a.txt", "Foo/Bar/b.dat");

    // Try the same for a file name pattern. Again, the filesystem casing is used because the
    // matching is done with a regex.
    assertMatches(unixFsCalls, unixRoot, "Foo/Bar/*.TXT");
    assertMatches(winFsCalls, winRoot, "Foo/Bar/*.TXT", "Foo/Bar/a.txt");

    // Try the same with a recursive pattern.
    assertMatches(unixFsCalls, unixRoot, "F*o*o/**", "Foo/Bar/a.txt", "Foo/Bar/b.dat");
    assertMatches(winFsCalls, winRoot, "F*o*o/**", "Foo/Bar/a.txt", "Foo/Bar/b.dat");

    // Try the same with wrong casing.
    assertMatches(unixFsCalls, unixRoot, "f*o*O/**");
    assertMatches(winFsCalls, winRoot, "f*o*O/**", "Foo/Bar/a.txt", "Foo/Bar/b.dat");

    // A "complex" first pattern with the right casing, but wrongly-cased root.
    assertMatches(unixFsCalls, unixRoot2, "F*o*o/**");
    assertMatches(winFsCalls, winRoot2, "F*o*o/**", "Foo/Bar/a.txt", "Foo/Bar/b.dat");

    // A "complex" first pattern with the wrong casing and wrongly-cased root.
    assertMatches(unixFsCalls, unixRoot2, "f*o*O/**");
    assertMatches(winFsCalls, winRoot2, "f*o*O/**", "Foo/Bar/a.txt", "Foo/Bar/b.dat");
  }

  @Test
  public void testStartsWithCase() {
    assertThat(UnixGlob.startsWithCase("", "", true)).isTrue();
    assertThat(UnixGlob.startsWithCase("", "", false)).isTrue();

    assertThat(UnixGlob.startsWithCase("Foo", "", true)).isTrue();
    assertThat(UnixGlob.startsWithCase("Foo", "", false)).isTrue();

    assertThat(UnixGlob.startsWithCase("", "Foo", true)).isFalse();
    assertThat(UnixGlob.startsWithCase("", "Foo", false)).isFalse();

    assertThat(UnixGlob.startsWithCase("Fo", "Foo", true)).isFalse();
    assertThat(UnixGlob.startsWithCase("Fo", "Foo", false)).isFalse();

    assertThat(UnixGlob.startsWithCase("Foo", "Foo", true)).isTrue();
    assertThat(UnixGlob.startsWithCase("Foo", "Foo", false)).isTrue();

    assertThat(UnixGlob.startsWithCase("Foox", "Foo", true)).isTrue();
    assertThat(UnixGlob.startsWithCase("Foox", "Foo", false)).isTrue();

    assertThat(UnixGlob.startsWithCase("xFoo", "Foo", true)).isFalse();
    assertThat(UnixGlob.startsWithCase("xFoo", "Foo", false)).isFalse();

    assertThat(UnixGlob.startsWithCase("Foox", "foO", true)).isFalse();
    assertThat(UnixGlob.startsWithCase("Foox", "foO", false)).isTrue();
  }

  @Test
  public void testEndsWithCase() {
    assertThat(UnixGlob.endsWithCase("", "", true)).isTrue();
    assertThat(UnixGlob.endsWithCase("", "", false)).isTrue();

    assertThat(UnixGlob.endsWithCase("Foo", "", true)).isTrue();
    assertThat(UnixGlob.endsWithCase("Foo", "", false)).isTrue();

    assertThat(UnixGlob.endsWithCase("", "Foo", true)).isFalse();
    assertThat(UnixGlob.endsWithCase("", "Foo", false)).isFalse();

    assertThat(UnixGlob.endsWithCase("Fo", "Foo", true)).isFalse();
    assertThat(UnixGlob.endsWithCase("Fo", "Foo", false)).isFalse();

    assertThat(UnixGlob.endsWithCase("Foo", "Foo", true)).isTrue();
    assertThat(UnixGlob.endsWithCase("Foo", "Foo", false)).isTrue();

    assertThat(UnixGlob.endsWithCase("Foox", "Foo", true)).isFalse();
    assertThat(UnixGlob.endsWithCase("Foox", "Foo", false)).isFalse();

    assertThat(UnixGlob.endsWithCase("xFoo", "Foo", true)).isTrue();
    assertThat(UnixGlob.endsWithCase("xFoo", "Foo", false)).isTrue();

    assertThat(UnixGlob.endsWithCase("xFoo", "foO", true)).isFalse();
    assertThat(UnixGlob.endsWithCase("xFoo", "foO", false)).isTrue();
  }
}
