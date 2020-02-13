// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.skyframe.DiffAwareness.View;
import com.google.devtools.build.lib.skyframe.LocalDiffAwareness.Options;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsProvider;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link MacOSXFsEventsDiffAwareness} */
@RunWith(JUnit4.class)
public class MacOSXFsEventsDiffAwarenessTest {

  private static void rmdirs(Path directory) throws IOException {
    Files.walkFileTree(
        directory,
        new SimpleFileVisitor<Path>() {
          @Override
          public FileVisitResult visitFile(Path file, BasicFileAttributes attrs)
              throws IOException {
            Files.delete(file);
            return FileVisitResult.CONTINUE;
          }

          @Override
          public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException {
            Files.delete(dir);
            return FileVisitResult.CONTINUE;
          }
        });
  }

  private MacOSXFsEventsDiffAwareness underTest;
  private Path watchedPath;
  private OptionsProvider watchFsEnabledProvider;

  @Before
  public void setUp() throws Exception {
    watchedPath = com.google.common.io.Files.createTempDir().getCanonicalFile().toPath();
    underTest = new MacOSXFsEventsDiffAwareness(watchedPath.toString());
    LocalDiffAwareness.Options localDiffOptions = new LocalDiffAwareness.Options();
    localDiffOptions.watchFS = true;
    watchFsEnabledProvider = new LocalDiffAwarenessOptionsProvider(localDiffOptions);
  }

  @After
  public void tearDown() throws Exception {
    underTest.close();
    rmdirs(watchedPath);
  }

  private void scratchFile(String path, String content) throws IOException {
    Path p = watchedPath.resolve(path);
    p.getParent().toFile().mkdirs();
    com.google.common.io.Files.write(content.getBytes(StandardCharsets.UTF_8), p.toFile());
  }

  private void scratchFile(String path) throws IOException {
    scratchFile(path, "");
  }

  /**
   * Checks that the union of the diffs between the current view and each member of some consecutive
   * sequence of views is the specific set of given files.
   *
   * @param view1 the view to compare to
   * @param rawPaths the files to expect in the view
   * @return the new view
   */
  private View assertDiff(View view1, String... rawPaths)
      throws IncompatibleViewException, BrokenDiffAwarenessException, InterruptedException {
    Set<PathFragment> pathsYetToBeSeen = new HashSet<>();
    for (String path : rawPaths) {
      pathsYetToBeSeen.add(PathFragment.create(path));
    }

    // fsevents may be delayed (especially under machine load), which means that we may not notice
    // all file system changes in one go. Try enough times (multiple seconds) for the events to be
    // delivered. Given that each time we call getCurrentView we may get a subset of the total
    // events we expect, track the events we have already seen by subtracting them from the
    // pathsYetToBeSeen set.
    int attempts = 0;
    for (; ; ) {
      View view2 = underTest.getCurrentView(watchFsEnabledProvider);

      ImmutableSet<PathFragment> modifiedSourceFiles =
          underTest.getDiff(view1, view2).modifiedSourceFiles();
      pathsYetToBeSeen.removeAll(modifiedSourceFiles);
      if (pathsYetToBeSeen.isEmpty()) {
        // Found all paths that we wanted to see as modified.
        return view2;
      }

      if (attempts == 600) {
        throw new AssertionError("Paths " + pathsYetToBeSeen + " not found as modified");
      }
      Thread.sleep(100);
      attempts++;
      view1 = view2; // getDiff requires views to be sequential if we want to get meaningful data.
    }
  }

  @Test
  @Ignore("test is flaky, see https://github.com/bazelbuild/bazel/issues/10776")
  public void testSimple() throws Exception {
    View view1 = underTest.getCurrentView(watchFsEnabledProvider);

    scratchFile("a/b/c");
    scratchFile("b/c/d");
    View view2 = assertDiff(view1, "a", "a/b", "a/b/c", "b", "b/c", "b/c/d");

    rmdirs(watchedPath.resolve("a"));
    rmdirs(watchedPath.resolve("b"));
    assertDiff(view2, "a", "a/b", "a/b/c", "b", "b/c", "b/c/d");
  }

  /**
   * Only returns a fixed options class for {@link LocalDiffAwareness.Options}.
   */
  private static final class LocalDiffAwarenessOptionsProvider implements OptionsProvider {
    private final Options localDiffOptions;

    private LocalDiffAwarenessOptionsProvider(Options localDiffOptions) {
      this.localDiffOptions = localDiffOptions;
    }

    @Override
    public <O extends OptionsBase> O getOptions(Class<O> optionsClass) {
      if (optionsClass.equals(LocalDiffAwareness.Options.class)) {
        return optionsClass.cast(localDiffOptions);
      }
      return null;
    }

    @Override
    public Map<String, Object> getStarlarkOptions() {
      return ImmutableMap.of();
    }
  }
}
