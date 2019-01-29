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

import static com.google.common.truth.Truth.assertThat;

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
import java.util.Map;
import org.junit.After;
import org.junit.Before;
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

  private void assertDiff(View view1, View view2, Object... paths)
      throws IncompatibleViewException, BrokenDiffAwarenessException {
    ImmutableSet<PathFragment> modifiedSourceFiles =
        underTest.getDiff(view1, view2).modifiedSourceFiles();
    ImmutableSet<String> toStringSourceFiles = toString(modifiedSourceFiles);
    assertThat(toStringSourceFiles).containsExactly(paths);
  }

  private static ImmutableSet<String> toString(ImmutableSet<PathFragment> modifiedSourceFiles) {
    ImmutableSet.Builder<String> builder = ImmutableSet.builder();
    for (PathFragment path : modifiedSourceFiles) {
      if (!path.toString().isEmpty()) {
        builder.add(path.toString());
      }
    }
    return builder.build();
  }

  @Test
  public void testSimple() throws Exception {
    View view1 = underTest.getCurrentView(watchFsEnabledProvider);
    scratchFile("a/b/c");
    scratchFile("b/c/d");
    Thread.sleep(200); // Wait until the events propagate
    View view2 = underTest.getCurrentView(watchFsEnabledProvider);
    assertDiff(view1, view2, "a", "a/b", "a/b/c", "b", "b/c", "b/c/d");
    rmdirs(watchedPath.resolve("a"));
    rmdirs(watchedPath.resolve("b"));
    Thread.sleep(200); // Wait until the events propagate
    View view3 = underTest.getCurrentView(watchFsEnabledProvider);
    assertDiff(view2, view3, "a", "a/b", "a/b/c", "b", "b/c", "b/c/d");
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
