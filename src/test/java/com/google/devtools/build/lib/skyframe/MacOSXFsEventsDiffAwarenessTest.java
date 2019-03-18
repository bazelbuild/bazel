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
import com.google.common.collect.Sets;
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
import java.util.Set;
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

  private View awaitDiffAndGetView(View view1, CheckedSupplier<View> nextViewSupplier,
      String... expectedPaths) throws Exception {
    Set<PathFragment> modifiedSourceFiles = Sets.newHashSetWithExpectedSize(expectedPaths.length);
    while (modifiedSourceFiles.size() != expectedPaths.length) {
      View view2 = nextViewSupplier.get();
      ImmutableSet<PathFragment> diff = underTest.getDiff(view1, view2).modifiedSourceFiles();
      modifiedSourceFiles.addAll(diff);
      view1 = view2;
      Thread.sleep(100);
    }
    ImmutableSet<String> toStringSourceFiles = toString(ImmutableSet.copyOf(modifiedSourceFiles));
    assertThat(toStringSourceFiles).containsExactly(expectedPaths);
    return view1;
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

  @Test(timeout = 5000)
  public void testSimple() throws Exception {
    CheckedSupplier<View> viewSupplier = () -> underTest.getCurrentView(watchFsEnabledProvider);
    View view1 = viewSupplier.get();
    scratchFile("a/b/c");
    scratchFile("b/c/d");
    String[] expectedPaths = new String[]{"a", "a/b", "a/b/c", "b", "b/c", "b/c/d"};
    View view2 = awaitDiffAndGetView(view1, viewSupplier, expectedPaths);
    rmdirs(watchedPath.resolve("a"));
    rmdirs(watchedPath.resolve("b"));
    awaitDiffAndGetView(view2, viewSupplier, expectedPaths);
  }

  @FunctionalInterface
  private interface CheckedSupplier<T> {
    T get() throws Exception;
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
