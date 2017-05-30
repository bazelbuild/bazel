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
package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Joiner;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.util.FileType.HasFilename;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link FileType} and {@link FileTypeSet}. */
@RunWith(JUnit4.class)
public class FileTypeTest {
  private static final FileType CFG = FileType.of(".cfg");
  private static final FileType HTML = FileType.of(".html");
  private static final FileType TEXT = FileType.of(".txt");
  private static final FileType CPP_SOURCE = FileType.of(".cc", ".cpp", ".cxx", ".C");
  private static final FileType JAVA_SOURCE = FileType.of(".java");
  private static final FileType PYTHON_SOURCE = FileType.of(".py");

  private static final class HasFilenameImpl implements HasFilename {
    private final String path;

    private HasFilenameImpl(String path) {
      this.path = path;
    }

    @Override
    public String getFilename() {
      return path;
    }

    @Override
    public String toString() {
      return path;
    }
  }

  @Test
  public void simpleDotMatch() {
    assertThat(TEXT.matches("readme.txt")).isTrue();
  }

  @Test
  public void doubleDotMatches() {
    assertThat(TEXT.matches("read.me.txt")).isTrue();
  }

  @Test
  public void noExtensionMatches() {
    assertThat(FileType.NO_EXTENSION.matches("hello")).isTrue();
    assertThat(FileType.NO_EXTENSION.matches("/path/to/hello")).isTrue();
  }

  @Test
  public void picksLastExtension() {
    assertThat(TEXT.matches("server.cfg.txt")).isTrue();
  }

  @Test
  public void onlyExtensionStillMatches() {
    assertThat(TEXT.matches(".txt")).isTrue();
  }

  @Test
  public void handlesPathObjects() {
    Path readme = new InMemoryFileSystem().getPath("/readme.txt");
    assertThat(TEXT.matches(readme)).isTrue();
  }

  @Test
  public void handlesPathFragmentObjects() {
    PathFragment readme = PathFragment.create("some/where/readme.txt");
    assertThat(TEXT.matches(readme)).isTrue();
  }

  @Test
  public void fileTypeSetContains() {
    FileTypeSet allowedTypes = FileTypeSet.of(TEXT, HTML);

    assertThat(allowedTypes.matches("readme.txt")).isTrue();
    assertThat(allowedTypes.matches("style.css")).isFalse();
  }

  private List<HasFilename> getArtifacts() {
    return Lists.<HasFilename>newArrayList(
        new HasFilenameImpl("Foo.java"),
        new HasFilenameImpl("bar.cc"),
        new HasFilenameImpl("baz.py"));
  }

  private String filterAll(FileType... fileTypes) {
    return Joiner.on(" ").join(FileType.filter(getArtifacts(), fileTypes));
  }

  @Test
  public void justJava() {
    assertThat(filterAll(JAVA_SOURCE)).isEqualTo("Foo.java");
  }

  @Test
  public void javaAndCpp() {
    assertThat(filterAll(JAVA_SOURCE, CPP_SOURCE)).isEqualTo("Foo.java bar.cc");
  }

  @Test
  public void allThree() {
    assertThat(filterAll(JAVA_SOURCE, CPP_SOURCE, PYTHON_SOURCE))
        .isEqualTo("Foo.java bar.cc baz.py");
  }

  private HasFilename filename(final String name) {
    return new HasFilename() {
      @Override
      public String getFilename() {
        return name;
      }
    };
  }

  @Test
  public void checkingSingleWithTypePredicate() throws Exception {
    FileType.HasFilename item = filename("config.txt");

    assertThat(FileType.contains(item, TEXT)).isTrue();
    assertThat(FileType.contains(item, CFG)).isFalse();
  }

  @Test
  public void checkingListWithTypePredicate() throws Exception {
    ImmutableList<FileType.HasFilename> unfiltered = ImmutableList.of(
        filename("config.txt"),
        filename("index.html"),
        filename("README.txt"));

    assertThat(FileType.contains(unfiltered, TEXT)).isTrue();
    assertThat(FileType.contains(unfiltered, CFG)).isFalse();
  }

  @Test
  public void filteringWithTypePredicate() throws Exception {
    ImmutableList<FileType.HasFilename> unfiltered = ImmutableList.of(
        filename("config.txt"),
        filename("index.html"),
        filename("README.txt"),
        filename("archive.zip"));

    assertThat(FileType.filter(unfiltered, TEXT)).containsExactly(unfiltered.get(0),
        unfiltered.get(2)).inOrder();
  }

  @Test
  public void filteringWithMatcherPredicate() throws Exception {
    ImmutableList<FileType.HasFilename> unfiltered = ImmutableList.of(
        filename("config.txt"),
        filename("index.html"),
        filename("README.txt"),
        filename("archive.zip"));

    Predicate<String> textFileTypeMatcher = new Predicate<String>() {
      @Override
      public boolean apply(String input) {
        return TEXT.matches(input);
      }
    };

    assertThat(FileType.filter(unfiltered, textFileTypeMatcher)).containsExactly(unfiltered.get(0),
        unfiltered.get(2)).inOrder();
  }

  @Test
  public void filteringWithAlwaysFalse() throws Exception {
    ImmutableList<FileType.HasFilename> unfiltered = ImmutableList.of(
        filename("config.txt"),
        filename("index.html"),
        filename("binary"),
        filename("archive.zip"));

    assertThat(FileType.filter(unfiltered, FileTypeSet.NO_FILE)).isEmpty();
  }

  @Test
  public void filteringWithAlwaysTrue() throws Exception {
    ImmutableList<FileType.HasFilename> unfiltered = ImmutableList.of(
        filename("config.txt"),
        filename("index.html"),
        filename("binary"),
        filename("archive.zip"));

    assertThat(FileType.filter(unfiltered, FileTypeSet.ANY_FILE)).containsExactly(unfiltered.get(0),
        unfiltered.get(1), unfiltered.get(2), unfiltered.get(3)).inOrder();
  }

  @Test
  public void exclusionWithTypePredicate() throws Exception {
    ImmutableList<FileType.HasFilename> unfiltered = ImmutableList.of(
        filename("config.txt"),
        filename("index.html"),
        filename("README.txt"),
        filename("server.cfg"));

    assertThat(FileType.except(unfiltered, TEXT)).containsExactly(unfiltered.get(1),
        unfiltered.get(3)).inOrder();
  }

  @Test
  public void listFiltering() throws Exception {
    ImmutableList<FileType.HasFilename> unfiltered = ImmutableList.of(
        filename("config.txt"),
        filename("index.html"),
        filename("README.txt"),
        filename("server.cfg"));
    FileTypeSet filter = FileTypeSet.of(HTML, CFG);

    assertThat(FileType.filterList(unfiltered, filter)).containsExactly(unfiltered.get(1),
        unfiltered.get(3)).inOrder();
  }
}
