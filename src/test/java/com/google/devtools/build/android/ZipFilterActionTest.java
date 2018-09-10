// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.Multimap;
import com.google.devtools.build.android.ZipFilterAction.HashMismatchCheckMode;
import com.google.devtools.build.singlejar.ZipEntryFilter.CustomMergeStrategy;
import com.google.devtools.build.singlejar.ZipEntryFilter.StrategyCallback;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Date;
import java.util.Enumeration;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipException;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ZipFilterAction}. */
@RunWith(JUnit4.class)
public class ZipFilterActionTest {

  private static final class Entry {
    private final String name;
    private final String contents;

    public Entry(String name, String contents) {
      this.name = name;
      this.contents = contents;
    }

    public String getName() {
      return name;
    }

    public String getContents() {
      return contents;
    }
  }

  private enum FilterOperation {
    SKIP,
    RENAME,
    CUSTOM_MERGE,
    COPY
  }

  private static final class TestingStrategyCallback implements StrategyCallback {
    private FilterOperation operation;

    public void assertOp(FilterOperation operation) {
      assertThat(this.operation).isEqualTo(operation);
    }

    @Override
    public void skip() throws IOException {
      operation = FilterOperation.SKIP;
    }

    @Override
    public void rename(String filename, Date date) throws IOException {
      operation = FilterOperation.RENAME;
    }

    @Override
    public void customMerge(Date date, CustomMergeStrategy strategy) throws IOException {
      operation = FilterOperation.CUSTOM_MERGE;
    }

    @Override
    public void copy(Date date) throws IOException {
      operation = FilterOperation.COPY;
    }
  }

  @Rule public ExpectedException thrown = ExpectedException.none();
  @Rule public TemporaryFolder tmp = new TemporaryFolder();

  private int fileCount;
  private TestingStrategyCallback callback;

  private Path createZip(String... filenames) throws IOException {
    Entry[] entries = new Entry[filenames.length];
    int i = 0;
    for (String filename : filenames) {
      entries[i++] = new Entry(filename, "" + fileCount++);
    }
    return createZip(entries);
  }

  private Path createZip(Entry... entries) throws IOException {
    File zip = tmp.newFile();
    try (ZipOutputStream zout = new ZipOutputStream(Files.newOutputStream(zip.toPath()))) {
      for (Entry entry : entries) {
        ZipEntry e = new ZipEntry(entry.getName());
        zout.putNextEntry(e);
        zout.write(entry.getContents().getBytes(UTF_8));
        zout.closeEntry();
      }
    }
    return zip.toPath();
  }

  private List<String> outputEntriesWithArgs(ImmutableList<String> args, File output)
      throws IOException {
    ZipFilterAction.run(args.toArray(new String[0]));
    List<String> filteredEntries = new ArrayList<>();
    try (ZipFile zip = new ZipFile(output)) {
      Enumeration<? extends ZipEntry> entries = zip.entries();
      while (entries.hasMoreElements()) {
        filteredEntries.add(entries.nextElement().getName());
      }
    }
    return filteredEntries;
  }

  @Before public void setup() {
    callback = new TestingStrategyCallback();
  }

  @Test public void testCreateFilter() throws IOException {
    ImmutableSet<Path> filters = ImmutableSet.of(
        createZip("foo.class", "bar.java"),
        createZip("foo.java", "bar.java", "baz.class"));
    ImmutableSet<String> types = ImmutableSet.of(".class");
    Multimap<String, Long> filterFiles = ZipFilterAction.getEntriesToOmit(filters, types);
    assertThat(filterFiles.keySet()).containsExactly("foo.class", "baz.class");
    assertThat(filterFiles).valuesForKey("foo.class").hasSize(1);
    assertThat(filterFiles).valuesForKey("baz.class").hasSize(1);
  }

  @Test public void testCreateFilter_NoZips() throws IOException {
    ImmutableSet<Path> filters = ImmutableSet.of();
    ImmutableSet<String> types = ImmutableSet.of(".class");
    Multimap<String, Long> filterFiles = ZipFilterAction.getEntriesToOmit(filters, types);
    assertThat(filterFiles).isEmpty();
  }

  @Test public void testCreateFilter_NoTypes() throws IOException {
    ImmutableSet<Path> filters = ImmutableSet.of(
        createZip("foo.class", "bar.java"),
        createZip("foo.java", "bar.java", "baz.class"));
    ImmutableSet<String> types = ImmutableSet.of();
    Multimap<String, Long> filterFiles = ZipFilterAction.getEntriesToOmit(filters, types);
    assertThat(filterFiles.keySet())
        .containsExactly("foo.class", "bar.java", "foo.java", "baz.class");
  }

  @Test public void testCreateFilter_MultipleTypes() throws IOException {
    ImmutableSet<Path> filters = ImmutableSet.of(
        createZip("foo.class", "bar.java"),
        createZip("foo.java", "bar.java", "baz.class"));
    ImmutableSet<String> types = ImmutableSet.of(".class", "bar.java");
    Multimap<String, Long> filterFiles = ZipFilterAction.getEntriesToOmit(filters, types);
    assertThat(filterFiles.keySet()).containsExactly("foo.class", "baz.class", "bar.java");
    assertThat(filterFiles).valuesForKey("foo.class").hasSize(1);
    assertThat(filterFiles).valuesForKey("bar.java").hasSize(2);
  }

  @Test public void testZipEntryFilter() throws Exception {
    ZipFilterEntryFilter filter =
        new ZipFilterEntryFilter(
            ".*R.class.*",
            ImmutableSetMultimap.of("foo.class", 1L, "baz.class", 2L),
            ImmutableMap.of("foo.class", 1L, "bar.class", 2L, "baz.class", 3L, "res/R.class", 4L),
            HashMismatchCheckMode.WARN);
    filter.accept("foo.class", callback);
    callback.assertOp(FilterOperation.SKIP);
    filter.accept("bar.class", callback);
    callback.assertOp(FilterOperation.COPY);
    filter.accept("baz.class", callback);
    callback.assertOp(FilterOperation.COPY);
    filter.accept("res/R.class", callback);
    callback.assertOp(FilterOperation.SKIP);
  }

  @Test public void testZipEntryFilter_ErrorOnMismatch() throws Exception {
    ZipFilterEntryFilter filter =
        new ZipFilterEntryFilter(
            ".*R.class.*",
            ImmutableSetMultimap.of("foo.class", 1L, "baz.class", 2L),
            ImmutableMap.of("foo.class", 1L, "bar.class", 2L, "baz.class", 3L, "res/R.class", 4L),
            HashMismatchCheckMode.ERROR);
    filter.accept("foo.class", callback);
    callback.assertOp(FilterOperation.SKIP);
    filter.accept("bar.class", callback);
    callback.assertOp(FilterOperation.COPY);
    filter.accept("res/R.class", callback);
    callback.assertOp(FilterOperation.SKIP);
    filter.accept("baz.class", callback);
    assertThat(filter.sawErrors()).isTrue();
  }

  @Test public void testFlags() throws Exception {
    File input = tmp.newFile("input");
    File output = tmp.newFile("output");
    output.delete();
    File filter1 = tmp.newFile("filter1");
    File filter2 = tmp.newFile("filter2");

    ImmutableList<String> args =
        ImmutableList.of(
            "--inputZip", input.getPath(),
            "--outputZip", output.getPath(),
            "--filterZips",
                Joiner.on(",").join(filter1.getPath(), filter2.getPath(), filter1.getPath()),
            "--filterTypes", Joiner.on(",").join(".class", ".class", ".java"),
            "--explicitFilters", Joiner.on(",").join("R\\.class", "R\\$.*\\.class"),
            "--outputMode", "DONT_CARE",
            "--checkHashMismatch", "IGNORE");
    thrown.expect(ZipException.class);
    thrown.expectMessage("Zip file 'filter1' is malformed");
    ZipFilterAction.run(args.toArray(new String[0]));
  }

  @Test public void testFullIntegration() throws IOException {
    Path input = createZip(new Entry("foo.java", "foo"), new Entry("bar.class", "bar"),
        new Entry("baz.class", "baz"), new Entry("1.class", "1"), new Entry("2.class", "2"),
        new Entry("R.class", "r"), new Entry("Read.class", "read"));
    File output = tmp.newFile();
    output.delete();
    Path filter1 = createZip(new Entry("1.class", "1"), new Entry("b.class", "b"));
    Path filter2 = createZip(new Entry("1.class", "2"), new Entry("2.class", "1"),
        new Entry("foo.java", "foo"), new Entry("bar.class", "bar"));
    ImmutableList<String> args = ImmutableList.of(
        "--inputZip", input.toFile().getPath(),
        "--outputZip", output.getPath(),
        "--filterZips", Joiner.on(",").join(filter1.toFile().getPath(), filter2.toFile().getPath(),
            filter1.toFile().getPath()),
        "--filterTypes", ".class",
        "--explicitFilters", Joiner.on(",").join("R\\.class", "R\\$.*\\.class"),
        "--outputMode", "DONT_CARE");
    assertThat(outputEntriesWithArgs(args, output))
        .containsExactly("foo.java", "baz.class", "2.class", "Read.class");
  }

  @Test public void testFullIntegrationErrorsOnHash() throws IOException {
    Path input = createZip("foo.java", "bar.class", "baz.class");
    File output = tmp.newFile();
    output.delete();
    Path filter = createZip("foo.java", "bar.class");
    ImmutableList<String> args =
        ImmutableList.of(
            "--inputZip",
            input.toFile().getPath(),
            "--outputZip",
            output.getPath(),
            "--filterZips",
            filter.toFile().getPath(),
            "--filterTypes",
            ".class",
            "--checkHashMismatch",
            "ERROR",
            "--outputMode",
            "DONT_CARE");
    int exitCode = ZipFilterAction.run(args.toArray(new String[0]));
    assertThat(exitCode).isEqualTo(1);
  }

  @Test
  public void testSkipHashMismatchCheck() throws IOException {
    Path input =
        createZip(
            new Entry("foo.java", "foo"),
            new Entry("bar.class", "bar1"),
            new Entry("baz.class", "baz"));
    File output = tmp.newFile();
    output.delete();
    Path filter = createZip(new Entry("foo.java", "foo"), new Entry("bar.class", "bar2"));
    ImmutableList<String> args =
        ImmutableList.of(
            "--inputZip",
            input.toFile().getPath(),
            "--outputZip",
            output.getPath(),
            "--filterZips",
            filter.toFile().getPath(),
            "--filterTypes",
            ".class",
            "--checkHashMismatch",
            "IGNORE",
            "--outputMode",
            "DONT_CARE");
    assertThat(outputEntriesWithArgs(args, output)).containsExactly("foo.java", "baz.class");
  }

  @Test public void testFullIntegrationErrorsOnHash_WithExplicitOverride()
      throws IOException {
    Path input = createZip("foo.java", "bar.class", "baz.class");
    File output = tmp.newFile();
    output.delete();
    Path filter = createZip("foo.java", "bar.class");
    ImmutableList<String> args = ImmutableList.of(
        "--inputZip", input.toFile().getPath(),
        "--outputZip", output.getPath(),
        "--filterZips", filter.toFile().getPath(),
        "--filterTypes", ".class",
        "--explicitFilters", "bar\\.class",
        "--outputMode", "DONT_CARE",
        "--errorOnHashMismatch");
    assertThat(outputEntriesWithArgs(args, output)).containsExactly("foo.java", "baz.class");
  }

}
