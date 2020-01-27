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

package com.google.devtools.build.singlejar;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.singlejar.ZipCombiner.OutputMode;
import com.google.devtools.build.singlejar.ZipEntryFilter.CustomMergeStrategy;
import com.google.devtools.build.zip.ExtraData;
import com.google.devtools.build.zip.ZipFileEntry;
import com.google.devtools.build.zip.ZipReader;
import com.google.devtools.build.zip.ZipUtil;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Date;
import java.util.GregorianCalendar;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.jar.JarOutputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipException;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link ZipCombiner}.
 */
@RunWith(JUnit4.class)
public class ZipCombinerTest {
  @Rule public TemporaryFolder tmp = new TemporaryFolder();
  @Rule public ExpectedException thrown = ExpectedException.none();

  private File sampleZip() throws IOException {
    ZipFactory factory = new ZipFactory();
    factory.addFile("hello.txt", "Hello World!");
    return writeInputStreamToFile(factory.toInputStream());
  }

  private File sampleZip2() throws IOException {
    ZipFactory factory = new ZipFactory();
    factory.addFile("hello2.txt", "Hello World 2!");
    return writeInputStreamToFile(factory.toInputStream());
  }

  private File sampleZipWithTwoEntries() throws IOException {
    ZipFactory factory = new ZipFactory();
    factory.addFile("hello.txt", "Hello World!");
    factory.addFile("hello2.txt", "Hello World 2!");
    return writeInputStreamToFile(factory.toInputStream());
  }

  private File sampleZipWithOneUncompressedEntry() throws IOException {
    ZipFactory factory = new ZipFactory();
    factory.addFile("hello.txt", "Hello World!", false);
    return writeInputStreamToFile(factory.toInputStream());
  }

  private File sampleZipWithTwoUncompressedEntries() throws IOException {
    ZipFactory factory = new ZipFactory();
    factory.addFile("hello.txt", "Hello World!", false);
    factory.addFile("hello2.txt", "Hello World 2!", false);
    return writeInputStreamToFile(factory.toInputStream());
  }

  private File writeInputStreamToFile(InputStream in) throws IOException {
    File out = tmp.newFile();
    Files.copy(in, out.toPath(), StandardCopyOption.REPLACE_EXISTING);
    return out;
  }

  private void assertEntry(ZipInputStream zipInput, String filename, long time, byte[] content)
      throws IOException {
    ZipEntry zipEntry = zipInput.getNextEntry();
    assertThat(zipEntry).isNotNull();
    assertThat(zipEntry.getName()).isEqualTo(filename);
    assertThat(zipEntry.getTime()).isEqualTo(time);
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    byte[] buffer = new byte[1024];
    int bytesCopied;
    while ((bytesCopied = zipInput.read(buffer)) != -1) {
      out.write(buffer, 0, bytesCopied);
    }
    assertThat(out.toByteArray()).isEqualTo(content);
  }

  private void assertEntry(ZipInputStream zipInput, String filename, byte[] content)
      throws IOException {
    assertEntry(zipInput, filename, ZipUtil.DOS_EPOCH, content);
  }

  private void assertEntry(ZipInputStream zipInput, String filename, String content)
      throws IOException {
    assertEntry(zipInput, filename, content.getBytes(ISO_8859_1));
  }

  private void assertEntry(ZipInputStream zipInput, String filename, Date date, String content)
      throws IOException {
    assertEntry(zipInput, filename, date.getTime(), content.getBytes(ISO_8859_1));
  }

  @Test
  public void testCompressedDontCare() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(out)) {
      zipCombiner.addZip(sampleZip());
    }
    FakeZipFile expectedResult = new FakeZipFile().addEntry("hello.txt", "Hello World!", true);
    expectedResult.assertSame(out.toByteArray());
  }

  @Test
  public void testCompressedForceDeflate() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(OutputMode.FORCE_DEFLATE, out)) {
      zipCombiner.addZip(sampleZip());
    }
    FakeZipFile expectedResult = new FakeZipFile().addEntry("hello.txt", "Hello World!", true);
    expectedResult.assertSame(out.toByteArray());
  }

  @Test
  public void testCompressedForceStored() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(OutputMode.FORCE_STORED, out)) {
      zipCombiner.addZip(sampleZip());
    }
    FakeZipFile expectedResult = new FakeZipFile().addEntry("hello.txt", "Hello World!", false);
    expectedResult.assertSame(out.toByteArray());
  }

  @Test
  public void testUncompressedDontCare() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(out)) {
      zipCombiner.addZip(sampleZipWithOneUncompressedEntry());
    }
    FakeZipFile expectedResult = new FakeZipFile().addEntry("hello.txt", "Hello World!", false);
    expectedResult.assertSame(out.toByteArray());
  }

  @Test
  public void testUncompressedForceDeflate() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(OutputMode.FORCE_DEFLATE, out)) {
      zipCombiner.addZip(sampleZipWithOneUncompressedEntry());
    }
    FakeZipFile expectedResult = new FakeZipFile().addEntry("hello.txt", "Hello World!", true);
    expectedResult.assertSame(out.toByteArray());
  }

  @Test
  public void testUncompressedForceStored() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(OutputMode.FORCE_STORED, out)) {
      zipCombiner.addZip(sampleZipWithOneUncompressedEntry());
    }
    FakeZipFile expectedResult = new FakeZipFile().addEntry("hello.txt", "Hello World!", false);
    expectedResult.assertSame(out.toByteArray());
  }

  @Test
  public void testCopyTwoEntries() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(out)) {
      zipCombiner.addZip(sampleZipWithTwoEntries());
    }
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello.txt", "Hello World!");
    assertEntry(zipInput, "hello2.txt", "Hello World 2!");
    assertThat(zipInput.getNextEntry()).isNull();
  }

  @Test
  public void testCopyTwoUncompressedEntries() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(out)) {
      zipCombiner.addZip(sampleZipWithTwoUncompressedEntries());
    }
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello.txt", "Hello World!");
    assertEntry(zipInput, "hello2.txt", "Hello World 2!");
    assertThat(zipInput.getNextEntry()).isNull();
  }

  @Test
  public void testCombine() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(out)) {
      zipCombiner.addZip(sampleZip());
      zipCombiner.addZip(sampleZip2());
    }
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello.txt", "Hello World!");
    assertEntry(zipInput, "hello2.txt", "Hello World 2!");
    assertThat(zipInput.getNextEntry()).isNull();
  }

  @Test
  public void testDuplicateEntry() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(out)) {
      zipCombiner.addZip(sampleZip());
      zipCombiner.addZip(sampleZip());
    }
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello.txt", "Hello World!");
    assertThat(zipInput.getNextEntry()).isNull();
  }

  @Test
  public void testBadZipFileNoEntry() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(out)) {
      thrown.expect(ZipException.class);
      thrown.expectMessage("It does not contain an end of central directory record.");
      zipCombiner.addZip(writeInputStreamToFile(new ByteArrayInputStream(new byte[] {1, 2, 3, 4})));
    }
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertThat(zipInput.getNextEntry()).isNull();
  }

  private InputStream asStream(String content) {
    return new ByteArrayInputStream(content.getBytes(UTF_8));
  }

  @Test
  public void testAddFile() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(out)) {
      zipCombiner.addFile("hello.txt", ZipCombiner.DOS_EPOCH, asStream("Hello World!"));
    }
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello.txt", "Hello World!");
    assertThat(zipInput.getNextEntry()).isNull();
  }

  @Test
  public void testAddFileAndDuplicateZipEntry() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(out)) {
      zipCombiner.addFile("hello.txt", ZipCombiner.DOS_EPOCH, asStream("Hello World!"));
      zipCombiner.addZip(sampleZip());
    }
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello.txt", "Hello World!");
    assertThat(zipInput.getNextEntry()).isNull();
  }

  static final class MergeStrategyPlaceHolder implements CustomMergeStrategy {

    @Override
    public void finish(OutputStream out) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void merge(InputStream in, OutputStream out) {
      throw new UnsupportedOperationException();
    }
  }

  private static final CustomMergeStrategy COPY_PLACEHOLDER = new MergeStrategyPlaceHolder();
  private static final CustomMergeStrategy SKIP_PLACEHOLDER = new MergeStrategyPlaceHolder();

  /**
   * A mock implementation that either uses the specified behavior or calls
   * through to copy.
   */
  class MockZipEntryFilter implements ZipEntryFilter {

    private Date date = ZipCombiner.DOS_EPOCH;
    private final List<String> calls = new ArrayList<>();
    // File name to merge strategy map.
    private final Map<String, CustomMergeStrategy> behavior =
        new HashMap<>();
    private final ListMultimap<String, String> renameMap = ArrayListMultimap.create();

    @Override
    public void accept(String filename, StrategyCallback callback) throws IOException {
      calls.add(filename);
      CustomMergeStrategy strategy = behavior.get(filename);
      if (strategy == null) {
        callback.copy(null);
      } else if (strategy == COPY_PLACEHOLDER) {
        List<String> names = renameMap.get(filename);
        if (names != null && !names.isEmpty()) {
          // rename to the next name in list of replacement names.
          String newName = names.get(0);
          callback.rename(newName, null);
          // Unless this is the last replacment names, we pop the used name.
          // The lastreplacement name applies any additional entries.
          if (names.size() > 1) {
            names.remove(0);
          }
        } else {
          callback.copy(null);
        }
      } else if (strategy == SKIP_PLACEHOLDER) {
        callback.skip();
      } else {
        callback.customMerge(date, strategy);
      }
    }
  }

  @Test
  public void testCopyCallsFilter() throws IOException {
    MockZipEntryFilter mockFilter = new MockZipEntryFilter();
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(mockFilter, out)) {
      zipCombiner.addZip(sampleZip());
    }
    assertThat(mockFilter.calls).containsExactly("hello.txt");
  }

  @Test
  public void testDuplicateEntryCallsFilterOnce() throws IOException {
    MockZipEntryFilter mockFilter = new MockZipEntryFilter();
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(mockFilter, out)) {
      zipCombiner.addZip(sampleZip());
      zipCombiner.addZip(sampleZip());
    }
    assertThat(mockFilter.calls).containsExactly("hello.txt");
  }

  @Test
  public void testMergeStrategy() throws IOException {
    MockZipEntryFilter mockFilter = new MockZipEntryFilter();
    mockFilter.behavior.put("hello.txt", new ConcatenateStrategy());
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(mockFilter, out)) {
      zipCombiner.addZip(sampleZip());
      zipCombiner.addZip(sampleZipWithTwoEntries());
    }
    assertThat(mockFilter.calls).containsExactly("hello.txt", "hello2.txt").inOrder();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello2.txt", "Hello World 2!");
    assertEntry(zipInput, "hello.txt", "Hello World!\nHello World!");
    assertThat(zipInput.getNextEntry()).isNull();
  }

  @Test
  public void testMergeStrategyWithUncompressedFiles() throws IOException {
    MockZipEntryFilter mockFilter = new MockZipEntryFilter();
    mockFilter.behavior.put("hello.txt", new ConcatenateStrategy());
    mockFilter.behavior.put("hello2.txt", SKIP_PLACEHOLDER);
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(mockFilter, out)) {
      zipCombiner.addZip(sampleZipWithTwoUncompressedEntries());
      zipCombiner.addZip(sampleZipWithTwoUncompressedEntries());
    }
    assertThat(mockFilter.calls).isEqualTo(Arrays.asList("hello.txt", "hello2.txt"));
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello.txt", "Hello World!\nHello World!");
    assertThat(zipInput.getNextEntry()).isNull();
  }

  @Test
  public void testMergeStrategyWithSlowCopy() throws IOException {
    MockZipEntryFilter mockFilter = new MockZipEntryFilter();
    mockFilter.behavior.put("hello.txt", new SlowConcatenateStrategy());
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(mockFilter, out)) {
      zipCombiner.addZip(sampleZip());
      zipCombiner.addZip(sampleZipWithTwoEntries());
    }
    assertThat(mockFilter.calls).isEqualTo(Arrays.asList("hello.txt", "hello2.txt"));
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello2.txt", "Hello World 2!");
    assertEntry(zipInput, "hello.txt", "Hello World!Hello World!");
    assertThat(zipInput.getNextEntry()).isNull();
  }

  @Test
  public void testMergeStrategyWithUncompressedFilesAndSlowCopy() throws IOException {
    MockZipEntryFilter mockFilter = new MockZipEntryFilter();
    mockFilter.behavior.put("hello.txt", new SlowConcatenateStrategy());
    mockFilter.behavior.put("hello2.txt", SKIP_PLACEHOLDER);
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(mockFilter, out)) {
      zipCombiner.addZip(sampleZipWithTwoUncompressedEntries());
      zipCombiner.addZip(sampleZipWithTwoUncompressedEntries());
    }
    assertThat(mockFilter.calls).containsExactly("hello.txt", "hello2.txt").inOrder();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello.txt", "Hello World!Hello World!");
    assertThat(zipInput.getNextEntry()).isNull();
  }

  private File specialZipWithMinusOne() throws IOException {
    ZipFactory factory = new ZipFactory();
    factory.addFile("hello.txt", new byte[] {-1});
    return writeInputStreamToFile(factory.toInputStream());
  }

  @Test
  public void testMergeStrategyWithSlowCopyAndNegativeBytes() throws IOException {
    MockZipEntryFilter mockFilter = new MockZipEntryFilter();
    mockFilter.behavior.put("hello.txt", new SlowConcatenateStrategy());
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(mockFilter, out)) {
      zipCombiner.addZip(specialZipWithMinusOne());
    }
    assertThat(mockFilter.calls).containsExactly("hello.txt");
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello.txt", new byte[] { -1 });
    assertThat(zipInput.getNextEntry()).isNull();
  }

  @Test
  public void testCopyDateHandling() throws IOException {
    final Date date = new GregorianCalendar(2009, 8, 2, 0, 0, 0).getTime();
    ZipEntryFilter mockFilter =
        new ZipEntryFilter() {
          @Override
          public void accept(String filename, StrategyCallback callback) throws IOException {
            assertThat(filename).isEqualTo("hello.txt");
            callback.copy(date);
          }
        };
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(mockFilter, out)) {
      zipCombiner.addZip(sampleZip());
    }
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello.txt", date, "Hello World!");
    assertThat(zipInput.getNextEntry()).isNull();
  }

  @Test
  public void testMergeDateHandling() throws IOException {
    MockZipEntryFilter mockFilter = new MockZipEntryFilter();
    mockFilter.behavior.put("hello.txt", new ConcatenateStrategy());
    mockFilter.date = new GregorianCalendar(2009, 8, 2, 0, 0, 0).getTime();
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(mockFilter, out)) {
      zipCombiner.addZip(sampleZip());
      zipCombiner.addZip(sampleZipWithTwoEntries());
    }
    assertThat(mockFilter.calls).containsExactly("hello.txt", "hello2.txt").inOrder();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello2.txt", ZipCombiner.DOS_EPOCH, "Hello World 2!");
    assertEntry(zipInput, "hello.txt", mockFilter.date, "Hello World!\nHello World!");
    assertThat(zipInput.getNextEntry()).isNull();
  }

  @Test
  public void testDuplicateCallThrowsException() throws IOException {
    ZipEntryFilter badFilter = new ZipEntryFilter() {
      @Override
      public void accept(String filename, StrategyCallback callback) throws IOException {
        // Duplicate callback call.
        callback.skip();
        callback.copy(null);
      }
    };
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(badFilter, out)) {
      assertThrows(IllegalStateException.class, () -> zipCombiner.addZip(sampleZip()));
    }
  }

  @Test
  public void testNoCallThrowsException() throws IOException {
    ZipEntryFilter badFilter = new ZipEntryFilter() {
      @Override
      public void accept(String filename, StrategyCallback callback) {
        // No callback call.
      }
    };
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(badFilter, out)) {
      assertThrows(IllegalStateException.class, () -> zipCombiner.addZip(sampleZip()));
    }
  }

  // This test verifies that if an entry A is renamed as A (identy mapping),
  // then subsequent entries named A are still subject to filtering.
  // Note: this is different from a copy, where subsequent entries are skipped.
  @Test
  public void testRenameIdentityMapping() throws IOException {
    MockZipEntryFilter mockFilter = new MockZipEntryFilter();
    mockFilter.behavior.put("hello.txt", COPY_PLACEHOLDER);
    mockFilter.behavior.put("hello2.txt", COPY_PLACEHOLDER);
    mockFilter.renameMap.put("hello.txt", "hello.txt");   // identity rename, not copy
    mockFilter.renameMap.put("hello2.txt", "hello2.txt"); // identity rename, not copy
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(mockFilter, out)) {
      zipCombiner.addZip(sampleZipWithTwoEntries());
      zipCombiner.addZip(sampleZipWithTwoEntries());
    }
    assertThat(mockFilter.calls)
        .containsExactly("hello.txt", "hello2.txt", "hello.txt", "hello2.txt")
        .inOrder();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello.txt", "Hello World!");
    assertEntry(zipInput, "hello2.txt", "Hello World 2!");
    assertThat(zipInput.getNextEntry()).isNull();
  }

  // This test verifies that multiple entries with the same name can be
  // renamed to unique names.
  @Test
  public void testRenameNoConflictMapping() throws IOException {
    MockZipEntryFilter mockFilter = new MockZipEntryFilter();
    mockFilter.behavior.put("hello.txt", COPY_PLACEHOLDER);
    mockFilter.behavior.put("hello2.txt", COPY_PLACEHOLDER);
    mockFilter.renameMap.putAll("hello.txt", Arrays.asList("hello1.txt", "hello2.txt"));
    mockFilter.renameMap.putAll("hello2.txt", Arrays.asList("world1.txt", "world2.txt"));
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(mockFilter, out)) {
      zipCombiner.addZip(sampleZipWithTwoEntries());
      zipCombiner.addZip(sampleZipWithTwoEntries());
    }
    assertThat(mockFilter.calls)
        .containsExactly("hello.txt", "hello2.txt", "hello.txt", "hello2.txt")
        .inOrder();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello1.txt", "Hello World!");
    assertEntry(zipInput, "world1.txt", "Hello World 2!");
    assertEntry(zipInput, "hello2.txt", "Hello World!");
    assertEntry(zipInput, "world2.txt", "Hello World 2!");
    assertThat(zipInput.getNextEntry()).isNull();
  }

  // This tests verifies that an attempt to rename an entry to a
  // name already written, results in the entry being skipped, after
  // calling the filter.
  @Test
  public void testRenameSkipUsedName() throws IOException {
    MockZipEntryFilter mockFilter = new MockZipEntryFilter();
    mockFilter.behavior.put("hello.txt", COPY_PLACEHOLDER);
    mockFilter.behavior.put("hello2.txt", COPY_PLACEHOLDER);
    mockFilter.renameMap.putAll("hello.txt",
        Arrays.asList("hello1.txt", "hello2.txt", "hello3.txt"));
    mockFilter.renameMap.put("hello2.txt", "hello2.txt");
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(mockFilter, out)) {
      zipCombiner.addZip(sampleZipWithTwoEntries());
      zipCombiner.addZip(sampleZipWithTwoEntries());
      zipCombiner.addZip(sampleZipWithTwoEntries());
    }
    assertThat(mockFilter.calls)
        .containsExactly(
            "hello.txt", "hello2.txt", "hello.txt", "hello2.txt", "hello.txt", "hello2.txt")
        .inOrder();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello1.txt", "Hello World!");
    assertEntry(zipInput, "hello2.txt", "Hello World 2!");
    assertEntry(zipInput, "hello3.txt", "Hello World!");
    assertThat(zipInput.getNextEntry()).isNull();
  }

  // This tests verifies that if an entry has been copied, then
  // further entries of the same name are skipped (filter not invoked),
  // and entries renamed to the same name are skipped (after calling filter).
  @Test
  public void testRenameAndCopy() throws IOException {
    MockZipEntryFilter mockFilter = new MockZipEntryFilter();
    mockFilter.behavior.put("hello.txt", COPY_PLACEHOLDER);
    mockFilter.behavior.put("hello2.txt", COPY_PLACEHOLDER);
    mockFilter.renameMap.putAll("hello.txt",
        Arrays.asList("hello1.txt", "hello2.txt", "hello3.txt"));
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(mockFilter, out)) {
      zipCombiner.addZip(sampleZipWithTwoEntries());
      zipCombiner.addZip(sampleZipWithTwoEntries());
      zipCombiner.addZip(sampleZipWithTwoEntries());
    }
    assertThat(mockFilter.calls)
        .containsExactly("hello.txt", "hello2.txt", "hello.txt", "hello.txt")
        .inOrder();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello1.txt", "Hello World!");
    assertEntry(zipInput, "hello2.txt", "Hello World 2!");
    assertEntry(zipInput, "hello3.txt", "Hello World!");
    assertThat(zipInput.getNextEntry()).isNull();
  }

  // This tests verifies that if an entry has been skipped, then
  // further entries of the same name are skipped (filter not invoked),
  // and entries renamed to the same name are skipped (after calling filter).
  @Test
  public void testRenameAndSkip() throws IOException {
    MockZipEntryFilter mockFilter = new MockZipEntryFilter();
    mockFilter.behavior.put("hello.txt", COPY_PLACEHOLDER);
    mockFilter.behavior.put("hello2.txt", SKIP_PLACEHOLDER);
    mockFilter.renameMap.putAll("hello.txt",
        Arrays.asList("hello1.txt", "hello2.txt", "hello3.txt"));
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(mockFilter, out)) {
      zipCombiner.addZip(sampleZipWithTwoEntries());
      zipCombiner.addZip(sampleZipWithTwoEntries());
      zipCombiner.addZip(sampleZipWithTwoEntries());
    }
    assertThat(mockFilter.calls)
        .containsExactly("hello.txt", "hello2.txt", "hello.txt", "hello.txt")
        .inOrder();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello1.txt", "Hello World!");
    assertEntry(zipInput, "hello3.txt", "Hello World!");
    assertThat(zipInput.getNextEntry()).isNull();
  }

  // This test verifies that renaming works when input and output
  // disagree on compression method. This is the simple case, where
  // content is read and rewritten, and no header repair is needed.
  @Test
  public void testRenameWithUncompressedFiles () throws IOException {
    MockZipEntryFilter mockFilter = new MockZipEntryFilter();
    mockFilter.behavior.put("hello.txt", COPY_PLACEHOLDER);
    mockFilter.behavior.put("hello2.txt", COPY_PLACEHOLDER);
    mockFilter.renameMap.putAll("hello.txt",
        Arrays.asList("hello1.txt", "hello2.txt", "hello3.txt"));
    mockFilter.renameMap.put("hello2.txt", "hello2.txt");
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(mockFilter, out)) {
      zipCombiner.addZip(sampleZipWithTwoUncompressedEntries());
      zipCombiner.addZip(sampleZipWithTwoUncompressedEntries());
      zipCombiner.addZip(sampleZipWithTwoUncompressedEntries());
    }
    assertThat(mockFilter.calls)
        .containsExactly(
            "hello.txt", "hello2.txt", "hello.txt", "hello2.txt", "hello.txt", "hello2.txt")
        .inOrder();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello1.txt", "Hello World!");
    assertEntry(zipInput, "hello2.txt", "Hello World 2!");
    assertEntry(zipInput, "hello3.txt", "Hello World!");
    assertThat(zipInput.getNextEntry()).isNull();
  }

  // The next two tests check that ZipCombiner can handle a ZIP with an data
  // descriptor marker in the compressed data, i.e. that it does not scan for
  // the data descriptor marker. It's unfortunately a bit tricky to create such
  // a ZIP.
  private static final int LOCAL_FILE_HEADER_MARKER = 0x04034b50;

  // Create a ZIP with a partial entry.
  private InputStream zipWithPartialEntry() {
    ByteBuffer out = ByteBuffer.wrap(new byte[32]).order(ByteOrder.LITTLE_ENDIAN);
    out.clear();
    // file header
    out.putInt(LOCAL_FILE_HEADER_MARKER);  // file header signature
    out.putShort((short) 6); // version to extract
    out.putShort((short) 0); // general purpose bit flag
    out.putShort((short) ZipOutputStream.STORED); // compression method
    out.putShort((short) 0); // mtime (00:00:00)
    out.putShort((short) 0x21); // mdate (1.1.1980)
    out.putInt(0); // crc32
    out.putInt(10); // compressed size
    out.putInt(10); // uncompressed size
    out.putShort((short) 1); // file name length
    out.putShort((short) 0); // extra field length
    out.put((byte) 'a'); // file name

    // file contents
    out.put((byte) 0x01);
    // Unexpected end of file.

    return new ByteArrayInputStream(out.array());
  }

  @Test
  public void testBadZipFilePartialEntry() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner = new ZipCombiner(out)) {
      thrown.expect(ZipException.class);
      thrown.expectMessage("It does not contain an end of central directory record.");
      zipCombiner.addZip(writeInputStreamToFile(zipWithPartialEntry()));
    }
  }

  @Test
  public void testZipCombinerAgainstJavaUtil() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (JarOutputStream jarOut = new JarOutputStream(out)) {
      ZipEntry entry;
      entry = new ZipEntry("META-INF/");
      entry.setTime(ZipCombiner.DOS_EPOCH.getTime());
      entry.setMethod(JarOutputStream.STORED);
      entry.setSize(0);
      entry.setCompressedSize(0);
      entry.setCrc(0);
      jarOut.putNextEntry(entry);
      entry = new ZipEntry("META-INF/MANIFEST.MF");
      entry.setTime(ZipCombiner.DOS_EPOCH.getTime());
      entry.setMethod(JarOutputStream.DEFLATED);
      jarOut.putNextEntry(entry);
      jarOut.write(new byte[] {1, 2, 3, 4});
    }
    File javaFile = writeInputStreamToFile(new ByteArrayInputStream(out.toByteArray()));
    out.reset();

    try (ZipCombiner zipcombiner = new ZipCombiner(out)) {
      zipcombiner.addDirectory("META-INF/", ZipCombiner.DOS_EPOCH,
          new ExtraData[] {new ExtraData((short) 0xCAFE, new byte[0])});
      zipcombiner.addFile("META-INF/MANIFEST.MF", ZipCombiner.DOS_EPOCH,
          new ByteArrayInputStream(new byte[] {1, 2, 3, 4}));
    }
    File zipCombinerFile = writeInputStreamToFile(new ByteArrayInputStream(out.toByteArray()));
    byte[] zipCombinerRaw = out.toByteArray();

    new ZipTester(zipCombinerRaw).validate();
    assertZipFilesEquivalent(zipCombinerFile, javaFile);
  }

  void assertZipFilesEquivalent(File a, File b) throws IOException {
    try (ZipReader x = new ZipReader(a);
        ZipReader y = new ZipReader(b)) {
      Collection<ZipFileEntry> xEntries = x.entries();
      Collection<ZipFileEntry> yEntries = y.entries();
      assertThat(xEntries).hasSize(yEntries.size());
      Iterator<ZipFileEntry> xIter = xEntries.iterator();
      Iterator<ZipFileEntry> yIter = yEntries.iterator();
      for (int i = 0; i < xEntries.size(); i++) {
        assertZipEntryEquivalent(xIter.next(), yIter.next());
      }
    }
  }

  void assertZipEntryEquivalent(ZipFileEntry x, ZipFileEntry y) {
    assertThat(x.getComment()).isEqualTo(y.getComment());
    assertThat(x.getCompressedSize()).isEqualTo(y.getCompressedSize());
    assertThat(x.getCrc()).isEqualTo(y.getCrc());
    assertThat(x.getExternalAttributes()).isEqualTo(y.getExternalAttributes());
    // The JDK adds different extra data to zip files on different platforms, so we don't compare
    // the extra data.
    assertThat(x.getInternalAttributes()).isEqualTo(y.getInternalAttributes());
    assertThat(x.getMethod()).isEqualTo(y.getMethod());
    assertThat(x.getName()).isEqualTo(y.getName());
    assertThat(x.getSize()).isEqualTo(y.getSize());
    assertThat(x.getTime()).isEqualTo(y.getTime());
    assertThat(x.getVersion()).isEqualTo(y.getVersion());
    assertThat(x.getVersionNeeded()).isEqualTo(y.getVersionNeeded());
    // Allow general purpose bit 3 (data descriptor) used in jdk7 to differ.
    // Allow general purpose bit 11 (UTF-8 encoding) used in jdk7 to differ.
    assertThat(x.getFlags() | (1 << 3) | (1 << 11))
        .isEqualTo(y.getFlags() | (1 << 3) | (1 << 11));
  }

  /**
   * Ensures that the code that grows the central directory and the code that patches it is not
   * obviously broken.
   */
  @Test
  public void testLotsOfFiles() throws IOException {
    int fileCount = 100;
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ZipCombiner zipCombiner =
        new ZipCombiner(OutputMode.DONT_CARE, new CopyEntryFilter(), out)) {
      for (int i = 0; i < fileCount; i++) {
        zipCombiner.addFile("hello" + i, ZipCombiner.DOS_EPOCH, asStream("Hello " + i + "!"));
      }
    }
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    for (int i = 0; i < fileCount; i++) {
      assertEntry(zipInput, "hello" + i, "Hello " + i + "!");
    }
    assertThat(zipInput.getNextEntry()).isNull();
    new ZipTester(out.toByteArray()).validate();
  }
}
