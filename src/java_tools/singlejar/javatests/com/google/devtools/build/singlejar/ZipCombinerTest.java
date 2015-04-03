// Copyright 2015 Google Inc. All rights reserved.
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
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.base.Preconditions;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.singlejar.ZipCombiner.OutputMode;
import com.google.devtools.build.singlejar.ZipEntryFilter.CustomMergeStrategy;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Date;
import java.util.GregorianCalendar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.jar.JarOutputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

/**
 * Unit tests for {@link ZipCombiner}.
 */
@RunWith(JUnit4.class)
public class ZipCombinerTest {

  private static final Date DOS_EPOCH = ZipCombiner.DOS_EPOCH;

  private InputStream sampleZip() {
    ZipFactory factory = new ZipFactory();
    factory.addFile("hello.txt", "Hello World!");
    return factory.toInputStream();
  }

  private InputStream sampleZip2() {
    ZipFactory factory = new ZipFactory();
    factory.addFile("hello2.txt", "Hello World 2!");
    return factory.toInputStream();
  }

  private InputStream sampleZipWithTwoEntries() {
    ZipFactory factory = new ZipFactory();
    factory.addFile("hello.txt", "Hello World!");
    factory.addFile("hello2.txt", "Hello World 2!");
    return factory.toInputStream();
  }

  private InputStream sampleZipWithOneUncompressedEntry() {
    ZipFactory factory = new ZipFactory();
    factory.addFile("hello.txt", "Hello World!", false);
    return factory.toInputStream();
  }

  private InputStream sampleZipWithTwoUncompressedEntries() {
    ZipFactory factory = new ZipFactory();
    factory.addFile("hello.txt", "Hello World!", false);
    factory.addFile("hello2.txt", "Hello World 2!", false);
    return factory.toInputStream();
  }

  private void assertEntry(ZipInputStream zipInput, String filename, long time, byte[] content)
      throws IOException {
    ZipEntry zipEntry = zipInput.getNextEntry();
    assertNotNull(zipEntry);
    assertEquals(filename, zipEntry.getName());
    assertEquals(time, zipEntry.getTime());
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    byte[] buffer = new byte[1024];
    int bytesCopied;
    while ((bytesCopied = zipInput.read(buffer)) != -1) {
      out.write(buffer, 0, bytesCopied);
    }
    assertTrue(Arrays.equals(content, out.toByteArray()));
  }

  private void assertEntry(ZipInputStream zipInput, String filename, byte[] content)
      throws IOException {
    assertEntry(zipInput, filename, ZipCombiner.DOS_EPOCH.getTime(), content);
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
  public void testDateToDosTime() {
    assertEquals(0x210000, ZipCombiner.dateToDosTime(ZipCombiner.DOS_EPOCH));
    Calendar calendar = new GregorianCalendar();
    for (int i = 1980; i <= 2107; i++) {
      calendar.set(i, 0, 1, 0, 0, 0);
      int result = ZipCombiner.dateToDosTime(calendar.getTime());
      assertEquals(i - 1980, result >>> 25);
      assertEquals(1, (result >> 21) & 0xf);
      assertEquals(1, (result >> 16) & 0x1f);
      assertEquals(0, result & 0xffff);
    }
  }

  @Test
  public void testDateToDosTimeFailsForBadValues() {
    try {
      Calendar calendar = new GregorianCalendar();
      calendar.set(1979, 0, 1, 0, 0, 0);
      ZipCombiner.dateToDosTime(calendar.getTime());
      fail();
    } catch (IllegalArgumentException e) {
      /* Expected exception. */
    }
    try {
      Calendar calendar = new GregorianCalendar();
      calendar.set(2108, 0, 1, 0, 0, 0);
      ZipCombiner.dateToDosTime(calendar.getTime());
      fail();
    } catch (IllegalArgumentException e) {
      /* Expected exception. */
    }
  }

  @Test
  public void testCompressedDontCare() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ZipCombiner singleJar = new ZipCombiner(out);
    singleJar.addZip(sampleZip());
    singleJar.close();
    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("hello.txt", "Hello World!", true);
    expectedResult.assertSame(out.toByteArray());
  }

  @Test
  public void testCompressedForceDeflate() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ZipCombiner singleJar = new ZipCombiner(OutputMode.FORCE_DEFLATE, out);
    singleJar.addZip(sampleZip());
    singleJar.close();
    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("hello.txt", "Hello World!", true);
    expectedResult.assertSame(out.toByteArray());
  }

  @Test
  public void testCompressedForceStored() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ZipCombiner singleJar = new ZipCombiner(OutputMode.FORCE_STORED, out);
    singleJar.addZip(sampleZip());
    singleJar.close();
    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("hello.txt", "Hello World!", false);
    expectedResult.assertSame(out.toByteArray());
  }

  @Test
  public void testUncompressedDontCare() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ZipCombiner singleJar = new ZipCombiner(out);
    singleJar.addZip(sampleZipWithOneUncompressedEntry());
    singleJar.close();
    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("hello.txt", "Hello World!", false);
    expectedResult.assertSame(out.toByteArray());
  }

  @Test
  public void testUncompressedForceDeflate() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ZipCombiner singleJar = new ZipCombiner(OutputMode.FORCE_DEFLATE, out);
    singleJar.addZip(sampleZipWithOneUncompressedEntry());
    singleJar.close();
    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("hello.txt", "Hello World!", true);
    expectedResult.assertSame(out.toByteArray());
  }

  @Test
  public void testUncompressedForceStored() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ZipCombiner singleJar = new ZipCombiner(OutputMode.FORCE_STORED, out);
    singleJar.addZip(sampleZipWithOneUncompressedEntry());
    singleJar.close();
    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("hello.txt", "Hello World!", false);
    expectedResult.assertSame(out.toByteArray());
  }

  @Test
  public void testCopyTwoEntries() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ZipCombiner singleJar = new ZipCombiner(out);
    singleJar.addZip(sampleZipWithTwoEntries());
    singleJar.close();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello.txt", "Hello World!");
    assertEntry(zipInput, "hello2.txt", "Hello World 2!");
    assertNull(zipInput.getNextEntry());
  }

  @Test
  public void testCopyTwoUncompressedEntries() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ZipCombiner singleJar = new ZipCombiner(out);
    singleJar.addZip(sampleZipWithTwoUncompressedEntries());
    singleJar.close();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello.txt", "Hello World!");
    assertEntry(zipInput, "hello2.txt", "Hello World 2!");
    assertNull(zipInput.getNextEntry());
  }

  @Test
  public void testCombine() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ZipCombiner singleJar = new ZipCombiner(out);
    singleJar.addZip(sampleZip());
    singleJar.addZip(sampleZip2());
    singleJar.close();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello.txt", "Hello World!");
    assertEntry(zipInput, "hello2.txt", "Hello World 2!");
    assertNull(zipInput.getNextEntry());
  }

  @Test
  public void testDuplicateEntry() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ZipCombiner singleJar = new ZipCombiner(out);
    singleJar.addZip(sampleZip());
    singleJar.addZip(sampleZip());
    singleJar.close();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello.txt", "Hello World!");
    assertNull(zipInput.getNextEntry());
  }

  // Returns an input stream that can only read one byte at a time.
  private InputStream slowRead(final InputStream in) {
    return new InputStream() {
      @Override
      public int read() throws IOException {
        return in.read();
      }
      @Override
      public int read(byte b[], int off, int len) throws IOException {
        Preconditions.checkArgument(b != null);
        Preconditions.checkArgument((len >= 0) && (off >= 0));
        Preconditions.checkArgument(len <= b.length - off);
        if (len == 0) {
          return 0;
        }
        int value = read();
        if (value == -1) {
          return -1;
        }
        b[off] = (byte) value;
        return 1;
      }
    };
  }

  @Test
  public void testDuplicateUncompressedEntryWithSlowRead() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ZipCombiner singleJar = new ZipCombiner(out);
    singleJar.addZip(slowRead(sampleZipWithOneUncompressedEntry()));
    singleJar.addZip(slowRead(sampleZipWithOneUncompressedEntry()));
    singleJar.close();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello.txt", "Hello World!");
    assertNull(zipInput.getNextEntry());
  }

  @Test
  public void testDuplicateEntryWithSlowRead() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ZipCombiner singleJar = new ZipCombiner(out);
    singleJar.addZip(slowRead(sampleZip()));
    singleJar.addZip(slowRead(sampleZip()));
    singleJar.close();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello.txt", "Hello World!");
    assertNull(zipInput.getNextEntry());
  }

  @Test
  public void testBadZipFileNoEntry() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ZipCombiner singleJar = new ZipCombiner(out);
    singleJar.addZip(new ByteArrayInputStream(new byte[] { 1, 2, 3, 4 }));
    singleJar.close();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertNull(zipInput.getNextEntry());
  }

  private InputStream asStream(String content) {
    return new ByteArrayInputStream(content.getBytes(UTF_8));
  }

  @Test
  public void testAddFile() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ZipCombiner singleJar = new ZipCombiner(out);
    singleJar.addFile("hello.txt", DOS_EPOCH, asStream("Hello World!"));
    singleJar.close();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello.txt", "Hello World!");
    assertNull(zipInput.getNextEntry());
  }

  @Test
  public void testAddFileAndDuplicateZipEntry() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ZipCombiner singleJar = new ZipCombiner(out);
    singleJar.addFile("hello.txt", DOS_EPOCH, asStream("Hello World!"));
    singleJar.addZip(sampleZip());
    singleJar.close();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello.txt", "Hello World!");
    assertNull(zipInput.getNextEntry());
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

    private Date date = DOS_EPOCH;
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
    ZipCombiner singleJar = new ZipCombiner(mockFilter, out);
    singleJar.addZip(sampleZip());
    singleJar.close();
    assertEquals(Arrays.asList("hello.txt"), mockFilter.calls);
  }

  @Test
  public void testDuplicateEntryCallsFilterOnce() throws IOException {
    MockZipEntryFilter mockFilter = new MockZipEntryFilter();
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ZipCombiner singleJar = new ZipCombiner(mockFilter, out);
    singleJar.addZip(sampleZip());
    singleJar.addZip(sampleZip());
    singleJar.close();
    assertEquals(Arrays.asList("hello.txt"), mockFilter.calls);
  }

  @Test
  public void testMergeStrategy() throws IOException {
    MockZipEntryFilter mockFilter = new MockZipEntryFilter();
    mockFilter.behavior.put("hello.txt", new ConcatenateStrategy());
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ZipCombiner singleJar = new ZipCombiner(mockFilter, out);
    singleJar.addZip(sampleZip());
    singleJar.addZip(sampleZipWithTwoEntries());
    singleJar.close();
    assertEquals(Arrays.asList("hello.txt", "hello2.txt"), mockFilter.calls);
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello2.txt", "Hello World 2!");
    assertEntry(zipInput, "hello.txt", "Hello World!\nHello World!");
    assertNull(zipInput.getNextEntry());
  }

  @Test
  public void testMergeStrategyWithUncompressedFiles() throws IOException {
    MockZipEntryFilter mockFilter = new MockZipEntryFilter();
    mockFilter.behavior.put("hello.txt", new ConcatenateStrategy());
    mockFilter.behavior.put("hello2.txt", SKIP_PLACEHOLDER);
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ZipCombiner singleJar = new ZipCombiner(mockFilter, out);
    singleJar.addZip(sampleZipWithTwoUncompressedEntries());
    singleJar.addZip(sampleZipWithTwoUncompressedEntries());
    singleJar.close();
    assertEquals(Arrays.asList("hello.txt", "hello2.txt"), mockFilter.calls);
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello.txt", "Hello World!\nHello World!");
    assertNull(zipInput.getNextEntry());
  }

  @Test
  public void testMergeStrategyWithUncompressedEntriesAndSlowRead() throws IOException {
    MockZipEntryFilter mockFilter = new MockZipEntryFilter();
    mockFilter.behavior.put("hello.txt", new ConcatenateStrategy());
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ZipCombiner singleJar = new ZipCombiner(mockFilter, out);
    singleJar.addZip(slowRead(sampleZipWithOneUncompressedEntry()));
    singleJar.addZip(slowRead(sampleZipWithTwoUncompressedEntries()));
    singleJar.close();
    assertEquals(Arrays.asList("hello.txt", "hello2.txt"), mockFilter.calls);
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello2.txt", "Hello World 2!");
    assertEntry(zipInput, "hello.txt", "Hello World!\nHello World!");
    assertNull(zipInput.getNextEntry());
  }

  @Test
  public void testMergeStrategyWithSlowCopy() throws IOException {
    MockZipEntryFilter mockFilter = new MockZipEntryFilter();
    mockFilter.behavior.put("hello.txt", new SlowConcatenateStrategy());
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ZipCombiner singleJar = new ZipCombiner(mockFilter, out);
    singleJar.addZip(sampleZip());
    singleJar.addZip(sampleZipWithTwoEntries());
    singleJar.close();
    assertEquals(Arrays.asList("hello.txt", "hello2.txt"), mockFilter.calls);
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello2.txt", "Hello World 2!");
    assertEntry(zipInput, "hello.txt", "Hello World!Hello World!");
    assertNull(zipInput.getNextEntry());
  }

  @Test
  public void testMergeStrategyWithUncompressedFilesAndSlowCopy() throws IOException {
    MockZipEntryFilter mockFilter = new MockZipEntryFilter();
    mockFilter.behavior.put("hello.txt", new SlowConcatenateStrategy());
    mockFilter.behavior.put("hello2.txt", SKIP_PLACEHOLDER);
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ZipCombiner singleJar = new ZipCombiner(mockFilter, out);
    singleJar.addZip(sampleZipWithTwoUncompressedEntries());
    singleJar.addZip(sampleZipWithTwoUncompressedEntries());
    singleJar.close();
    assertEquals(Arrays.asList("hello.txt", "hello2.txt"), mockFilter.calls);
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello.txt", "Hello World!Hello World!");
    assertNull(zipInput.getNextEntry());
  }

  private InputStream specialZipWithMinusOne() {
    ZipFactory factory = new ZipFactory();
    factory.addFile("hello.txt", new byte[] {-1});
    return factory.toInputStream();
  }

  @Test
  public void testMergeStrategyWithSlowCopyAndNegativeBytes() throws IOException {
    MockZipEntryFilter mockFilter = new MockZipEntryFilter();
    mockFilter.behavior.put("hello.txt", new SlowConcatenateStrategy());
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ZipCombiner singleJar = new ZipCombiner(mockFilter, out);
    singleJar.addZip(specialZipWithMinusOne());
    singleJar.close();
    assertEquals(Arrays.asList("hello.txt"), mockFilter.calls);
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello.txt", new byte[] { -1 });
    assertNull(zipInput.getNextEntry());
  }

  @Test
  public void testCopyDateHandling() throws IOException {
    final Date date = new GregorianCalendar(2009, 8, 2, 0, 0, 0).getTime();
    ZipEntryFilter mockFilter = new ZipEntryFilter() {
      @Override
      public void accept(String filename, StrategyCallback callback) throws IOException {
        assertEquals("hello.txt", filename);
        callback.copy(date);
      }
    };
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ZipCombiner singleJar = new ZipCombiner(mockFilter, out);
    singleJar.addZip(sampleZip());
    singleJar.close();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello.txt", date, "Hello World!");
    assertNull(zipInput.getNextEntry());
  }

  @Test
  public void testMergeDateHandling() throws IOException {
    MockZipEntryFilter mockFilter = new MockZipEntryFilter();
    mockFilter.behavior.put("hello.txt", new ConcatenateStrategy());
    mockFilter.date = new GregorianCalendar(2009, 8, 2, 0, 0, 0).getTime();
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ZipCombiner singleJar = new ZipCombiner(mockFilter, out);
    singleJar.addZip(sampleZip());
    singleJar.addZip(sampleZipWithTwoEntries());
    singleJar.close();
    assertEquals(Arrays.asList("hello.txt", "hello2.txt"), mockFilter.calls);
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello2.txt", DOS_EPOCH, "Hello World 2!");
    assertEntry(zipInput, "hello.txt", mockFilter.date, "Hello World!\nHello World!");
    assertNull(zipInput.getNextEntry());
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
    try (ZipCombiner singleJar = new ZipCombiner(badFilter, out)) {
      singleJar.addZip(sampleZip());
      fail();
    } catch (IllegalStateException e) {
      // Expected exception.
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
    try (ZipCombiner singleJar = new ZipCombiner(badFilter, out)) {
      singleJar.addZip(sampleZip());
      fail();
    } catch (IllegalStateException e) {
      // Expected exception.
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
    ZipCombiner singleJar = new ZipCombiner(mockFilter, out);
    singleJar.addZip(sampleZipWithTwoEntries());
    singleJar.addZip(sampleZipWithTwoEntries());
    singleJar.close();
    assertThat(mockFilter.calls).containsExactly("hello.txt", "hello2.txt",
        "hello.txt", "hello2.txt").inOrder();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello.txt", "Hello World!");
    assertEntry(zipInput, "hello2.txt", "Hello World 2!");
    assertNull(zipInput.getNextEntry());
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
    ZipCombiner singleJar = new ZipCombiner(mockFilter, out);
    singleJar.addZip(sampleZipWithTwoEntries());
    singleJar.addZip(sampleZipWithTwoEntries());
    singleJar.close();
    assertThat(mockFilter.calls).containsExactly("hello.txt", "hello2.txt",
        "hello.txt", "hello2.txt").inOrder();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello1.txt", "Hello World!");
    assertEntry(zipInput, "world1.txt", "Hello World 2!");
    assertEntry(zipInput, "hello2.txt", "Hello World!");
    assertEntry(zipInput, "world2.txt", "Hello World 2!");
    assertNull(zipInput.getNextEntry());
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
    ZipCombiner singleJar = new ZipCombiner(mockFilter, out);
    singleJar.addZip(sampleZipWithTwoEntries());
    singleJar.addZip(sampleZipWithTwoEntries());
    singleJar.addZip(sampleZipWithTwoEntries());
    singleJar.close();
    assertThat(mockFilter.calls).containsExactly("hello.txt", "hello2.txt",
        "hello.txt", "hello2.txt", "hello.txt", "hello2.txt").inOrder();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello1.txt", "Hello World!");
    assertEntry(zipInput, "hello2.txt", "Hello World 2!");
    assertEntry(zipInput, "hello3.txt", "Hello World!");
    assertNull(zipInput.getNextEntry());
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
    ZipCombiner singleJar = new ZipCombiner(mockFilter, out);
    singleJar.addZip(sampleZipWithTwoEntries());
    singleJar.addZip(sampleZipWithTwoEntries());
    singleJar.addZip(sampleZipWithTwoEntries());
    singleJar.close();
    assertThat(mockFilter.calls).containsExactly("hello.txt", "hello2.txt",
        "hello.txt", "hello.txt").inOrder();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello1.txt", "Hello World!");
    assertEntry(zipInput, "hello2.txt", "Hello World 2!");
    assertEntry(zipInput, "hello3.txt", "Hello World!");
    assertNull(zipInput.getNextEntry());
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
    ZipCombiner singleJar = new ZipCombiner(mockFilter, out);
    singleJar.addZip(sampleZipWithTwoEntries());
    singleJar.addZip(sampleZipWithTwoEntries());
    singleJar.addZip(sampleZipWithTwoEntries());
    singleJar.close();
    assertThat(mockFilter.calls).containsExactly("hello.txt", "hello2.txt",
        "hello.txt", "hello.txt").inOrder();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello1.txt", "Hello World!");
    assertEntry(zipInput, "hello3.txt", "Hello World!");
    assertNull(zipInput.getNextEntry());
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
    ZipCombiner singleJar = new ZipCombiner(mockFilter, out);
    singleJar.addZip(sampleZipWithTwoUncompressedEntries());
    singleJar.addZip(sampleZipWithTwoUncompressedEntries());
    singleJar.addZip(sampleZipWithTwoUncompressedEntries());
    singleJar.close();
    assertThat(mockFilter.calls).containsExactly("hello.txt", "hello2.txt",
        "hello.txt", "hello2.txt", "hello.txt", "hello2.txt").inOrder();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "hello1.txt", "Hello World!");
    assertEntry(zipInput, "hello2.txt", "Hello World 2!");
    assertEntry(zipInput, "hello3.txt", "Hello World!");
    assertNull(zipInput.getNextEntry());
  }

  // The next two tests check that ZipCombiner can handle a ZIP with an data
  // descriptor marker in the compressed data, i.e. that it does not scan for
  // the data descriptor marker. It's unfortunately a bit tricky to create such
  // a ZIP.
  private static final int LOCAL_FILE_HEADER_MARKER = 0x04034b50;
  private static final int DATA_DESCRIPTOR_MARKER = 0x08074b50;
  private static final byte[] DATA_DESCRIPTOR_MARKER_AS_BYTES = new byte[] {
    0x50, 0x4b, 0x07, 0x08
  };

  // Create a ZIP with an data descriptor marker in the DEFLATE content of a
  // file. To do that, we build the ZIP byte by byte.
  private InputStream zipWithUnexpectedDataDescriptorMarker() {
    ByteBuffer out = ByteBuffer.wrap(new byte[200]).order(ByteOrder.LITTLE_ENDIAN);
    out.clear();
    // file header
    out.putInt(LOCAL_FILE_HEADER_MARKER);  // file header signature
    out.putShort((short) 6); // version to extract
    out.putShort((short) 8); // general purpose bit flag
    out.putShort((short) ZipOutputStream.DEFLATED); // compression method
    out.putShort((short) 0); // mtime (00:00:00)
    out.putShort((short) 0x21); // mdate (1.1.1980)
    out.putInt(0); // crc32
    out.putInt(0); // compressed size
    out.putInt(0); // uncompressed size
    out.putShort((short) 1); // file name length
    out.putShort((short) 0); // extra field length
    out.put((byte) 'a'); // file name

    // file contents
    out.put((byte) 0x01); // deflated content block is last block and uncompressed
    out.putShort((short) 4); // uncompressed block length
    out.putShort((short) ~4); // negated uncompressed block length
    out.putInt(DATA_DESCRIPTOR_MARKER); // 4 bytes uncompressed data

    // data descriptor
    out.putInt(DATA_DESCRIPTOR_MARKER); // data descriptor with marker
    out.putInt((int) ZipFactory.calculateCrc32(DATA_DESCRIPTOR_MARKER_AS_BYTES));
    out.putInt(9);
    out.putInt(4);
    // We omit the central directory here. It's currently not used by
    // ZipCombiner or by java.util.zip.ZipInputStream, so that shouldn't be a
    // problem.
    return new ByteArrayInputStream(out.array());
  }

  // Check that the created ZIP is correct.
  @Test
  public void testZipWithUnexpectedDataDescriptorMarkerIsCorrect() throws IOException {
    ZipInputStream zipInput = new ZipInputStream(zipWithUnexpectedDataDescriptorMarker());
    assertEntry(zipInput, "a", DATA_DESCRIPTOR_MARKER_AS_BYTES);
    assertNull(zipInput.getNextEntry());
  }

  // Check that ZipCombiner handles the ZIP correctly.
  @Test
  public void testZipWithUnexpectedDataDescriptorMarker() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ZipCombiner singleJar = new ZipCombiner(out);
    singleJar.addZip(zipWithUnexpectedDataDescriptorMarker());
    singleJar.close();
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
    assertEntry(zipInput, "a", DATA_DESCRIPTOR_MARKER_AS_BYTES);
    assertNull(zipInput.getNextEntry());
  }

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
    try (ZipCombiner singleJar = new ZipCombiner(out)) {
      singleJar.addZip(zipWithPartialEntry());
      fail();
    } catch (EOFException e) {
      // Expected exception.
    }
  }

  @Test
  public void testSimpleJarAgainstJavaUtil() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    JarOutputStream jarOut = new JarOutputStream(out);
    ZipEntry entry;
    entry = new ZipEntry("META-INF/");
    entry.setTime(DOS_EPOCH.getTime());
    entry.setMethod(JarOutputStream.STORED);
    entry.setSize(0);
    entry.setCompressedSize(0);
    entry.setCrc(0);
    jarOut.putNextEntry(entry);
    entry = new ZipEntry("META-INF/MANIFEST.MF");
    entry.setTime(DOS_EPOCH.getTime());
    entry.setMethod(JarOutputStream.DEFLATED);
    jarOut.putNextEntry(entry);
    jarOut.write(new byte[] { 1, 2, 3, 4 });
    jarOut.close();
    byte[] javaFile = out.toByteArray();
    out.reset();

    ZipCombiner singleJar = new ZipCombiner(out);
    singleJar.addDirectory("META-INF/", DOS_EPOCH,
        new ExtraData[] { new ExtraData((short) 0xCAFE, new byte[0]) });
    singleJar.addFile("META-INF/MANIFEST.MF", DOS_EPOCH,
        new ByteArrayInputStream(new byte[] { 1, 2, 3, 4 }));
    singleJar.close();
    byte[] singlejarFile = out.toByteArray();

    new ZipTester(singlejarFile).validate();
    assertZipFilesEquivalent(singlejarFile, javaFile);
  }

  void assertZipFilesEquivalent(byte[] x, byte[] y) {
    assertEquals(x.length, y.length);

    for (int i = 0; i < x.length; i++) {
      if (x[i] != y[i]) {
        // Allow general purpose bit 11 (UTF-8 encoding) used in jdk7 to differ
        assertEquals("at position " + i, 0x08, x[i] ^ y[i]);
        // Check that x[i] is the second byte of a general purpose bit flag.
        // Phil Katz, you will never be forgotten.
        assertTrue(
            // Local header
            x[i-7] == 'P' && x[i-6] == 'K' && x[i-5] == 3 && x[i-4] == 4 ||
            // Central directory header
            x[i-9] == 'P' && x[i-8] == 'K' && x[i-7] == 1 && x[i-6] == 2);
      }
    }
  }

  /**
   * Ensures that the code that grows the central directory and the code that patches it is not
   * obviously broken.
   */
  @Test
  public void testLotsOfFiles() throws IOException {
    int fileCount = 100;
    for (int blockSize : new int[] { 1, 2, 3, 4, 10, 1000 }) {
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      ZipCombiner zipCombiner = new ZipCombiner(
          OutputMode.DONT_CARE, new CopyEntryFilter(), out, blockSize);
      for (int i = 0; i < fileCount; i++) {
        zipCombiner.addFile("hello" + i, DOS_EPOCH, asStream("Hello " + i + "!"));
      }
      zipCombiner.close();
      ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()));
      for (int i = 0; i < fileCount; i++) {
        assertEntry(zipInput, "hello" + i, "Hello " + i + "!");
      }
      assertNull(zipInput.getNextEntry());
      new ZipTester(out.toByteArray()).validate();
    }
  }
}
