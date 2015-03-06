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
package com.google.devtools.build.lib.actions.cache;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.util.FsApparatus;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.EOFException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;

/**
 * Test for the PersistentStringIndexer class.
 */
@RunWith(JUnit4.class)
public class PersistentStringIndexerTest {

  private static class ManualClock implements Clock {
    private long currentTime = 0L;

    ManualClock() { }

    @Override public long currentTimeMillis() {
      throw new AssertionError("unexpected method call");
    }

    @Override  public long nanoTime() {
      return currentTime;
    }

    void advance(long time) {
      currentTime += time;
    }
  }

  private PersistentStringIndexer psi;
  private Map<Integer, String> mappings = new ConcurrentHashMap<>();
  private FsApparatus scratch = FsApparatus.newInMemory();
  private ManualClock clock = new ManualClock();
  private Path dataPath;
  private Path journalPath;


  @Before
  public void setUp() throws Exception {
    dataPath = scratch.path("/cache/test.dat");
    journalPath = scratch.path("/cache/test.journal");
    psi = PersistentStringIndexer.newPersistentStringIndexer(dataPath, clock);
  }

  private void assertSize(int expected) {
    assertEquals(expected, psi.size());
  }

  private void assertIndex(int expected, String s) {
    int index = psi.getOrCreateIndex(s);
    assertEquals(expected, index);
    mappings.put(expected, s);
  }

  private void assertContent() {
    for (int i = 0; i < psi.size(); i++) {
      if(mappings.get(i) != null) {
        assertThat(mappings).containsEntry(i, psi.getStringForIndex(i));
      }
    }
  }


  private void setupTestContent() {
    assertSize(0);
    assertIndex(0, "abcdefghi");  // Create leafs
    assertIndex(1, "abcdefjkl");
    assertIndex(2, "abcdefmno");
    assertIndex(3, "abcdefjklpr");
    assertIndex(3, "abcdefjklpr");
    assertIndex(4, "abcdstr");
    assertIndex(5, "012345");
    assertSize(6);
    assertIndex(6, "abcdef");  // Validate inner nodes
    assertIndex(7, "abcd");
    assertIndex(8, "");
    assertSize(9);
    assertContent();
  }

  /**
   * Writes lots of entries with labels "fooconcurrent[int]" at the same time.
   * The set of labels written is deterministic, but the label:index mapping is
   * not.
   */
  private void writeLotsOfEntriesConcurrently(final int numToWrite) throws InterruptedException {
    final int NUM_THREADS = 10;
    final CountDownLatch synchronizerLatch = new CountDownLatch(NUM_THREADS);

    class IndexAdder extends TestThread {
      @Override
      public void runTest() throws Exception {
        for (int i = 0; i < numToWrite; i++) {
          synchronizerLatch.countDown();
          synchronizerLatch.await();

          String value = "fooconcurrent" + i;
          mappings.put(psi.getOrCreateIndex(value), value);
        }
      }
    }

    Collection<TestThread> threads = new ArrayList<>();
    for (int i = 0; i < NUM_THREADS; i++) {
      TestThread thread = new IndexAdder();
      thread.start();
      threads.add(thread);
    }

    for (TestThread thread : threads) {
      thread.joinAndAssertState(0);
    }
  }

  @Test
  public void testNormalOperation() throws Exception {
    assertFalse(dataPath.exists());
    assertFalse(journalPath.exists());
    setupTestContent();
    assertFalse(dataPath.exists());
    assertFalse(journalPath.exists());

    clock.advance(4);
    assertIndex(9, "xyzqwerty"); // This should flush journal to disk.
    assertFalse(dataPath.exists());
    assertTrue(journalPath.exists());

    psi.save(); // Successful save will remove journal file.
    assertTrue(dataPath.exists());
    assertFalse(journalPath.exists());

    // Now restore data from file and verify it.
    psi = PersistentStringIndexer.newPersistentStringIndexer(dataPath, clock);
    assertFalse(journalPath.exists());
    clock.advance(4);
    assertSize(10);
    assertContent();
    assertFalse(journalPath.exists());
  }

  @Test
  public void testJournalRecoveryWithoutMainDataFile() throws Exception {
    assertFalse(dataPath.exists());
    assertFalse(journalPath.exists());
    setupTestContent();
    assertFalse(dataPath.exists());
    assertFalse(journalPath.exists());

    clock.advance(4);
    assertIndex(9, "abc1234"); // This should flush journal to disk.
    assertFalse(dataPath.exists());
    assertTrue(journalPath.exists());

    // Now restore data from file and verify it. All data should be restored from journal;
    psi = PersistentStringIndexer.newPersistentStringIndexer(dataPath, clock);
    assertTrue(dataPath.exists());
    assertFalse(journalPath.exists());
    clock.advance(4);
    assertSize(10);
    assertContent();
    assertFalse(journalPath.exists());
  }

  @Test
  public void testJournalRecovery() throws Exception {
    assertFalse(dataPath.exists());
    assertFalse(journalPath.exists());
    setupTestContent();
    psi.save();
    assertTrue(dataPath.exists());
    assertFalse(journalPath.exists());
    long oldDataFileLen = dataPath.getFileSize();

    clock.advance(4);
    assertIndex(9, "another record"); // This should flush journal to disk.
    assertSize(10);
    assertTrue(dataPath.exists());
    assertTrue(journalPath.exists());

    // Now restore data from file and verify it. All data should be restored from journal;
    psi = PersistentStringIndexer.newPersistentStringIndexer(dataPath, clock);
    assertTrue(dataPath.exists());
    assertFalse(journalPath.exists());
    assertTrue(dataPath.getFileSize() > oldDataFileLen); // data file should have been updated
    clock.advance(4);
    assertSize(10);
    assertContent();
    assertFalse(journalPath.exists());
  }

  @Test
  public void testConcurrentWritesJournalRecovery() throws Exception {
    assertFalse(dataPath.exists());
    assertFalse(journalPath.exists());
    setupTestContent();
    psi.save();
    assertTrue(dataPath.exists());
    assertFalse(journalPath.exists());
    long oldDataFileLen = dataPath.getFileSize();

    int size = psi.size();
    int numToWrite = 50000;
    writeLotsOfEntriesConcurrently(numToWrite);
    assertFalse(journalPath.exists());
    clock.advance(4);
    assertIndex(size + numToWrite, "another record"); // This should flush journal to disk.
    assertSize(size + numToWrite + 1);
    assertTrue(dataPath.exists());
    assertTrue(journalPath.exists());

    // Now restore data from file and verify it. All data should be restored from journal;
    psi = PersistentStringIndexer.newPersistentStringIndexer(dataPath, clock);
    assertTrue(dataPath.exists());
    assertFalse(journalPath.exists());
    assertTrue(dataPath.getFileSize() > oldDataFileLen); // data file should have been updated
    clock.advance(4);
    assertSize(size + numToWrite + 1);
    assertContent();
    assertFalse(journalPath.exists());
  }

  @Test
  public void testCorruptedJournal() throws Exception {
    FileSystemUtils.createDirectoryAndParents(journalPath.getParentDirectory());
    FileSystemUtils.writeContentAsLatin1(journalPath, "bogus content");
    try {
      psi = PersistentStringIndexer.newPersistentStringIndexer(dataPath, clock);
      fail();
    } catch (IOException e) {
      assertThat(e.getMessage()).contains("too short: Only 13 bytes");
    }

    journalPath.delete();
    setupTestContent();
    assertFalse(dataPath.exists());
    assertFalse(journalPath.exists());

    clock.advance(4);
    assertIndex(9, "abc1234"); // This should flush journal to disk.
    assertFalse(dataPath.exists());
    assertTrue(journalPath.exists());

    byte[] journalContent = FileSystemUtils.readContent(journalPath);

    // Now restore data from file and verify it. All data should be restored from journal;
    psi = PersistentStringIndexer.newPersistentStringIndexer(dataPath, clock);
    assertTrue(dataPath.exists());
    assertFalse(journalPath.exists());

    // Now put back truncated journal. We should get an error.
    assertTrue(dataPath.delete());
    FileSystemUtils.writeContent(journalPath,
        Arrays.copyOf(journalContent, journalContent.length - 1));
    try {
      psi = PersistentStringIndexer.newPersistentStringIndexer(dataPath, clock);
      fail();
    } catch (EOFException e) {
      // Expected.
    }

    // Corrupt the journal with a negative size value.
    byte[] journalCopy = journalContent.clone();
    // Flip this bit to make the key size negative.
    journalCopy[95] = -2;
    FileSystemUtils.writeContent(journalPath,  journalCopy);
    try {
      psi = PersistentStringIndexer.newPersistentStringIndexer(dataPath, clock);
      fail();
    } catch (IOException e) {
      // Expected.
      assertThat(e.getMessage()).contains("corrupt key length");
    }

    // Now put back corrupted journal. We should get an error.
    journalContent[journalContent.length - 13] = 100;
    FileSystemUtils.writeContent(journalPath,  journalContent);
    try {
      psi = PersistentStringIndexer.newPersistentStringIndexer(dataPath, clock);
      fail();
    } catch (IOException e) {
      // Expected.
    }
  }

  @Test
  public void testDupeIndexCorruption() throws Exception {
    setupTestContent();
    assertFalse(dataPath.exists());
    assertFalse(journalPath.exists());

    assertIndex(9, "abc1234"); // This should flush journal to disk.
    psi.save();
    assertTrue(dataPath.exists());
    assertFalse(journalPath.exists());

    byte[] content = FileSystemUtils.readContent(dataPath);

    // We remove the data file, and instead create a corrupt journal.
    //
    // The journal has a header followed by a sequence of (String, int) pairs, where each int is a
    // unique value. The String is encoded by the length (as an int), and the int is simply encoded
    // as an int. Note that the DataOutputStream class uses big endian by default, so the low-order
    // bits are at the end.
    //
    // For the purpose of this test, we want to make the journal contain two entries with the same
    // index (which is illegal). The PersistentStringIndexer assigns int values in the usual order,
    // starting with zero, and it now contains 9 entries. We simply change the last entry to an
    // index that is guaranteed to already exist. If it is the index 1, we change it to 2, otherwise
    // we change it to 1 - in both cases, the code currently guarantees that the duplicate comes
    // earlier in the stream.
    assertTrue(dataPath.delete());
    content[content.length - 1] = content[content.length - 1] == 1 ? (byte) 2 : (byte) 1;
    FileSystemUtils.writeContent(journalPath, content);

    try {
      psi = PersistentStringIndexer.newPersistentStringIndexer(dataPath, clock);
      fail();
    } catch (IOException e) {
      // Expected.
      assertThat(e.getMessage()).contains("Corrupted filename index has duplicate entry");
    }
  }

  @Test
  public void testDeferredIOFailure() throws Exception {
    assertFalse(dataPath.exists());
    assertFalse(journalPath.exists());
    setupTestContent();
    assertFalse(dataPath.exists());
    assertFalse(journalPath.exists());

    // Ensure that journal cannot be saved.
    FileSystemUtils.createDirectoryAndParents(journalPath);

    clock.advance(4);
    assertIndex(9, "abc1234"); // This should flush journal to disk (and fail at that).
    assertFalse(dataPath.exists());

    // Subsequent updates should succeed even though journaling is disabled at this point.
    clock.advance(4);
    assertIndex(10, "another record");
    try {
      // Save should actually save main data file but then return us deferred IO failure
      // from failed journal write.
      psi.save();
      fail();
    } catch(IOException e) {
      assertThat(e.getMessage()).contains(journalPath.getPathString() + " (Is a directory)");
    }
  }
}
