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

import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link PersistentMap}. */
@RunWith(JUnit4.class)
public final class PersistentMapTest {
  private static class PersistentStringMap extends PersistentMap<String, String> {
    boolean updateJournal = true;
    boolean keepJournal = false;

    PersistentStringMap(ConcurrentMap<String, String> map, Path mapFile, Path journalFile)
        throws IOException {
      super(0x0, map, mapFile, journalFile);
      load();
    }

    @Override
    protected String readKey(DataInputStream in) throws IOException {
      return in.readUTF();
    }
    @Override
    protected String readValue(DataInputStream in) throws IOException {
      return in.readUTF();
    }
    @Override
    protected void writeKey(String key, DataOutputStream out)
        throws IOException {
      out.writeUTF(key);
    }
    @Override
    protected void writeValue(String value, DataOutputStream out)
        throws IOException {
      out.writeUTF(value);
    }
    @Override
    protected boolean updateJournal() {
      return updateJournal;
    }
    @Override
    protected boolean keepJournal() {
      return keepJournal;
    }
  }

  private final Scratch scratch = new Scratch();

  private PersistentStringMap map;
  private Path mapFile;
  private Path journalFile;

  @Before
  public final void createFiles() throws Exception  {
    mapFile = scratch.resolve("/tmp/map.txt");
    journalFile = scratch.resolve("/tmp/journal.txt");
    createMap();
  }

  private void createMap() throws Exception {
    ConcurrentMap<String, String> map = new ConcurrentHashMap<>();
    this.map = new PersistentStringMap(map, mapFile, journalFile);
  }

  @Test
  public void map() throws Exception {
    createMap();
    map.put("foo", "bar");
    map.put("baz", "bang");
    assertThat(map).containsEntry("foo", "bar");
    assertThat(map).containsEntry("baz", "bang");
    assertThat(map).hasSize(2);
    long size = map.save();
    assertThat(size).isEqualTo(mapFile.getFileSize());
    assertThat(map).containsEntry("foo", "bar");
    assertThat(map).containsEntry("baz", "bang");
    assertThat(map).hasSize(2);

    createMap(); // create a new map
    assertThat(map).containsEntry("foo", "bar");
    assertThat(map).containsEntry("baz", "bang");
    assertThat(map).hasSize(2);
  }

  @Test
  public void putIfAbsent() throws Exception {
    createMap();
    assertThat(map.putIfAbsent("foo", "bar")).isNull();
    assertThat(map.putIfAbsent("foo", "ignored")).isEqualTo("bar");
    assertThat(map.putIfAbsent("baz", "bang")).isNull();
    assertThat(map.putIfAbsent("baz", "ignored")).isEqualTo("bang");
    assertThat(map).containsEntry("foo", "bar");
    assertThat(map).containsEntry("baz", "bang");
    assertThat(map).hasSize(2);
    long size = map.save();
    assertThat(size).isEqualTo(mapFile.getFileSize());
    assertThat(map).containsEntry("foo", "bar");
    assertThat(map).containsEntry("baz", "bang");
    assertThat(map).hasSize(2);

    createMap(); // create a new map
    assertThat(map).containsEntry("foo", "bar");
    assertThat(map).containsEntry("baz", "bang");
    assertThat(map).hasSize(2);
  }

  @Test
  public void remove() throws Exception {
    createMap();
    map.put("foo", "bar");
    map.put("baz", "bang");
    long size = map.save();
    assertThat(size).isEqualTo(mapFile.getFileSize());
    assertThat(journalFile.exists()).isFalse();
    map.remove("foo");
    assertThat(map).hasSize(1);
    assertThat(journalFile.exists()).isTrue();
    createMap(); // create a new map
    assertThat(map).hasSize(1);
  }

  @Test
  public void clear() throws Exception {
    createMap();
    map.put("foo", "bar");
    map.put("baz", "bang");
    map.save();
    assertThat(mapFile.exists()).isTrue();
    assertThat(journalFile.exists()).isFalse();
    map.clear();
    assertThat(map).isEmpty();
    assertThat(mapFile.exists()).isTrue();
    assertThat(journalFile.exists()).isFalse();
    createMap(); // create a new map
    assertThat(map).isEmpty();
  }

  @Test
  public void noUpdateJournal() throws Exception {
    createMap();
    map.put("foo", "bar");
    map.put("baz", "bang");
    map.save();
    assertThat(journalFile.exists()).isFalse();
    // prevent updating the journal
    map.updateJournal = false;
    // remove an entry
    map.remove("foo");
    assertThat(map).hasSize(1);
    // no journal file written
    assertThat(journalFile.exists()).isFalse();
    createMap(); // create a new map
    // both entries are still in the map on disk
    assertThat(map).hasSize(2);
  }

  @Test
  public void keepJournal() throws Exception {
    createMap();
    map.put("foo", "bar");
    map.put("baz", "bang");
    map.save();
    assertThat(journalFile.exists()).isFalse();

    // Keep the journal through the save.
    map.updateJournal = false;
    map.keepJournal = true;

    // remove an entry
    map.remove("foo");
    assertThat(map).hasSize(1);
    // no journal file written
    assertThat(journalFile.exists()).isFalse();

    long size = map.save();
    assertThat(map).hasSize(1);
    // The journal must be serialzed on save(), even if !updateJournal.
    assertThat(journalFile.exists()).isTrue();
    assertThat(size).isEqualTo(journalFile.getFileSize() + mapFile.getFileSize());

    map.load();
    assertThat(map).hasSize(1);
    assertThat(journalFile.exists()).isTrue();

    createMap(); // create a new map
    assertThat(map).hasSize(1);

    map.keepJournal = false;
    map.save();
    assertThat(map).hasSize(1);
    assertThat(journalFile.exists()).isFalse();
  }

  @Test
  public void keepJournalWithMultipleSaves() throws Exception {
    createMap();
    map.put("foo", "bar");
    map.put("baz", "bang");
    map.save();
    map.updateJournal = false;
    map.keepJournal = true;
    map.remove("foo");
    assertThat(map).hasSize(1);
    map.save();
    map.remove("baz");
    map.save();
    assertThat(map).isEmpty();
    // Ensure recreating the map loads the correct state.
    createMap();
    assertThat(map).isEmpty();
    assertThat(journalFile.exists()).isFalse();
  }

  @Test
  public void multipleJournalUpdates() throws Exception {
    createMap();
    map.put("foo", "bar");
    map.save();
    assertThat(journalFile.exists()).isFalse();
    // add an entry
    map.put("baz", "bang");
    assertThat(map).hasSize(2);
    // journal file written
    assertThat(journalFile.exists()).isTrue();
    createMap(); // create a new map
    // both entries are still in the map on disk
    assertThat(map).hasSize(2);
    // add another entry
    map.put("baz2", "bang2");
    assertThat(map).hasSize(3);
    // journal file written
    assertThat(journalFile.exists()).isTrue();
    createMap(); // create a new map
    // all three entries are still in the map on disk
    assertThat(map).hasSize(3);
  }

  @Test
  public void concurrentOperations() throws Exception {
    createMap();
    map.keepJournal = true;
    int numIters = 1000;
    TestThread fooPutter =
        new TestThread(
            () -> {
              for (int i = 0; i < numIters; i++) {
                map.put("foo", "bar" + i);
                map.remove("baz");
              }
            });
    TestThread bazPutter =
        new TestThread(
            () -> {
              for (int i = 0; i < numIters; i++) {
                map.put("baz", "bar" + i);
                map.remove("noexist");
              }
            });
    fooPutter.start();
    bazPutter.start();
    fooPutter.joinAndAssertState(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
    bazPutter.joinAndAssertState(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
    map.save();
    assertThat(journalFile.exists()).isTrue();
    createMap();
    assertThat(map).containsEntry("foo", "bar" + (numIters - 1));
    String bazValue = map.get("baz");
    if (bazValue != null) {
      assertThat(bazValue).isEqualTo("bar" + (numIters - 1));
    }
  }
}
