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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for the {@link PersistentMap}.
 */
@RunWith(JUnit4.class)
public class PersistentMapTest {
  public static class PersistentStringMap extends PersistentMap<String, String> {
    boolean updateJournal = true;
    boolean keepJournal = false;

    public PersistentStringMap(Map<String, String> map, Path mapFile,
        Path journalFile) throws IOException {
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

  private Scratch scratch = new Scratch();

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
    Map<String, String> map = new HashMap<>();
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
    assertEquals(mapFile.getFileSize(), size);
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
    assertEquals(mapFile.getFileSize(), size);
    assertFalse(journalFile.exists());
    map.remove("foo");
    assertThat(map).hasSize(1);
    assertTrue(journalFile.exists());
    createMap(); // create a new map
    assertThat(map).hasSize(1);
  }

  @Test
  public void clear() throws Exception {
    createMap();
    map.put("foo", "bar");
    map.put("baz", "bang");
    map.save();
    assertTrue(mapFile.exists());
    assertFalse(journalFile.exists());
    map.clear();
    assertThat(map).isEmpty();
    assertTrue(mapFile.exists());
    assertFalse(journalFile.exists());
    createMap(); // create a new map
    assertThat(map).isEmpty();
  }

  @Test
  public void noUpdateJournal() throws Exception {
    createMap();
    map.put("foo", "bar");
    map.put("baz", "bang");
    map.save();
    assertFalse(journalFile.exists());
    // prevent updating the journal
    map.updateJournal = false;
    // remove an entry
    map.remove("foo");
    assertThat(map).hasSize(1);
    // no journal file written
    assertFalse(journalFile.exists());
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
    assertFalse(journalFile.exists());

    // Keep the journal through the save.
    map.updateJournal = false;
    map.keepJournal = true;

    // remove an entry
    map.remove("foo");
    assertThat(map).hasSize(1);
    // no journal file written
    assertFalse(journalFile.exists());

    long size = map.save();
    assertThat(map).hasSize(1);
    // The journal must be serialzed on save(), even if !updateJournal.
    assertTrue(journalFile.exists());
    assertEquals(journalFile.getFileSize() + mapFile.getFileSize(), size);

    map.load();
    assertThat(map).hasSize(1);
    assertTrue(journalFile.exists());

    createMap(); // create a new map
    assertThat(map).hasSize(1);

    map.keepJournal = false;
    map.save();
    assertThat(map).hasSize(1);
    assertFalse(journalFile.exists());
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
    assertThat(map).hasSize(0);
    // Ensure recreating the map loads the correct state.
    createMap();
    assertThat(map).hasSize(0);
    assertFalse(journalFile.exists());
  }

  @Test
  public void multipleJournalUpdates() throws Exception {
    createMap();
    map.put("foo", "bar");
    map.save();
    assertFalse(journalFile.exists());
    // add an entry
    map.put("baz", "bang");
    assertThat(map).hasSize(2);
    // journal file written
    assertTrue(journalFile.exists());
    createMap(); // create a new map
    // both entries are still in the map on disk
    assertThat(map).hasSize(2);
    // add another entry
    map.put("baz2", "bang2");
    assertThat(map).hasSize(3);
    // journal file written
    assertTrue(journalFile.exists());
    createMap(); // create a new map
    // all three entries are still in the map on disk
    assertThat(map).hasSize(3);
  }
}
