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
package com.google.devtools.build.android.ziputils;

import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENOFF;

import com.google.common.base.Preconditions;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.NavigableMap;
import java.util.TreeMap;

/**
 * Provides a view of a zip file's central directory. For reading, a single memory mapped view is
 * used. For writing, the central directory is stored as one or more views, each backed by a direct
 * byte buffer.
 */
public class CentralDirectory extends View<CentralDirectory> {

  // Cached map from entry name to directory entry.
  private NavigableMap<String, DirectoryEntry> mapByNameSorted;
  // Cached map from entry file offset to directory entry.
  private NavigableMap<Integer, DirectoryEntry> mapByOffsetSorted;
  // Number of directory entries in this view.
  private int count;
  // Parsed or added entries
  private List<DirectoryEntry> entries;

  /**
   * Gets the number of directory entries in this view.
   */
  public int getCount() {
    return count;
  }

  /**
   * Returns a list of directory entries, in the order they occur in the central directory.
   * This will typically also be the order of entries in the zip file, although that's not
   * guaranteed.
   */
  public List<DirectoryEntry> list() {
    return entries;
  }

  /**
   * Returns a navigable map of directory entries, by zip entry file offset.
   */
  public NavigableMap<Integer, DirectoryEntry> mapByOffset() {
    if (entries == null) {
      return null;
    }
    return mapEntriesByOffset();
  }

  /**
   * Returns a navigable map of directory entries, by entry filename.
   */
  public NavigableMap<String, DirectoryEntry> mapByFilename() {
    if (entries == null) {
      return null;
    }
    return mapEntriesByName();
  }

  /**
   * Returns a {@code CentralDirectory} of the given buffer. This may be a full or a partial
   * central directory. This method assumes ownership of the underlying buffer. Unlike most
   * "view-of" methods, this method doesn't slice the argument buffer, and rather than advancing
   * the buffer position, it sets it to 0.
   *
   * @param buffer containing data of a central directory.
   * @return a {@code CentralDirectory} of the data at the current position of the given byte
   * buffer.
   */
  public static CentralDirectory viewOf(ByteBuffer buffer) {
    buffer.position(0);
    return new CentralDirectory(buffer);
  }

  private CentralDirectory(ByteBuffer buffer) {
    super(buffer);
    count = -1;
  }

  /**
   * Parses this central directory, and maps the contained entries with {@link DirectoryEntry}s.
   *
   * @return this central directory view
   * @throws IllegalStateException if the file offset is not set prior to parsing
   */
  public CentralDirectory parse() throws IllegalStateException {
    Preconditions.checkState(fileOffset != -1, "File offset not set prior to parsing");
    count = 0;
    clearMaps();
    int relPos = 0;
    buffer.position(0);
    while (buffer.hasRemaining() && buffer.getInt(buffer.position()) == DirectoryEntry.SIGNATURE) {
      count++;
      DirectoryEntry entry = DirectoryEntry.viewOf(buffer).at(fileOffset + relPos);
      entries.add(entry);
      relPos += entry.getSize();
      buffer.position(relPos);
    }
    return this;
  }

  /**
   * Creates a new directory entry for output. The given entry is copied into the buffer of this
   * central directory, and a view of the copied data is returned.
   *
   * @param entry prototype directory entry, typically an entry read from another zip file, for
   * an entry being copied.
   * @return a directory entry view of the copied entry.
   */
  public DirectoryEntry nextEntry(DirectoryEntry entry) {
    DirectoryEntry clone = entry.copy(buffer);
    if (count == -1) {
      clearMaps();
      count = 1;
    } else {
      count++;
    }
    entries.add(clone);
    return clone;
  }

  private NavigableMap<String, DirectoryEntry> mapEntriesByName() {
    if (mapByNameSorted == null) {
      mapByNameSorted = new TreeMap<>();
      for (DirectoryEntry entry : entries) {
        mapByNameSorted.put(entry.getFilename(), entry);
      }
    }
    return mapByNameSorted;
  }

  private NavigableMap<Integer, DirectoryEntry> mapEntriesByOffset() {
    if (mapByOffsetSorted == null) {
      mapByOffsetSorted = new TreeMap<>();
      for (DirectoryEntry entry : entries) {
        mapByOffsetSorted.put(entry.get(CENOFF), entry);
      }
    }
    return mapByOffsetSorted;
  }

  private void clearMaps() {
    entries = new ArrayList<>();
    mapByOffsetSorted = null;
    mapByNameSorted = null;
  }
}
