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

package com.google.devtools.build.zip;

import java.io.IOException;
import java.io.InputStream;
import java.util.Iterator;
import java.util.LinkedHashMap;

/**
 * A list of {@link ExtraData} records to be associated with a {@link ZipFileEntry}. Supports
 * creating the list directly from a byte array and modifying the list without reallocating the
 * underlying buffer. 
 */
public class ExtraDataList {
  public static final short ZIP64 = 0x0001;
  public static final short EXTENDED_TIMESTAMP = 0x5455;
  // Some documentation says that this is actually 0x7855, but zip files do not seem to corroborate
  // this
  public static final short INFOZIP_UNIX_NEW = 0x7875;
  private final LinkedHashMap<Short, ExtraData> entries;

  /**
   * Create a new empty extra data list.
   *
   * <p><em>NOTE:</em> entries in a list created this way will be backed by their own storage.
   */
  public ExtraDataList() {
    entries = new LinkedHashMap<>();
  }

  public ExtraDataList(ExtraDataList other) {
    this.entries = new LinkedHashMap<>();
    this.entries.putAll(other.entries);
  }

  /**
   * Creates an extra data list from the given extra data records.
   *
   * <p><em>NOTE:</em> entries in a list created this way will be backed by their own storage.
   *
   * @param extra the extra data records
   */
  public ExtraDataList(ExtraData... extra) {
    this();
    for (ExtraData e : extra) {
      add(e);
    }
  }

  /**
   * Creates an extra data list from the entries contained in the given array.
   *
   * <p><em>NOTE:</em> entries in a list created this way will be backed by the buffer. No defensive
   * copying is performed.
   *
   * @param buffer the array containing sequential extra data entries
   */
  public ExtraDataList(byte[] buffer) {
    if (buffer.length > 0xffff) {
      throw new IllegalArgumentException("invalid extra field length");
    }
    entries = new LinkedHashMap<>();
    int index = 0;
    while (index < buffer.length) {
      ExtraData extra = new ExtraData(buffer, index);
      entries.put(extra.getId(), extra);
      index += extra.getLength();
    }
  }

  /**
   * Returns the extra data record with the specified id, or null if it does not exist.
   */
  public ExtraData get(short id) {
    return entries.get(id);
  }

  /**
   * Removes and returns the extra data record with the specified id if it exists.
   *
   * <p><em>NOTE:</em> does not modify the underlying storage, only marks the record as removed.
   */
  public ExtraData remove(short id) {
    return entries.remove(id);
  }

  /**
   * Returns if the list contains an extra data record with the specified id.
   */
  public boolean contains(short id) {
    return entries.containsKey(id);
  }

  /**
   * Adds a new entry to the end of the list.
   *
   * @throws IllegalArgumentException if adding the entry will make the list too long for the ZIP
   *    format
   */
  public void add(ExtraData entry) {
    if (getLength() + entry.getLength() > 0xffff) {
      throw new IllegalArgumentException("adding entry will make the extra field be too long");
    }
    entries.put(entry.getId(), entry);
  }

  /**
   * Returns the overall length of the list in bytes.
   */
  public int getLength() {
    int length = 0;
    for (ExtraData e : entries.values()) {
      length += e.getLength();
    }
    return length;
  }

  /**
   * Creates and returns a byte array of the extra data list.
   */
  public byte[] getBytes() {
    byte[] extra = new byte[getLength()];
    try {
      getByteStream().read(extra);
    } catch (IOException impossible) {
      throw new AssertionError(impossible);
    }
    return extra;
  }

  /**
   * Returns an input stream for reading the extra data list entries.
   */
  public InputStream getByteStream() {
    return new InputStream() {
      private final Iterator<ExtraData> itr = entries.values().iterator();
      private ExtraData entry;
      private int index;

      @Override
      public int read() {
        if (entry == null) {
          if (itr.hasNext()) {
            entry = itr.next();
            index = 0;
          } else {
            return -1;
          }
        }
        byte val = entry.getByte(index++);
        if (index >= entry.getLength()) {
          entry = null;
        }
        return val & 0xff;
      }
    };
  }
}
