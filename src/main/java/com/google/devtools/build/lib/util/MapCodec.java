// Copyright 2025 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.vfs.Path;
import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import javax.annotation.Nullable;

/** Converts map entries between their in-memory and on-disk representations. */
public abstract class MapCodec<K, V> {
  /** Exception thrown when persisted data read back from disk is in an incompatible format. */
  public static final class IncompatibleFormatException extends IOException {
    public IncompatibleFormatException(String message) {
      super(message);
    }
  }

  private static final int MAGIC = 0x20071105;
  private static final int MIN_MAPFILE_SIZE = 16;
  private static final int MAX_ARRAY_SIZE = Integer.MAX_VALUE - 8;
  private static final int ENTRY_MAGIC = 0xfe;

  /** Reads a key from a {@link DataInputStream}. */
  protected abstract K readKey(DataInputStream in) throws IOException;

  /** Reads a value from a {@link DataInputStream}. */
  protected abstract V readValue(DataInputStream in) throws IOException;

  /** Writes a key into a {@link DataOutputStream}. */
  protected abstract void writeKey(K key, DataOutputStream out) throws IOException;

  /** Writes a value into a {@link DataOutputStream}. */
  protected abstract void writeValue(V value, DataOutputStream out) throws IOException;

  /**
   * A key/value pair representing the presence or absence of a map entry.
   *
   * @param key the entry key
   * @param value the entry value, or null if the entry is absent
   */
  public record Entry<K, V>(K key, @Nullable V value) {}

  /**
   * Creates a new reader.
   *
   * <p>The file contents are eagerly read into memory, under the assumption that they will be
   * iterated to completion in short order.
   *
   * @param path the path to the file to read
   * @param version the expected version number
   * @throws IncompatibleFormatException if the on-disk data is in an incompatible format
   * @throws IOException if some other I/O error occurs
   */
  public Reader createReader(Path path, long version) throws IOException {
    long size = path.getFileSize();
    if (size < MIN_MAPFILE_SIZE) {
      throw new IncompatibleFormatException("%s is too short: %s bytes".formatted(path, size));
    }
    if (size > MAX_ARRAY_SIZE) {
      throw new IncompatibleFormatException("%s is too long: %s bytes".formatted(path, size));
    }

    // Read the whole file upfront as a performance optimization (minimize syscalls).
    byte[] bytes;
    try (InputStream in = path.getInputStream()) {
      bytes = in.readAllBytes();
    }

    DataInputStream in = new DataInputStream(new ByteArrayInputStream(bytes));

    if (in.readLong() != MAGIC) {
      // Not a PersistentMap.
      throw new IncompatibleFormatException("Bad magic number");
    }
    long persistedVersion = in.readLong();
    if (persistedVersion != version) {
      // Incompatible version.
      throw new IncompatibleFormatException(
          "Incompatible version: want %d, got %d".formatted(version, persistedVersion));
    }

    return new Reader(in);
  }

  /** Reads key/value pairs from a {@link DataInputStream}. */
  public final class Reader implements AutoCloseable {
    private final DataInputStream in;

    private Reader(DataInputStream in) {
      this.in = in;
    }

    /** Closes the reader, releasing associated resources and rendering it unusable. */
    @Override
    public void close() throws IOException {
      in.close();
    }

    /**
     * Reads an {@link Entry}.
     *
     * @return the entry, or null if there are no more entries.
     */
    @Nullable
    public Entry<K, V> readEntry() throws IOException {
      if (in.available() == 0) {
        return null;
      }
      if (in.readUnsignedByte() != ENTRY_MAGIC) {
        throw new IOException("Corrupted entry separator");
      }
      K key = readKey(in);
      boolean hasValue = in.readBoolean();
      V value = hasValue ? readValue(in) : null;
      return new Entry<>(key, value);
    }
  }

  /**
   * Creates a new writer.
   *
   * @param path the path to the file to write
   * @param version the version number to write
   * @param overwrite whether to overwrite an existing file instead of appending to it
   * @throws IOException if an I/O error occurs
   */
  public Writer createWriter(Path path, int version, boolean overwrite) throws IOException {
    DataOutputStream out;
    if (!overwrite && path.exists()) {
      out =
          new DataOutputStream(new BufferedOutputStream(path.getOutputStream(/* append= */ true)));
    } else {
      out = new DataOutputStream(new BufferedOutputStream(path.getOutputStream()));
      out.writeLong(MAGIC);
      out.writeLong(version);
    }
    return new Writer(out);
  }

  /** Writes key/value pairs to a {@link DataOutputStream}. */
  public final class Writer implements AutoCloseable {
    private final DataOutputStream out;

    private Writer(DataOutputStream out) {
      this.out = out;
    }

    /** Flushes the writer, forcing any pending writes to be written to disk. */
    public void flush() throws IOException {
      out.flush();
    }

    /** Closes the writer, releasing associated resources and rendering it unusable. */
    @Override
    public void close() throws IOException {
      out.close();
    }

    /**
     * Writes a key/value pair.
     *
     * @param key the key to write.
     * @param value the value to write, or null to write a tombstone.
     */
    public void writeEntry(K key, @Nullable V value) throws IOException {
      out.writeByte(ENTRY_MAGIC);
      writeKey(key, out);
      boolean hasValue = value != null;
      out.writeBoolean(hasValue);
      if (hasValue) {
        writeValue(value, out);
      }
    }
  }
}
