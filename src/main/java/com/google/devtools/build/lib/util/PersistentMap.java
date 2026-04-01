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

import com.google.common.collect.ForwardingConcurrentMap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.MapCodec.IncompatibleFormatException;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.LinkedBlockingQueue;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/**
 * A map that is backed by persistent storage. It uses two files on disk for this: The first file
 * contains all the entries and gets written when invoking the {@link #save()} method. The second
 * file contains a journal of all entries that were added to or removed from the map since
 * constructing the instance of the map or the last invocation of {@link #save()} and gets written
 * after each update of the map although sub-classes are free to implement their own journal update
 * strategy.
 *
 * <p><b>Ceci n'est pas un Map</b>. Strictly speaking, the {@link Map} interface doesn't permit the
 * possibility of failure. This class uses persistence; persistence means I/O, and I/O means the
 * possibility of failure. Therefore the semantics of this may deviate from the Map contract in
 * failure cases. In particular, updates are not guaranteed to succeed. However, I/O failures are
 * guaranteed to be reported upon the subsequent call to a method that throws {@code IOException}
 * such as {@link #save}.
 *
 * <p>To populate the map entries using the previously persisted entries call {@link #load()} prior
 * to invoking any other map operation.
 *
 * <p>Like {@link Hashtable} but unlike {@link HashMap}, this class does <em>not</em> allow
 * <tt>null</tt> to be used as a key or a value.
 *
 * <p>IO failures during reading or writing the map entries to disk may result in {@link
 * AssertionError} getting thrown from the failing method.
 *
 * <p>The constructor allows passing in a version number that gets written to the files on disk and
 * checked before reading from disk. Files with an incompatible version number will be ignored. This
 * allows the client code to change the persistence format without polluting the file system name
 * space.
 */
public abstract class PersistentMap<K, V> extends ForwardingConcurrentMap<K, V> {
  private final int version;

  @GuardedBy("this")
  private final Path mapFile;

  @GuardedBy("this")
  private final Path journalFile;

  private final LinkedBlockingQueue<K> journal;

  /**
   * If non-null, contains the message from an {@code IOException} thrown by a previously failed
   * write. This error is deferred until the next call to a method which is able to throw an
   * exception.
   */
  @GuardedBy("this")
  @Nullable
  private String deferredIOFailure = null;

  /**
   * 'loaded' is true when the in-memory representation is at least as recent as the on-disk
   * representation.
   */
  private boolean loaded;

  private final ConcurrentMap<K, V> delegate;

  private final MapCodec<K, V> codec;

  /**
   * Creates a new PersistentMap instance using the specified backing map.
   *
   * @param version the version tag. Changing the version tag allows updating the on disk format.
   *     The map will never read from a file that was written using a different version tag.
   * @param codec the codec used to convert between the in-memory and on-disk representations.
   * @param map the backing map to use for this PersistentMap.
   * @param mapFile the file to save the map entries to.
   * @param journalFile the journal file to write entries between invocations of {@link #save()}.
   */
  protected PersistentMap(
      int version, MapCodec<K, V> codec, ConcurrentMap<K, V> map, Path mapFile, Path journalFile) {
    this.version = version;
    this.codec = codec;
    journal = new LinkedBlockingQueue<>();
    this.mapFile = mapFile;
    this.journalFile = journalFile;
    delegate = map;
  }

  @Override
  protected final ConcurrentMap<K, V> delegate() {
    return delegate;
  }

  @ThreadSafe
  @Override
  @Nullable
  public V put(K key, V value) {
    V previous = delegate.put(key, value);
    journal.add(key);
    maybeFlushJournal();
    return previous;
  }

  @ThreadSafe
  @Override
  @Nullable
  public V putIfAbsent(K key, V value) {
    V previous = delegate.putIfAbsent(key, value);
    if (previous == null) {
      journal.add(key);
      maybeFlushJournal();
    }
    return previous;
  }

  /**
   * Potentially flushes the in-memory journal to disk, as determined by {@link
   * #shouldFlushJournal()}.
   */
  @ThreadSafe
  private void maybeFlushJournal() {
    if (shouldFlushJournal()) {
      flushJournal();
    }
  }

  /**
   * Determines whether the in-memory journal should be flushed to disk.
   *
   * <p>Called whenever an update is appended to the in-memory journal. The default is to flush it
   * immediately, but subclasses are free to override this to implement their own strategy.
   */
  protected boolean shouldFlushJournal() {
    return true;
  }

  @ThreadSafe
  @Override
  @SuppressWarnings("unchecked")
  @Nullable
  public V remove(Object object) {
    V previous = delegate.remove(object);
    if (previous != null) {
      // we know that 'object' must be an instance of K, because the
      // remove call succeeded, i.e. 'object' was mapped to 'previous'.
      journal.add((K) object); // unchecked
      maybeFlushJournal();
    }
    return previous;
  }

  @Override
  public V replace(K key, V value) {
    throw new UnsupportedOperationException();
  }

  @Override
  public boolean replace(K key, V oldValue, V newValue) {
    throw new UnsupportedOperationException();
  }

  /** Flushes the in-memory journal to disk. */
  public synchronized void flushJournal() {
    // Append to a preexisting journal file, which may have been left around after the last save()
    // because shouldKeepJournal() was true.
    try (var journalOut = codec.createWriter(journalFile, version, /* overwrite= */ false)) {
      // Journal may have duplicates, we can ignore them.
      LinkedHashSet<K> keys = Sets.newLinkedHashSetWithExpectedSize(journal.size());
      journal.drainTo(keys);
      writeEntries(journalOut, keys);
    } catch (IOException e) {
      this.deferredIOFailure = e.getMessage() + " during journal append";
    }
  }

  /**
   * Loads the previous written map entries from disk.
   *
   * <p>If no on-disk state exists, loading is successful and produces an empty map.
   *
   * <p>Data corruption is handled differently for each file:
   *
   * <ul>
   *   <li>Corruption in the map file is treated as an error, as the file is updated atomically.
   *   <li>Corruption in the journal file is tolerated by ignoring the remaining contents, as the
   *       file is updated non-atomically.
   *
   * @throws IncompatibleFormatException if the on-disk data is in an incompatible format
   * @throws IOException if data corruption is detected and cannot be recovered from, or some other
   *     I/O error occurs
   */
  public synchronized void load() throws IOException {
    if (!loaded) {
      if (mapFile.exists()) {
        loadEntries(mapFile);
      }
      if (journalFile.exists()) {
        try {
          loadEntries(journalFile);
        } catch (IOException e) {
          if (e instanceof IncompatibleFormatException) {
            throw e;
          }
        }
        // Merge the journal into the map file and delete the former, ensuring that we don't keep
        // appending to a corrupted journal.
        // TODO(tjgq): Avoid doing this unless journal corruption was detected.
        save(/* fullSave= */ true);
      }
      loaded = true;
    }
  }

  @Override
  public synchronized void clear() {
    super.clear();
    try {
      // We must do a full save because we're bypassing the journal.
      save(/* fullSave= */ true);
    } catch (IOException e) {
      this.deferredIOFailure = e.getMessage() + " during map clear";
    }
  }

  /**
   * Saves the map to disk.
   *
   * @throws IOException if there was an I/O error during this call, or any previous call since the
   *     last save().
   */
  public synchronized long save() throws IOException {
    return save(false);
  }

  /**
   * Saves the map to disk.
   *
   * @param fullSave if true, the journal file will be merged into the map file and deleted;
   *     otherwise, the decision is made by {@link #shouldKeepJournal()}.
   * @throws IOException if there was an I/O error during this call, or any previous call since the
   *     last save().
   */
  private synchronized long save(boolean fullSave) throws IOException {
    /* Report a previously failing I/O operation. */
    if (deferredIOFailure != null) {
      try {
        throw new IOException(deferredIOFailure);
      } finally {
        deferredIOFailure = null;
      }
    }
    if (!fullSave && shouldKeepJournal()) {
      flushJournal();
    } else {
      Path mapTemp =
          mapFile.getRelative(FileSystemUtils.replaceExtension(mapFile.asFragment(), ".tmp"));
      try {
        saveEntries(mapTemp);
        mapFile.delete();
        mapTemp.renameTo(mapFile);
      } finally {
        mapTemp.delete();
      }
      clearJournal();
      journalFile.delete();
    }
    return journalSize() + cacheSize();
  }

  protected final synchronized long journalSize() throws IOException {
    return journalFile.exists() ? journalFile.getFileSize() : 0;
  }

  protected final synchronized long cacheSize() throws IOException {
    return mapFile.exists() ? mapFile.getFileSize() : 0;
  }

  /**
   * Whether to keep the journal file on save.
   *
   * <p>The default is to always merge the journal file into the main file and delete the former,
   * but subclasses are free to override this to implement their own strategy.
   */
  protected boolean shouldKeepJournal() {
    return false;
  }

  private synchronized void clearJournal() {
    journal.clear();
  }

  /**
   * Loads all entries from the given file into the backing map.
   *
   * @throws IncompatibleFormatException if the file is in an incompatible format
   * @throws IOException if the file is corrupted or an I/O error occurs
   */
  private synchronized void loadEntries(Path mapFile) throws IOException {
    try (MapCodec<K, V>.Reader in = codec.createReader(mapFile, version)) {
      readEntries(in);
    }
  }

  /** Saves all backing map entries to the given file, overwriting preexisting contents. */
  private synchronized void saveEntries(Path mapFile) throws IOException {
    try (MapCodec<K, V>.Writer out = codec.createWriter(mapFile, version, /* overwrite= */ true)) {
      writeEntries(out, null);
    }
  }

  /**
   * Writes backing map entries for a set of keys into a {@link MapCodec.Writer}.
   *
   * @param out the {@link MapCodec.Writer} to write to.
   * @param keys the keys that are to be written, or null to write all keys.
   * @throws IOException
   */
  private void writeEntries(MapCodec<K, V>.Writer out, @Nullable Set<K> keys) throws IOException {
    Map<K, V> map = delegate();
    for (K key : keys != null ? keys : map.keySet()) {
      out.writeEntry(key, map.get(key));
    }
  }

  /**
   * Reads entries from a {@link MapCodec.Reader} into the backing map.
   *
   * @param in the {@link MapCodec.Reader} to read from.
   */
  private void readEntries(MapCodec<K, V>.Reader in) throws IOException {
    Map<K, V> map = delegate();
    MapCodec.Entry<K, V> entry;
    while ((entry = in.readEntry()) != null) {
      K key = entry.key();
      V value = entry.value();
      if (value != null) {
        map.put(key, value);
      } else {
        map.remove(key);
      }
    }
  }
}
