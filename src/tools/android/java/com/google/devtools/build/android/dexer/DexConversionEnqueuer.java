// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.dexer;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.util.concurrent.Futures.immediateFuture;

import com.google.common.cache.Cache;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.android.dexer.Dexing.DexingKey;
import java.io.IOException;
import java.io.InputStream;
import java.util.Enumeration;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import javax.annotation.Nullable;

/**
 * Worker that reads an input Jar file and creates futures to convert .class to .dex files while
 * leaving other files in the Jar unchanged.  Converted files appear in {@link #getFiles()}.
 * Because the queue of files is size-limited, this {@link Callable} must not be invoked on the
 * main thread to avoid deadlocking.
 *
 * <p>Note on the name: this callable enqueues futures to convert .class files into a thread pool;
 * it doesn't return a value itself other than successful termination or an exception during input
 * file reading.
 */
class DexConversionEnqueuer implements Callable<Void> {

  private static final byte[] EMPTY = new byte[0];

  private final ZipFile in;
  private final DexConverter dexer;
  private final ExecutorService executor;
  @Nullable private final Cache<DexingKey, byte[]> dexCache;

  /** Converted content of the input file.  See {@link #getFiles()} for more details. */
  // Rate-limit to 30000 files in flight at once, which is about what we've tested.  Theoretically,
  // an unbounded queue can lead to OutOfMemoryErrors, and any limit is helpful so that one can
  // set -Xmx to handle even large files.  The "downside" is that this callable can theoretically
  // block trying to add more elements to the queue, wasting the thread during that time.
  private final BlockingQueue<Future<ZipEntryContent>> files = new ArrayBlockingQueue<>(30000);

  public DexConversionEnqueuer(ZipFile in, ExecutorService executor, DexConverter dexer,
      @Nullable Cache<DexingKey, byte[]> dexCache) {
    this.in = in;
    this.executor = executor;
    this.dexer = dexer;
    this.dexCache = dexCache;
  }

  @Override
  public Void call() throws InterruptedException, IOException {
    try {
      Enumeration<? extends ZipEntry> entries = in.entries();
      while (entries.hasMoreElements()) {
        ZipEntry entry = entries.nextElement();
        // Since these entries come from an existing zip file, they should always know their size
        // (meaning, never return -1). We also can't handle files that don't fit into a byte array.
        checkArgument(entry.getSize() >= 0, "Cannot process entry with unknown size: %s", entry);
        checkArgument(entry.getSize() <= Integer.MAX_VALUE, "Entry too large: %s", entry);
        byte[] content;
        if (entry.getSize() == 0L) {
          content = EMPTY; // this in particular covers directory entries
        } else {
          try (InputStream entryStream = in.getInputStream(entry)) {
            // Read all the content at once, which avoids temporary arrays and extra array copies
            content = new byte[(int) entry.getSize()];
            ByteStreams.readFully(entryStream, content); // throws if file is smaller than expected
            checkState(entryStream.read() == -1,
                "Too many bytes in jar entry %s, expected %s", entry, entry.getSize());
          }
        }
        if (!entry.isDirectory() && entry.getName().endsWith(".class")) {
          files.put(toDex(entry, content));
        } else {
          // Copy other files and directory entries
          if (entry.getCompressedSize() != 0) {
            entry.setCompressedSize(-1L); // We may compress differently from source Zip
          }
          files.put(immediateFuture(new ZipEntryContent(entry, content)));
        }
      }
    } finally {
      // Use try-finally to make absolutely sure we do this, otherwise the reader might deadlock
      files.put(immediateFuture((ZipEntryContent) null)); // "end of stream" marker
    }
    return null;
  }

  private Future<ZipEntryContent> toDex(ZipEntry entry, byte[] content) {
    byte[] cached = dexCache != null ? dexCache.getIfPresent(dexer.getDexingKey(content)) : null;
    return cached != null
        ? immediateFuture(storedDexEntry(entry, cached))
        : executor.submit(new ClassToDex(entry, content, dexer, dexCache));
  }

  /**
   * Converted .dex files as well as (unchanged) resources in the order they appear in {@link #in
   * the input zip file}.  For simplicity we use a separate future for each file, followed by a
   * future returning {@code null} to indicate that the input zip file is exhausted.  To achieve
   * determinism, the consumer of this queue should write the content of each file in the order
   * they appear in this queue.  Note that no files will appear in this queue until this callable is
   * {@link #call invoked}, typically by submitting it to an {@link ExecutorService}.  Once a
   * future returning {@code null} appears in the queue, no more elements will follow, so the
   * consumer should be a single-threaded loop that terminates on this {@code null} sentinel.
   */
  public BlockingQueue<Future<ZipEntryContent>> getFiles() {
    return files;
  }

  private static ZipEntryContent storedDexEntry(ZipEntry classfile, byte[] dexed) {
    return new ZipEntryContent(
        storedEntry(classfile.getName() + ".dex", classfile.getTime(), dexed),
        dexed);
  }

  private static ZipEntry storedEntry(String filename, long time, byte[] content) {
    // Need to pre-compute checksum for STORED (uncompressed) entries)
    CRC32 checksum = new CRC32();
    checksum.update(content);

    ZipEntry result = new ZipEntry(filename);
    result.setTime(time);
    result.setCrc(checksum.getValue());
    result.setSize(content.length);
    result.setCompressedSize(content.length);
    // Write uncompressed, since this is just an intermediary artifact that
    // we will convert to .dex
    result.setMethod(ZipEntry.STORED);
    return result;
  }

  /**
   * Worker to convert a {@code byte[]} representing a .class file into a {@code byte[]}
   * representing a .dex file.
   */
  private static class ClassToDex implements Callable<ZipEntryContent> {

    private final ZipEntry entry;
    private final byte[] content;
    private final DexConverter dexer;
    @Nullable private final Cache<DexingKey, byte[]> dexCache;

    public ClassToDex(ZipEntry entry, byte[] content, DexConverter dexer,
        @Nullable Cache<DexingKey, byte[]> dexCache) {
      this.entry = entry;
      this.content = content;
      this.dexer = dexer;
      this.dexCache = dexCache;
    }

    @Override
    public ZipEntryContent call() throws Exception {
      byte[] dexed = DexFiles.encode(dexer.toDexFile(content, entry.getName()));
      if (dexCache != null) {
        dexCache.put(dexer.getDexingKey(content), dexed);
      }
      // Use .class.dex suffix expected by SplitZip
      return storedDexEntry(entry, dexed);
    }
  }
}
