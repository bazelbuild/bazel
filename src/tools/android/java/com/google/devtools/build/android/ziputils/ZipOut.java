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
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDOFF;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDSIZ;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDSUB;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDTOT;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * API for writing to a zip archive. This does not currently perform compression,
 * but merely provides the facilities for creating a zip archive. The client must
 * ensure that file content is written conforming to the created headers.
 */
public class ZipOut {
  /**
   * Central directory output buffer block size
   */
  private static final int DIR_BLOCK_SIZE =  1024 * 1024;

  private final String filename;
  // To ensure good performance when writing to a remote file system
  // we need to use asynchronous output. However, because we may want this
  // to run on and off devices, we can't depend on java.nio.AsynchronousFileChannel (JDK 1.7).
  // Instead, we use a FileChannel (JDK 1.4), and use a single-thread pool,
  // to execute writes serially, but non-blocking from our client's
  // point-of-view. All data written, are assumed to remain unchanged until the write is complete.
  // ZipIn is designed to not reuse internal buffers, to make direct data transfer safe.
  // This optimizes the common cases where input is processed serially.
  private final FileChannel fileChannel;
  private final ExecutorService executor;
  private final List<Future<?>> futures;
  private final List<CentralDirectory> centralDirectory;
  private CentralDirectory current;
  private int fileOffset = 0;
  private int entryCount = 0;
  private boolean finished = false;
  private final boolean verbose = false;

  /**
   * Creates a {@code ZipOut} for writing to file, with the given (nick)name.
   *
   * @param channel File channel open for output.
   * @param filename File name or nickname.
   * @throws java.io.IOException
   */
  public ZipOut(FileChannel channel, String filename) throws IOException {
    this.executor = Executors.newSingleThreadExecutor();
    this.futures = new ArrayList<>();
    this.fileChannel = channel;
    this.filename = filename;
    centralDirectory = new ArrayList<>();
    fileOffset =  (int) fileChannel.position();
  }

  /**
   * Returns a writable copy of the given
   * {@link com.google.devtools.build.android.ziputils.DirectoryEntry}, backed by an internal
   * direct byte buffer, allocated over storage for this files central directory. The file offset
   * is set, and must not be changed. The variable entry data (filename, extra data and comment),
   * must not be changed (in a way that changes the total size of the directory entry).
   * @param entry directory entry to copy.
   * @return a writable directory entry view, over the provided byte buffer.
   */
  public DirectoryEntry nextEntry(DirectoryEntry entry) {
    entryCount++;
    int size = entry.getSize();
    if (current == null || current.buffer.remaining() < size) {
      ByteBuffer buffer = ByteBuffer.allocateDirect(DIR_BLOCK_SIZE);
      current = CentralDirectory.viewOf(buffer);
      centralDirectory.add(current);
    }
    return current.nextEntry(entry).set(CENOFF, fileOffset);
  }

  /**
   * Writes content to the current entry. Content is written as-is, and the client is responsible
   * for compression, consistent with the storage method set in the current directory entry. The
   * client must first write an appropriate local file header, and if necessary, complete the entry
   * with a data descriptor. If the header indicates that the content is compressed, the client is
   * responsible for compressing the data before writing.
   *
   * <p>Data is written serially, but asynchronously, to the output file. The client must not change
   * the underlying data after it has been scheduled for writing. Usually, a client will release
   * any references to the data, so that storage may be eligible for GC, once the write operation
   * has completed. It's safe to pass references to views of data obtained from a {#link ZipIn},
   * object, because {@code ZipIn} doesn't reuse internal buffers.
   *
   * @param content
   */
  public synchronized void write(ByteBuffer content) {
    fileOffset += content.remaining();
    futures.add(executor.submit(new OutputTask(content)));
  }

  /**
   * Writes a {@link com.google.devtools.build.android.ziputils.View} to the current entry.
   * Used to write a {@link com.google.devtools.build.android.ziputils.LocalFileHeader}
   * before the content, and if needed, a
   * {@link com.google.devtools.build.android.ziputils.DataDescriptor} after the content.
   * <P>
   * See also {@link #write(java.nio.ByteBuffer)}.
   * </P>
   *
   * @param view the view to write as part of the current entry.
   * @throws java.io.IOException
   */
  public void write(View<?> view) throws IOException {
    view.at(fileOffset).buffer.rewind();
    write(view.buffer);
  }

  /**
   * Returns the file position for the next write operation. Because writes are asynchronous, this
   * may not be the actual position of the underlying file channel.
   * @return the file position position for the next write operation.
   */
  public int fileOffset() {
    return fileOffset;
  }

  /**
   * Writes out the central directory. This doesn't close the output file.
   */
  public void finish() throws IOException {
    if (finished) {
      return;
    }
    finished = true;
    int cdOffset = fileOffset;
    for (CentralDirectory cd : centralDirectory) {
      //cd.finish().buffer.rewind();
      cd.buffer.flip();
      write(cd.buffer);
    }
    int size = fileOffset - cdOffset;
    verbose("ZipOut finishing: " + filename);
    verbose("-- CDIR: " + cdOffset + " count: " + entryCount);
    verbose("-- EOCD: " + fileOffset + " size: " + size);
    EndOfCentralDirectory eocd = EndOfCentralDirectory.allocate(null)
        .set(ENDSUB, (short) entryCount)
        .set(ENDTOT, (short) entryCount)
        .set(ENDSIZ, size)
        .set(ENDOFF, cdOffset)
        .at(fileOffset);
    eocd.buffer.rewind();
    write(eocd.buffer);
    verbose("-- size: " + fileOffset);
  }

  /**
   * Closes the output file. If this object has not been finished yet, this method will call
   * {@link #finish()} before closing the output channel.
   *
   * @throws java.io.IOException
   */
  public void close() throws IOException {
    if (!finished) {
      finish();
    }
    try {
      executor.shutdown();
      executor.awaitTermination(30, TimeUnit.SECONDS);
      for (Future<?> f : futures) {
        try {
          f.get();
        } catch (ExecutionException ex) {
          throw new IOException(ex.getCause().getMessage());
        }
      }
    } catch (InterruptedException ex) {
      executor.shutdownNow();
      Thread.currentThread().interrupt();
    }
    fileChannel.close();
  }

  /**
   * Helper class to write asynchronously to the output channel.
   */
  private class OutputTask implements Runnable {

    final ByteBuffer buffer;

    public OutputTask(ByteBuffer buffer) {
      this.buffer = buffer;
    }

    @Override
    public void run() {
      try {
        fileChannel.write(buffer);
      } catch (IOException ex) {
        throw new IllegalStateException("Unexpected IOException writing to output channel");
      }
    }
  }

  private void verbose(String msg) {
    if (verbose) {
      System.out.println(msg);
    }
  }
}
