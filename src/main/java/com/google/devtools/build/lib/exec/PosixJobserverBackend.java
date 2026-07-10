// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.exec;

import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.unix.NativePosixFilesException;
import com.google.devtools.build.lib.unix.NativePosixFilesServiceImpl;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Arrays;
import javax.annotation.Nullable;

/**
 * The Linux/macOS {@link LocalJobserver.Backend}: a fifo. A client reads a byte to take a token and
 * writes one back to return it.
 *
 * <p>Accounting is exact every tick: of the {@code written - drained} tokens this manager has
 * handed to the fifo, the ones still sitting in it (counted via a non-blocking drain) are
 * unclaimed, so {@code written - drained - inFifo} are held by running tools.
 *
 */
public final class PosixJobserverBackend implements LocalJobserver.Backend {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final String dirPath;

  @Nullable private File fifoFile;
  @Nullable private NativePosixFilesServiceImpl posix;
  @Nullable private RandomAccessFile fifo;
  @Nullable private FileOutputStream out;
  @Nullable private String writableDir;

  private long written = 0;
  private long drained = 0;

  public PosixJobserverBackend(String dirPath) {
    this.dirPath = dirPath;
  }

  @Override
  public String start() throws IOException {
    File dir = new File(dirPath);
    if (!dir.isDirectory() && !dir.mkdirs()) {
      throw new IOException("cannot create jobserver directory " + dirPath);
    }
    File file = new File(dir, "fifo");
    file.delete();
    this.posix = new NativePosixFilesServiceImpl();
    try {
      posix.mkfifo(file.getPath(), 0600);
    } catch (NativePosixFilesException e) {
      throw new IOException("cannot create jobserver fifo " + file.getPath(), e);
    }
    this.fifoFile = file;
    // O_RDWR opens a fifo without blocking, unlike a lone read or write end.
    this.fifo = new RandomAccessFile(file, "rw");
    this.out = new FileOutputStream(fifo.getFD());
    this.writableDir = dir.getPath();
    return "fifo:" + file.getPath();
  }

  @Override
  @Nullable
  public String writableDir() {
    return writableDir;
  }

  @Override
  public int tick(int targetTokens) throws IOException {
    // available() is broken on macOS fifos (returns 0), so we can't peek the unclaimed count: drain
    // the whole pool, count it, and rewrite the tokens to keep. This empties the fifo for
    // ~microseconds each 100ms tick, but clients block on acquire and the count is re-derived each
    // tick, so it self-heals.
    int inFifo;
    try {
      inFifo = posix.drainFifoNonBlocking(fifo.getFD());
    } catch (NativePosixFilesException e) {
      throw new IOException("jobserver fifo read failed", e);
    }
    int held = Math.max(0, (int) (written - drained) - inFifo);
    int desired = Math.max(0, targetTokens - held);
    if (inFifo > desired) {
      drained += inFifo - desired;
    } else if (desired > inFifo) {
      written += desired - inFifo;
    }
    if (desired > 0) {
      byte[] tokens = new byte[desired];
      Arrays.fill(tokens, (byte) '+');
      out.write(tokens);
      out.flush();
    }
    return held;
  }

  @Override
  public void wakeForShutdown() {
    if (out == null) {
      return;
    }
    try {
      // Unblock the manager if it is parked in a token-reclaiming read.
      out.write('+');
      out.flush();
    } catch (IOException e) {
      logger.atWarning().withCause(e).log("Failed to nudge jobserver manager");
    }
  }

  @Override
  public void close() {
    if (fifoFile == null) {
      return;
    }
    if (fifo != null) {
      try {
        fifo.close();
      } catch (IOException e) {
        logger.atWarning().withCause(e).log("Failed to close jobserver fifo");
      }
    }
    fifoFile.delete();
    posix = null;
    fifo = null;
    out = null;
    fifoFile = null;
  }
}
