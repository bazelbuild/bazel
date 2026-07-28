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
import com.google.devtools.build.lib.unix.NativePosixFilesService;
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
 * <p>Peeks the pool by draining the fifo non-blockingly and refills it by writing {@code '+'} bytes;
 * the shared {@code issued}/{@code available}/{@code held} accounting lives in {@link
 * LocalJobserver.Backend}.
 */
public final class PosixJobserverBackend extends LocalJobserver.Backend {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final String dirPath;
  private final NativePosixFilesService posix;

  @Nullable private File fifoFile;
  @Nullable private RandomAccessFile fifo;
  @Nullable private FileOutputStream out;
  @Nullable private String writableDir;

  public PosixJobserverBackend(String dirPath, NativePosixFilesService posix) {
    this.dirPath = dirPath;
    this.posix = posix;
  }

  @Override
  public String start() throws IOException {
    if (dirPath.chars().anyMatch(Character::isWhitespace)) {
      // MAKEFLAGS has no interoperable escaping for whitespace in fifo auth paths.
      throw new IOException("local jobserver requires a whitespace-free output base path");
    }
    File dir = new File(dirPath);
    if (!dir.isDirectory() && !dir.mkdirs()) {
      throw new IOException("cannot create jobserver directory " + dirPath);
    }
    File file = new File(dir, "fifo");
    file.delete();
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

  // available() is broken on macOS fifos (returns 0), so the pool can't be peeked without emptying
  // it: drain the whole fifo and count the bytes. This empties it for ~microseconds each tick, but
  // clients block on acquire and the superclass refills the tokens to keep, so it self-heals.
  @Override
  protected int drainPool() throws IOException {
    try {
      return posix.drainFifoNonBlocking(fifo.getFD());
    } catch (NativePosixFilesException e) {
      throw new IOException("jobserver fifo read failed", e);
    }
  }

  @Override
  protected void refillPool(int count) throws IOException {
    if (count <= 0) {
      return;
    }
    byte[] tokens = new byte[count];
    Arrays.fill(tokens, (byte) '+');
    out.write(tokens);
    out.flush();
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
    fifo = null;
    out = null;
    fifoFile = null;
  }
}
