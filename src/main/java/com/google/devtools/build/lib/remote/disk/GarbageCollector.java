// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.disk;

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.concurrent.ErrorClassifier;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.remote.disk.Sqlite.Connection;
import com.google.devtools.build.lib.remote.disk.Sqlite.Result;
import com.google.devtools.build.lib.remote.disk.Sqlite.Statement;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.IORuntimeException;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Semaphore;
import javax.annotation.Nullable;

/**
 * The garbage collector for {@link DiskCacheClient}.
 *
 * <p>The garbage collector aims to keep the total size of the disk cache under a configured limit
 * by deleting least recently accessed entries to make space for newly inserted ones. Note that, for
 * robustness, the time of last access to an entry is indicated by its file modification time.
 *
 * <p>In order to avoid costly filesystem scans, the garbage collector is aided by a SQLite database
 * containing the names, sizes and modification times of every cache entry. This database is
 * initially computed from a filesystem scan and subsequently kept in sync with modifications caused
 * by disk cache accesses.
 */
public final class GarbageCollector {
  private static final String DB_NAME = "db.sqlite";

  private final Path dbPath;
  private final ImmutableSet<Path> excludedDirs;

  private final GcThread gcThread;

  GarbageCollector(
      Path root, String gcDir, ImmutableSet<String> excludedDirs, ExecutorService executorService)
      throws IOException {
    this.dbPath = root.getChild(gcDir).getChild(DB_NAME);
    this.excludedDirs = excludedDirs.stream().map(root::getChild).collect(toImmutableSet());
    this.gcThread = new GcThread(root, executorService);
    gcThread.start();
  }

  /**
   * The thread where all garbage collector work occurs.
   *
   * <p>Using a separate thread makes it possible to collect garbage in the background and ensures
   * that access to the database is serialized.
   */
  private final class GcThread extends Thread {
    private final Path root;
    private final ExecutorService executorService;
    private final Connection conn;
    @Nullable private Exception exception;

    GcThread(Path root, ExecutorService executorService) throws IOException {
      super("disk-cache-gc");
      this.root = root;
      this.executorService = executorService;
      this.conn = Sqlite.newConnection(dbPath);
    }

    /**
     * Runs the garbage collector thread.
     *
     * <p>The thread terminates when there is no work left to do or an unrecoverable exception
     * occurs. The latter may be retrieved via {@link #getException} after {@link #join} returns.
     */
    @Override
    public void run() {
      try {
        setupDatabase();
      } catch (IOException e) {
        exception = e;
      } finally {
        try {
          conn.close();
        } catch (IOException e) {
          if (exception != null) {
            exception.addSuppressed(e);
          } else {
            exception = e;
          }
        }
      }
    }

    /**
     * Returns the exception that caused the garbage collector thread to terminate prematurely, or
     * null if no such exception occurred.
     *
     * <p>Must not be called before {@link #join} returns.
     */
    @Nullable
    Exception getException() {
      return exception;
    }

    private void setupDatabase() throws IOException {
      try (SilentCloseable c = Profiler.instance().profile("DiskCacheGC/setupDatabase")) {
        // Set the journal mode to WAL, which performs much better for frequent writes.
        conn.executeUpdate("PRAGMA journal_mode=WAL");

        // Set the synchronous mode to NORMAL, which performs better than FULL and, according to
        // the documentation, is sufficient to prevent corruption in WAL mode.
        conn.executeUpdate("PRAGMA synchronous=NORMAL");

        // Create the entry table if it doesn't yet exist.
        // Enforce column types (STRICT) and don't waste space with ids (WITHOUT ROWID).
        conn.executeUpdate(
            """
            CREATE TABLE IF NOT EXISTS entries
            (path TEXT PRIMARY KEY NOT NULL, size INTEGER NOT NULL, mtime INTEGER NOT NULL)
            STRICT, WITHOUT ROWID
            """);

        // Create the index on the entry table if it doesn't yet exist.
        conn.executeUpdate("CREATE INDEX IF NOT EXISTS mtime_index ON entries (mtime)");

        // Recreate the database from the filesystem if needed.
        maybeRecreateDatabase();
      }
    }

    /** Checks whether the database is empty, and if so, recreates it from the filesystem. */
    private void maybeRecreateDatabase() throws IOException {
      try (SilentCloseable c = Profiler.instance().profile("DiskCacheGC/maybeRecreateDatabase");
          Statement countStmt = conn.newStatement("SELECT COUNT(*) FROM entries");
          Statement clearStmt = conn.newStatement("DELETE FROM entries");
          Statement insertStmt = conn.newStatement("INSERT INTO entries VALUES (?, ?, ?)")) {

        // Mutex to guard access to the database connection.
        Semaphore mutex = new Semaphore(1);

        // Whether the filesystem traversal was interrupted.
        boolean interrupted = false;

        // Wrap in a transaction to make it possible to rollback on interruption.
        conn.executeUpdate("BEGIN");
        try {
          try (Result r = countStmt.executeQuery()) {
            checkState(r.next());
            if (r.getLong(0) > 0) {
              // No need to recreate.
              // TODO(tjgq): Detect staleness.
              return;
            }
          }

          clearStmt.executeUpdate();

          new FileSystemTraversal(
                  root,
                  excludedDirs,
                  executorService,
                  (path, size, mtime) -> {
                    mutex.acquireUninterruptibly();
                    try {
                      insertStmt.bindString(1, path);
                      insertStmt.bindLong(2, size);
                      insertStmt.bindLong(3, mtime);
                      insertStmt.executeUpdate();
                    } finally {
                      mutex.release();
                    }
                  })
              .run();
        } catch (InterruptedException e) {
          interrupted = true;
        } finally {
          if (interrupted) {
            conn.executeUpdate("ROLLBACK");
          } else {
            conn.executeUpdate("COMMIT");
          }
        }
      }
    }
  }

  /** Blocks until the garbage collector is finished. */
  public void close() {
    // Wait for the garbage collector thread to terminate.
    Uninterruptibles.joinUninterruptibly(gcThread);
    Exception e = gcThread.getException();
    if (e != null) {
      // TODO(tjgq): Surface error in a better way.
      throw new AssertionError(e);
    }
  }

  /** Traverses a filesystem hierarchy. */
  private static final class FileSystemTraversal extends AbstractQueueVisitor {
    private final Path root;
    private final ImmutableSet<Path> excludedDirs;
    private final Visitor visitor;

    /** The visitor invoked for each file in the hierarchy. */
    private interface Visitor {
      /**
       * @param path the file path relative to {@code root}
       * @param size the file size
       * @param mtime the file modification time
       */
      void visitFile(String path, long size, long mtime) throws IOException;
    }

    FileSystemTraversal(
        Path root,
        ImmutableSet<Path> excludedDirs,
        ExecutorService executorService,
        Visitor visitor) {
      super(
          executorService,
          ExecutorOwnership.SHARED,
          ExceptionHandlingMode.FAIL_FAST,
          ErrorClassifier.DEFAULT);
      this.root = root;
      this.excludedDirs = excludedDirs;
      this.visitor = visitor;
    }

    void run() throws IOException, InterruptedException {
      execute(() -> visitDirectory(root));
      try {
        awaitQuiescence(true);
      } catch (IORuntimeException e) {
        throw e.getCauseIOException();
      }
    }

    private void visitDirectory(Path path) {
      try {
        for (Dirent dirent : path.readdir(Symlinks.NOFOLLOW)) {
          Path childPath = path.getChild(dirent.getName());
          if (dirent.getType().equals(Dirent.Type.FILE)) {
            // The file may be gone by the time we open it.
            FileStatus status = childPath.statNullable();
            if (status != null) {
              visitor.visitFile(
                  childPath.relativeTo(root).getPathString(),
                  status.getSize(),
                  status.getLastModifiedTime());
            }
          } else if (dirent.getType().equals(Dirent.Type.DIRECTORY)
              && !excludedDirs.contains(childPath)) {
            execute(() -> visitDirectory(childPath));
          }
          // Deliberately ignore other file types, which should never occur.
        }
      } catch (IOException e) {
        throw new IORuntimeException(e);
      }
    }
  }
}
