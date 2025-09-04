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

package com.google.devtools.build.lib.skyframe.serialization.analysis;

import com.google.common.base.Verify;
import com.google.common.flogger.GoogleLogger;
import com.google.gson.Strictness;
import com.google.gson.stream.JsonWriter;
import java.io.Closeable;
import java.io.IOException;
import java.time.Instant;
import java.util.concurrent.locks.ReentrantLock;

/** Writes a detailed JSON log of what's happening with remote analysis caching. */
public class RemoteAnalysisJsonLogWriter implements Closeable {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final ReentrantLock lock;
  private JsonWriter jsonWriter;
  private boolean hadErrors;

  public RemoteAnalysisJsonLogWriter(JsonWriter jsonWriter) {
    jsonWriter.setStrictness(Strictness.LENIENT); // For multiple top-level values
    this.jsonWriter = jsonWriter;
    this.lock = new ReentrantLock();
    this.hadErrors = false;
  }

  /**
   * Returns whether any errors happened during log writing.
   *
   * <p>Should only be called after {@link #close()} was.
   */
  public boolean hadErrors() {
    Verify.verify(jsonWriter == null);
    return hadErrors;
  }

  /**
   * Starts a new log entry.
   *
   * <p>Returns a {@link Closeable} that can be used to add data to the log entry. Care should be
   * taken to close this object as soon as possible, since while it's active, a lock is held.
   */
  public Entry startEntry(String op, Instant start, Instant end) {
    Entry entry = new Entry(); // Acquire the lock

    try {
      Verify.verify(jsonWriter != null);
      jsonWriter.beginObject();
      jsonWriter.name("start");
      writeInstant(start);
      jsonWriter.name("end");
      writeInstant(end);
      jsonWriter.name("op").value(op);
    } catch (IOException e) {
      hadErrors = true;
      logger.atWarning().withCause(e).log("Cannot write JSON log entry");
    } catch (RuntimeException e) {
      // Even if there is a bug, make sure to release the lock so that there is no deadlock
      entry.close();
      throw e;
    }

    return entry;
  }

  /**
   * A log entry being written.
   *
   * <p>Comes with some trivial methods to actually add data to it.
   */
  public final class Entry implements Closeable {
    private Entry() {
      lock.lock();
    }

    @Override
    public void close() {
      Verify.verify(lock.isHeldByCurrentThread());
      try {
        jsonWriter.endObject();
      } catch (IOException e) {
        logger.atWarning().withCause(e).log("Cannot write JSON log entry");
      }
      lock.unlock();
    }

    /** Adds a new field to this log entry. */
    public void addField(String name, String value) {
      Verify.verify(lock.isHeldByCurrentThread());
      try {
        jsonWriter.name(name).value(value);
      } catch (IOException e) {
        logger.atWarning().withCause(e).log("Cannot write JSON log entry");
      }
    }

    /** Adds a new field to this log entry. */
    public void addField(String name, long value) {
      Verify.verify(lock.isHeldByCurrentThread());
      try {
        jsonWriter.name(name).value(value);
      } catch (IOException e) {
        hadErrors = true;
        logger.atWarning().withCause(e).log("Cannot write JSON log entry");
      }
    }
  }

  @Override
  public void close() {
    lock.lock();
    try {
      jsonWriter.close();
    } catch (IOException e) {
      hadErrors = true;
      logger.atWarning().withCause(e).log("Cannot close JSON log");
    } finally {
      jsonWriter = null;
      lock.unlock();
    }
  }

  private void writeInstant(Instant instant) throws IOException {
    jsonWriter.beginObject();
    jsonWriter.name("seconds").value(instant.getEpochSecond());
    jsonWriter.name("nanos").value(instant.getNano());
    jsonWriter.endObject();
  }
}
