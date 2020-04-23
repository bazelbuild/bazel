// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.sandbox;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonToken;
import com.google.gson.stream.JsonWriter;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutionException;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/**
 * A sandboxfs implementation that uses an external sandboxfs binary to manage the mount point.
 *
 * <p>This implementation provides support for the reconfiguration protocol introduced in 0.2.0.
 */
final class RealSandboxfs02Process extends RealSandboxfsProcess {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /**
   * Writer with which to send data to the sandboxfs instance. Null only after {@link #destroy()}
   * has been invoked.
   */
  @GuardedBy("this")
  private JsonWriter processStdIn;

  /**
   * Collection of active reconfiguration requests.
   *
   * <p>Each entry in this map is keyed by the identifier of the sandbox being affected by a
   * reconfiguration request and points to a future that is set when the request completes.
   *
   * <p>New entries can be added to this map at any time, but only before {@link #destroy} is
   * called. Once the sandboxfs instance has been destroyed, we do not expect any new requests to
   * come in. However, existing requests will be drained by the {@link ResponsesReader} thread.
   */
  private final ConcurrentMap<String, SettableFuture<Void>> inFlightRequests =
      new ConcurrentHashMap<>();

  /**
   * Thread that reads responses from sandboxfs and dispatches them to the futures maintained by
   * {@link #inFlightRequests}.
   */
  private final Thread responsesReader;

  /** Representation of a response returned by sandboxfs. */
  private static class Response {
    /** Identifier given in the request. Null if this carries a fatal error. */
    @Nullable final String id;

    /**
     * Error message returned by sandboxfs if not null. If {@link #id} is not null, then this error
     * corresponds to a specific request and is recoverable. Otherwise corresponds to a fatal
     * condition, in which case sandboxfs will have stopped listening for requests.
     */
    @Nullable final String error;

    /** Constructs a new response with the given values. */
    Response(@Nullable String id, @Nullable String error) {
      this.id = id;
      this.error = error;
    }
  }

  /**
   * A thread that reads responses from the sandboxfs output stream and dispatches them to the
   * futures awaiting for them.
   */
  private static class ResponsesReader extends Thread {

    private final JsonReader reader;
    private final ConcurrentMap<String, SettableFuture<Void>> inFlightRequests;

    ResponsesReader(
        JsonReader reader, ConcurrentMap<String, SettableFuture<Void>> inFlightRequests) {
      this.reader = reader;
      this.inFlightRequests = inFlightRequests;
    }

    /** Waits for responses and dispatches them. */
    private void processResponses() throws IOException {
      while (!Thread.interrupted() && reader.peek() != JsonToken.END_DOCUMENT) {
        Response response = readResponse(reader);
        if (response.id == null) {
          // Non-recoverable error: abort.
          throw new IOException(response.error != null ? response.error : "No error reported");
        }

        SettableFuture<Void> future = inFlightRequests.remove(response.id);
        if (future == null) {
          throw new IOException("sandboxfs returned response for unknown id " + response.id);
        }
        if (response.error == null) {
          future.set(null);
        } else {
          future.setException(new IOException(response.error));
        }
      }
    }

    @Override
    public void run() {
      try {
        processResponses();
      } catch (EOFException e) {
        // OK, nothing to do.
      } catch (IOException e) {
        logger.atWarning().withCause(e).log("Failed to read responses from sandboxfs");
      }

      // sandboxfs has either replied with an unrecoverable error or has stopped providing
      // responses. Either way, we have to clean up any pending in-flight requests to unblock the
      // threads waiting for them.
      //
      // Given that we only get here once destroy() has been called, we do not expect any new
      // requests to show up in the inFlightRequests map. This is why we do not synchronize
      // accesses to the map during the iteration.
      while (!inFlightRequests.isEmpty()) {
        Iterator<Map.Entry<String, SettableFuture<Void>>> iter =
            inFlightRequests.entrySet().iterator();
        while (iter.hasNext()) {
          Map.Entry<String, SettableFuture<Void>> entry = iter.next();
          entry.getValue().cancel(true);
          iter.remove();
        }
      }
    }
  }

  /** A pair that represents the identifier and the path of a prefix as it will be serialized. */
  private static class Prefix {
    Integer id;
    String serializedPath;

    /**
     * Constructs a new prefix from its components.
     *
     * @param id the numerical identifier of the prefix
     * @param serializedPath the path of the prefix. Given that this is typically comes from a call
     *     to {@link Path#getParentDirectory()}, the value may be null or empty.
     */
    Prefix(Integer id, @Nullable PathFragment serializedPath) {
      this.id = id;
      if (serializedPath == null || serializedPath.isEmpty()) {
        this.serializedPath = "/";
      } else {
        this.serializedPath = serializedPath.getPathString();
      }
    }
  }

  /** Registers and assigns unique identifiers to path prefixes. */
  private static class Prefixes {
    /** Collection of all known path prefixes to this sandboxfs instance. */
    @GuardedBy("this")
    private final Map<PathFragment, Integer> prefixes = new HashMap<>();

    /** Last identifier assigned to a path prefix. */
    @GuardedBy("this")
    private Integer lastPrefixId = 1;

    /**
     * Registers the given path as a prefix.
     *
     * @param path the path to register as a prefix, which should always be the return value of a
     *     call to {@link Path#getParentDirectory()}. As a result, this should either be an absolute
     *     path, or a null or empty path.
     * @param newPrefixes tracker for any new prefixes registered by this function. If the given
     *     path is not yet known, it will be added to this collection.
     * @return the identifier for the prefix
     */
    private synchronized Integer registerPrefix(PathFragment path, ArrayList<Prefix> newPrefixes) {
      Integer id = prefixes.get(path);
      if (id != null) {
        return id;
      }

      id = lastPrefixId;
      prefixes.put(path, id);
      newPrefixes.add(new Prefix(id, path));
      lastPrefixId++;

      return id;
    }
  }

  /** Tracker for all known path prefixes to this sandboxfs instance. */
  private final Prefixes allPrefixes = new Prefixes();

  /**
   * Initializes a new sandboxfs process instance.
   *
   * @param process process handle for the already-running sandboxfs instance
   */
  RealSandboxfs02Process(Path mountPoint, Subprocess process) {
    super(mountPoint, process);

    this.processStdIn =
        new JsonWriter(
            new BufferedWriter(new OutputStreamWriter(process.getOutputStream(), UTF_8)));
    JsonReader processStdOut =
        new JsonReader(new BufferedReader(new InputStreamReader(process.getInputStream(), UTF_8)));

    // Must use lenient writing and parsing to accept a stream of separate top-level JSON objects.
    this.processStdIn.setLenient(true);
    processStdOut.setLenient(true);

    responsesReader = new ResponsesReader(processStdOut, inFlightRequests);
    responsesReader.start();
  }

  @Override
  public synchronized void destroy() {
    super.destroy();

    responsesReader.interrupt();
    try {
      responsesReader.join();
    } catch (InterruptedException e) {
      logger.atWarning().withCause(e).log(
          "Interrupted while waiting for responses processor thread");
      Thread.currentThread().interrupt();
    }

    processStdIn = null;
  }

  /**
   * Waits for a single response from sandboxfs and returns it.
   *
   * @param input the stream connected to sandboxfs's stdout
   * @return the response obtained from the stream
   * @throws IOException if sandboxfs fails to read from the stream for any reason, including EOF
   */
  private static Response readResponse(JsonReader input) throws IOException {
    input.beginObject();
    String id = null;
    String error = null;
    while (input.hasNext()) {
      String name = input.nextName();
      switch (name) {
        case "error":
          if (input.peek() == JsonToken.NULL) {
            input.nextNull();
          } else {
            checkState(error == null);
            error = input.nextString();
          }
          break;

        case "id":
          if (input.peek() == JsonToken.NULL) {
            input.nextNull();
          } else {
            checkState(id == null);
            id = input.nextString();
          }
          break;

        default:
          throw new IOException("Invalid field name in response: " + name);
      }
    }
    input.endObject();
    return new Response(id, error);
  }

  /**
   * Registers a new in-flight operation for the given sandbox identifier.
   *
   * <p>The caller must wait for the returned operation using {@link #waitForRequest}.
   *
   * @param id the identifier of the sandbox for which the request will be issued. There can only be
   *     one in-flight request per identifier.
   * @return the future used to wait for the request's completion
   */
  private SettableFuture<Void> newRequest(String id) {
    SettableFuture<Void> future = SettableFuture.create();
    SettableFuture<Void> other = inFlightRequests.put(id, future);
    checkState(other == null, "Cannot have two in-flight requests for sandbox '%s'", id);
    return future;
  }

  /**
   * Waits for a request to complete and unregisters its in-flight operation.
   *
   * @param future the value returned by {@link #newRequest}.
   * @throws IOException if the request cannot be waited for or if it raised an error
   */
  private static void waitForRequest(SettableFuture<Void> future) throws IOException {
    try {
      future.get();
    } catch (ExecutionException e) {
      Throwable cause = e.getCause();
      if (cause instanceof IOException) {
        throw (IOException) cause;
      } else {
        throw new AssertionError("Unexpected exception type thrown by readResponse()", cause);
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IOException("Interrupted while waiting for sandboxfs response");
    }
  }

  @Override
  @SuppressWarnings("UnnecessaryParentheses")
  public void createSandbox(String id, SandboxCreator creator) throws IOException {
    checkArgument(!PathFragment.containsSeparator(id));

    SettableFuture<Void> future = newRequest(id);
    synchronized (this) {
      processStdIn.beginObject();
      {
        processStdIn.name("C");
        processStdIn.beginObject();
        {
          processStdIn.name("i");
          processStdIn.value(id);
          processStdIn.name("m");
          processStdIn.beginArray();
          ArrayList<Prefix> newPrefixes = new ArrayList<>();
          creator.create(
              (path, underlyingPath, writable) -> {
                Integer pathPrefix =
                    allPrefixes.registerPrefix(path.getParentDirectory(), newPrefixes);
                Integer underlyingPathPrefix =
                    allPrefixes.registerPrefix(underlyingPath.getParentDirectory(), newPrefixes);

                // This synchronized block is theoretically unnecessary because the lock is already
                // held by the caller of this lambda. However, we need to reenter it to appease
                // the @GuardedBy clauses.
                synchronized (this) {
                  processStdIn.beginObject();
                  {
                    processStdIn.name("x");
                    processStdIn.value(pathPrefix);
                    processStdIn.name("p");
                    processStdIn.value(path.getBaseName());
                    processStdIn.name("y");
                    processStdIn.value(underlyingPathPrefix);
                    processStdIn.name("u");
                    processStdIn.value(underlyingPath.getBaseName());
                    if (writable) {
                      processStdIn.name("w");
                      processStdIn.value(writable);
                    }
                  }
                  processStdIn.endObject();
                }
              });
          processStdIn.endArray();
          if (!newPrefixes.isEmpty()) {
            processStdIn.name("q");
            processStdIn.beginObject();
            for (Prefix prefix : newPrefixes) {
              processStdIn.name(prefix.id.toString());
              processStdIn.value(prefix.serializedPath);
            }
            processStdIn.endObject();
          }
        }
        processStdIn.endObject();
      }
      processStdIn.endObject();

      processStdIn.flush();
    }
    waitForRequest(future);
  }

  @Override
  @SuppressWarnings("UnnecessaryParentheses")
  public void destroySandbox(String id) throws IOException {
    checkArgument(!PathFragment.containsSeparator(id));

    SettableFuture<Void> future = newRequest(id);
    synchronized (this) {
      processStdIn.beginObject();
      {
        processStdIn.name("D");
        processStdIn.value(id);
      }
      processStdIn.endObject();

      processStdIn.flush();
    }
    waitForRequest(future);
  }
}
