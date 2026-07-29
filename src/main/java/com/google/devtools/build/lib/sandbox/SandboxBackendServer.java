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

package com.google.devtools.build.lib.sandbox;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.sandbox.proto.SandboxProto.Collect;
import com.google.devtools.build.lib.sandbox.proto.SandboxProto.Content;
import com.google.devtools.build.lib.sandbox.proto.SandboxProto.Create;
import com.google.devtools.build.lib.sandbox.proto.SandboxProto.CustomConfinement;
import com.google.devtools.build.lib.sandbox.proto.SandboxProto.Destroy;
import com.google.devtools.build.lib.sandbox.proto.SandboxProto.Error;
import com.google.devtools.build.lib.sandbox.proto.SandboxProto.Manifest;
import com.google.devtools.build.lib.sandbox.proto.SandboxProto.Negotiate;
import com.google.devtools.build.lib.sandbox.proto.SandboxProto.Push;
import com.google.devtools.build.lib.sandbox.proto.SandboxProto.Request;
import com.google.devtools.build.lib.sandbox.proto.SandboxProto.Response;
import com.google.devtools.build.lib.sandbox.proto.SandboxProto.Retention;
import com.google.devtools.build.lib.sandbox.proto.SandboxProto.Version;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.ByteString;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Level;
import javax.annotation.Nullable;

/**
 * Long-lived client to the sandbox backend running in {@code serve} mode.
 *
 * <p>Wire protocol over stdin/stdout: each message is a varint length-delimited {@link Request} (or
 * {@link Response}) proto ({@code src/main/protobuf/sandbox.proto}), framed via protobuf's {@code
 * writeDelimitedTo}/{@code parseDelimitedFrom}. Each request carries a caller-assigned {@code rid};
 * responses arrive asynchronously and are matched back by rid. A single reader thread drains stdout
 * and completes the {@link CompletableFuture}s of the issuing Bazel threads, so actions multiplex
 * over one controller.
 *
 * <p>One controller per Bazel server (JVM): lazily spawned on first use, cached in a static
 * singleton, shut down via a JVM shutdown hook by closing stdin and NEVER killing it — it must exit
 * on its own so it can hand warm state off to the next controller. If it dies, the next call
 * re-spawns it.
 */
final class SandboxBackendServer {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  // Sandbox protocol versions Bazel speaks, advertised in the Negotiate handshake. One today.
  private static final ImmutableList<Version> PROTOCOL_VERSIONS =
      ImmutableList.of(Version.VERSION_1);

  // Bound on create attempts. Attempt 1 relies on the proactive push; each MissingContent reply
  // drives one more retry after pushing exactly what was named. One retry all but always suffices
  // (the controller is authoritative once we push what it asked for); more than this means content
  // is being evicted between the push and the re-Create, which we fail rather than spin on.
  private static final int MAX_CREATE_ATTEMPTS = 3;

  /**
   * Response from a {@code create} request.
   *
   * @param path absolute sandbox root path the action's cwd will be derived from
   * @param custom the backend's own jail to apply, or {@code null} if the backend brought none
   * @param unconfined {@code true} if the backend confines itself and Bazel should apply no jail;
   *     when both {@code custom} is null and this is false, Bazel applies its platform default
   */
  record CreateResponse(String path, @Nullable CustomConfinement custom, boolean unconfined) {}

  // One server per backend name, lazily spawned and reused across actions. A name's server is
  // respawned when its launch identity (binary + options) changes between commands or after it dies.
  private static final Object SERVERS_LOCK = new Object();
  private static final Map<String, SandboxBackendServer> servers = new HashMap<>();
  private static boolean shutdownHookInstalled;

  /**
   * Returns the server for backend {@code name}, spawning one if absent, dead, or launched with a
   * different binary/options. Relays {@code options} via the Negotiate handshake before returning.
   */
  static SandboxBackendServer getOrSpawn(
      String name,
      PathFragment binary,
      ImmutableList<String> options,
      ImmutableMap<String, String> clientEnv,
      String workspace)
      throws IOException {
    String identity = binary.getPathString() + '\0' + String.join("\0", options);
    synchronized (SERVERS_LOCK) {
      SandboxBackendServer existing = servers.get(name);
      if (existing != null && existing.alive && identity.equals(existing.identity)) {
        return existing;
      }
      if (existing != null) {
        try {
          existing.close();
        } catch (Exception ignored) {
          // best-effort cleanup of the previous server
        }
      }
      SandboxBackendServer server = spawn(binary, identity, clientEnv, workspace);
      server.negotiate(options);
      servers.put(name, server);
      if (!shutdownHookInstalled) {
        shutdownHookInstalled = true;
        Runtime.getRuntime()
            .addShutdownHook(
                new Thread(
                    () -> {
                      synchronized (SERVERS_LOCK) {
                        for (SandboxBackendServer s : servers.values()) {
                          try {
                            s.close();
                          } catch (Exception ignored) {
                          }
                        }
                      }
                    },
                    "SandboxBackendServer-shutdown"));
      }
      return server;
    }
  }

  private static SandboxBackendServer spawn(
      PathFragment binary,
      String identity,
      ImmutableMap<String, String> clientEnv,
      String workspace)
      throws IOException {
    Subprocess process =
        new SubprocessBuilder(clientEnv)
            .setArgv(
                ImmutableList.of(binary.getPathString(), "serve", "--workspace", workspace))
            .setStdout(SubprocessBuilder.StreamAction.STREAM)
            .setWorkingDirectory(new File("."))
            .start();
    return new SandboxBackendServer(process, binary, identity, requestTimeout(clientEnv));
  }

  /**
   * Per-request read deadline. The protocol round-trip (mount a sandbox, tear it down) is near-
   * instant and input-size-independent — the controller projects a virtual FS, it doesn't copy the
   * tree — so a request that hasn't answered within a second means the controller is wedged or its
   * stdout frame stream is corrupted, not that it's busy. A bound is mandatory: without it a
   * corrupted stream parks every execution thread on {@code future.get()} forever (the bug this
   * guards against). Overridable via {@code SANDBOX_BACKEND_REQUEST_TIMEOUT_SECS} for pathological hosts.
   */
  private static Duration requestTimeout(ImmutableMap<String, String> clientEnv) {
    String override = clientEnv.get("SANDBOX_BACKEND_REQUEST_TIMEOUT_SECS");
    if (override != null) {
      try {
        long secs = Long.parseLong(override.trim());
        if (secs > 0) {
          return Duration.ofSeconds(secs);
        }
      } catch (NumberFormatException e) {
        logger.atWarning().log(
            "ignoring non-numeric SANDBOX_BACKEND_REQUEST_TIMEOUT_SECS=%s", override);
      }
    }
    return DEFAULT_REQUEST_TIMEOUT;
  }

  // A cold full build of a large input tree (e.g. node_modules) is thousands of clonefiles, which
  // legitimately takes seconds. The timeout is only a wedged/corrupt-stream backstop, so keep it
  // generous; SANDBOX_BACKEND_REQUEST_TIMEOUT_SECS overrides.
  private static final Duration DEFAULT_REQUEST_TIMEOUT = Duration.ofSeconds(60);

  // Cap on the controller stderr we retain for diagnostics: keep the tail, drop older bytes. A
  // healthy controller is quiet; a dying one prints a panic/error that easily fits here.
  private static final int STDERR_TAIL_LIMIT = 8 * 1024;

  private final Subprocess process;
  private final PathFragment binary;
  // Launch identity (binary + options); a name's server is respawned when this changes.
  private final String identity;
  private final OutputStream out;
  private final InputStream in;
  private final Object writeLock = new Object();
  private final ConcurrentMap<Long, CompletableFuture<Response>> pending =
      new ConcurrentHashMap<>();
  private final AtomicLong nextRid = new AtomicLong(1);
  // Best-effort record of digest hashes already pushed into this controller's store as REUSABLE
  // content — directory blobs and RETENTION_REUSE content. Novel entries are pushed once and reused
  // across actions; single-use content is never recorded here (it is unique per action and would
  // grow the set unbounded). Only a hint: it may under-claim (a freshly respawned controller we
  // never pushed to — harmless, we just re-push) or over-claim (an entry we believe present was
  // evicted), and a Create referencing a truly-absent digest self-corrects via
  // Create.Result.MissingContent. Lives on the instance, so a respawned controller starts fresh.
  private final Set<String> pushed = ConcurrentHashMap.newKeySet();
  private volatile boolean alive = true;
  // Drains the controller's stderr so the pipe never backs up and so we can attribute its death.
  private final Thread stderrThread;
  // Tail of the controller's stderr, capped at STDERR_TAIL_LIMIT. Guards its own access.
  private final StringBuilder stderrTail = new StringBuilder();
  private final Duration requestTimeout;

  private SandboxBackendServer(
      Subprocess process, PathFragment binary, String identity, Duration requestTimeout) {
    this.process = process;
    this.binary = binary;
    this.identity = identity;
    this.requestTimeout = requestTimeout;
    this.out = process.getOutputStream();
    this.in = process.getInputStream();
    this.stderrThread = new Thread(this::drainStderr, "SandboxBackendServer-stderr");
    this.stderrThread.setDaemon(true);
    this.stderrThread.start();
    Thread t = new Thread(this::readLoop, "SandboxBackendServer-reader");
    t.setDaemon(true);
    t.start();
  }

  /**
   * Sends the one-time Negotiate handshake relaying the backend's configured options. Called once
   * per spawned server, before any sandbox is created.
   */
  private void negotiate(ImmutableList<String> options) throws IOException {
    long rid = nextRid.getAndIncrement();
    Response resp =
        request(
            rid,
            Request.newBuilder()
                .setRid(rid)
                .setNegotiate(
                    Negotiate.newBuilder()
                        .addAllOptions(options)
                        .addAllVersions(PROTOCOL_VERSIONS))
                .build(),
            "negotiate");
    Negotiate.Result result = expect(resp, Response.ResultCase.NEGOTIATE, "negotiate").getNegotiate();
    switch (result.getOutcomeCase()) {
      case OK -> {
        // The negotiated version is fixed (VERSION_1) today; nothing to record yet.
      }
      case ERROR -> throw failed("negotiate", result.getError());
      case OUTCOME_NOT_SET -> throw noOutcome("negotiate");
    }
  }

  /**
   * Creates a sandbox for {@code sandboxId} from {@code manifest}, whose tree references {@code
   * blobs} (digest hash -> serialized Directory) and {@code content} (digest hash -> leaf content,
   * inline bytes or a host path for the controller to capture).
   *
   * <p>The manifest references its structure and content only by digest; the bytes ship separately
   * via {@link #push}. We proactively push the reusable entries this controller has not seen
   * (tracked in {@link #pushed}) plus all single-use content ahead of the Create — the single
   * ordered pipe guarantees they arrive first — then send the Create. If the controller still
   * reports digests missing (a stale hint, an eviction, or a fresh controller), it names them in
   * {@link Create.Result.MissingContent}; we push exactly those and retry, bounded by {@link
   * #MAX_CREATE_ATTEMPTS} so eviction thrash can't spin forever.
   */
  CreateResponse createSandbox(
      String sandboxId,
      Manifest manifest,
      Map<String, ByteString> blobs,
      Map<String, Content> content)
      throws IOException {
    pushNovel(blobs, content);
    for (int attempt = 1; ; attempt++) {
      long rid = nextRid.getAndIncrement();
      Request req =
          Request.newBuilder()
              .setRid(rid)
              .setSandboxId(sandboxId)
              .setCreate(Create.newBuilder().setManifest(manifest))
              .build();
      Response resp;
      try (SilentCloseable c =
          Profiler.instance().profile("sandbox.wireRoundTrip(" + req.getSerializedSize() + "B)")) {
        resp = request(rid, req, "create");
      }
      Create.Result result = expect(resp, Response.ResultCase.CREATE, "create").getCreate();
      switch (result.getOutcomeCase()) {
        case OK -> {
          Create.Result.Ok ok = result.getOk();
          if (ok.getPath().isEmpty()) {
            throw new IOException("sandbox backend create response missing 'path'");
          }
          return new CreateResponse(
              ok.getPath(), ok.hasCustom() ? ok.getCustom() : null, ok.hasUnconfined());
        }
        case MISSING_CONTENT -> {
          if (attempt >= MAX_CREATE_ATTEMPTS) {
            throw new IOException(
                "sandbox backend still reports missing content after "
                    + attempt
                    + " create attempts for sandbox "
                    + sandboxId);
          }
          pushMissing(result.getMissingContent().getDigestHashesList(), blobs, content);
        }
        case ERROR -> throw failed("create", result.getError());
        case OUTCOME_NOT_SET -> throw noOutcome("create");
      }
    }
  }

  /**
   * Pushes the reusable entries (blobs and {@code RETENTION_REUSE} content) this controller has not
   * been told about yet, plus every single-use content entry; a no-op when there is nothing to send.
   *
   * <p>Reusable digests are reserved optimistically — marked pushed before the frame is flushed — so
   * a thread that loses the race with a Create reading the store first self-corrects via that
   * Create's MissingContent. Single-use content is sent every time and never recorded: it is unique
   * per action, so tracking it would only grow {@link #pushed} without ever yielding a reuse hit.
   */
  private void pushNovel(Map<String, ByteString> blobs, Map<String, Content> content)
      throws IOException {
    LinkedHashMap<String, ByteString> novelBlobs = new LinkedHashMap<>();
    for (Map.Entry<String, ByteString> entry : blobs.entrySet()) {
      if (pushed.add(entry.getKey())) {
        novelBlobs.put(entry.getKey(), entry.getValue());
      }
    }
    LinkedHashMap<String, Content> novelContent = new LinkedHashMap<>();
    for (Map.Entry<String, Content> entry : content.entrySet()) {
      if (entry.getValue().getRetention() == Retention.RETENTION_SINGLE_USE) {
        novelContent.put(entry.getKey(), entry.getValue());
      } else if (pushed.add(entry.getKey())) {
        novelContent.put(entry.getKey(), entry.getValue());
      }
    }
    push(novelBlobs, novelContent);
  }

  /**
   * Re-pushes exactly the digests the controller reported missing, from this action's blobs and
   * content. The content map holds this build's live locations, so a re-pushed location is captured
   * from a path valid for the current build.
   */
  private void pushMissing(
      List<String> digestHashes, Map<String, ByteString> blobs, Map<String, Content> content)
      throws IOException {
    LinkedHashMap<String, ByteString> missingBlobs = new LinkedHashMap<>();
    LinkedHashMap<String, Content> missingContent = new LinkedHashMap<>();
    for (String hash : digestHashes) {
      ByteString blob = blobs.get(hash);
      if (blob != null) {
        missingBlobs.put(hash, blob);
        pushed.add(hash);
        continue;
      }
      Content entry = content.get(hash);
      if (entry != null) {
        missingContent.put(hash, entry);
        if (entry.getRetention() != Retention.RETENTION_SINGLE_USE) {
          pushed.add(hash);
        }
        continue;
      }
      // The controller asked for a digest that is not in this action's input tree; we cannot satisfy
      // it, so the request is unrecoverable.
      throw new IOException(
          "sandbox backend requested content not in the action's input tree: " + hash);
    }
    push(missingBlobs, missingContent);
  }

  /**
   * Writes a {@code push} frame carrying {@code blobs} and {@code content} into the controller's
   * store. Fire-and-forget: there is no reply. Ordering (a push before any Create that reads it)
   * comes from the single ordered write pipe, not from an acknowledgement; a lost entry surfaces as
   * a later Create's MissingContent. A no-op when both maps are empty.
   */
  private void push(Map<String, ByteString> blobs, Map<String, Content> content)
      throws IOException {
    if (blobs.isEmpty() && content.isEmpty()) {
      return;
    }
    long rid = nextRid.getAndIncrement();
    Request req =
        Request.newBuilder()
            .setRid(rid)
            .setPush(Push.newBuilder().putAllBlobs(blobs).putAllContent(content))
            .build();
    try (SilentCloseable c =
        Profiler.instance()
            .profile(
                "sandbox.push("
                    + blobs.size()
                    + " blobs, "
                    + content.size()
                    + " content, "
                    + req.getSerializedSize()
                    + "B)")) {
      synchronized (writeLock) {
        if (!alive) {
          throw new IOException("sandbox backend at " + binary + " is not running");
        }
        req.writeDelimitedTo(out);
        out.flush();
      }
    } catch (IOException e) {
      markDead(e);
      throw e;
    }
  }

  /**
   * Sends a {@code collect} request: the controller moves the action's declared outputs out of the
   * sandbox to their place under {@code execRootParent} (the directory containing the workspace
   * exec root). Blocks until the move completes; throws on controller error.
   */
  void collectOutputs(String sandboxId, String execRootParent) throws IOException {
    long rid = nextRid.getAndIncrement();
    Response resp =
        request(
            rid,
            Request.newBuilder()
                .setRid(rid)
                .setSandboxId(sandboxId)
                .setCollect(Collect.newBuilder().setExecRoot(execRootParent))
                .build(),
            "collect");
    Collect.Result result = expect(resp, Response.ResultCase.COLLECT, "collect").getCollect();
    switch (result.getOutcomeCase()) {
      case OK -> {}
      case ERROR -> throw failed("collect", result.getError());
      case OUTCOME_NOT_SET -> throw noOutcome("collect");
    }
  }

  /** Sends a {@code destroy} request for the given sandbox id. */
  void destroySandbox(String sandboxId) throws IOException {
    long rid = nextRid.getAndIncrement();
    Response resp =
        request(
            rid,
            Request.newBuilder()
                .setRid(rid)
                .setSandboxId(sandboxId)
                .setDestroy(Destroy.getDefaultInstance())
                .build(),
            "destroy");
    Destroy.Result result = expect(resp, Response.ResultCase.DESTROY, "destroy").getDestroy();
    switch (result.getOutcomeCase()) {
      case OK -> {}
      case ERROR -> throw failed("destroy", result.getError());
      case OUTCOME_NOT_SET -> throw noOutcome("destroy");
    }
  }

  private Response request(long rid, Request req, String op) throws IOException {
    CompletableFuture<Response> future = new CompletableFuture<>();
    pending.put(rid, future);
    try {
      synchronized (writeLock) {
        if (!alive) {
          throw new IOException("sandbox backend at " + binary + " is not running");
        }
        // Varint length-delimited framing (protobuf writeDelimitedTo / parseDelimitedFrom).
        req.writeDelimitedTo(out);
        out.flush();
      }
    } catch (IOException e) {
      pending.remove(rid);
      markDead(e);
      throw e;
    }
    Response resp;
    try {
      resp = future.get(requestTimeout.toMillis(), TimeUnit.MILLISECONDS);
    } catch (InterruptedException e) {
      pending.remove(rid);
      Thread.currentThread().interrupt();
      throw new IOException("interrupted waiting for sandbox backend response", e);
    } catch (ExecutionException e) {
      pending.remove(rid);
      Throwable cause = e.getCause();
      if (cause instanceof IOException ioException) {
        throw ioException;
      }
      throw new IOException("sandbox backend request failed", cause);
    } catch (TimeoutException e) {
      // The controller never answered within the deadline — wedged or its stdout frame stream is
      // corrupted. Tear it down so this build fails now instead of parking forever, and so the next
      // action respawns a fresh controller. killController marks every other in-flight request dead
      // too, so they fail immediately rather than each burning its own full timeout.
      pending.remove(rid);
      IOException io =
          new IOException(
              "sandbox backend "
                  + op
                  + " timed out after "
                  + requestTimeout.toSeconds()
                  + "s (controller wedged or protocol stream corrupted)"
                  + stderrSuffix());
      killController(io);
      throw io;
    }
    return resp;
  }

  /**
   * Asserts the reply carries the result arm its request op expects (they are paired one-to-one, so
   * a mismatch is a controller protocol violation), and returns it for the caller to unwrap.
   */
  private static Response expect(Response resp, Response.ResultCase expected, String op)
      throws IOException {
    if (resp.getResultCase() != expected) {
      throw new IOException(
          "sandbox backend " + op + " got an unexpected response arm: " + resp.getResultCase());
    }
    return resp;
  }

  /** Turns an operation's {@code error} outcome into the exception to throw. */
  private static IOException failed(String op, Error error) {
    String message = error.getMessage();
    return new IOException(
        "sandbox backend " + op + " failed: " + (message.isEmpty() ? "(no error message)" : message));
  }

  /** Thrown when a result carries no outcome — a controller that answered with an empty oneof. */
  private static IOException noOutcome(String op) {
    return new IOException("sandbox backend " + op + " returned no outcome");
  }

  private void readLoop() {
    try {
      // parseDelimitedFrom reads one varint length-delimited Response per call, returning null at
      // clean EOF (controller closed its stdout).
      Response resp;
      while ((resp = Response.parseDelimitedFrom(in)) != null) {
        dispatch(resp);
      }
      logger.at(Level.FINE).log("sandbox backend closed its stdout");
    } catch (IOException e) {
      logger.at(Level.FINE).withCause(e).log("sandbox backend reader stopped");
    }
    markDead(new IOException(exitDiagnostic()));
  }

  /**
   * Continuously copies the controller's stderr into {@link #stderrTail}, capped at the last {@link
   * #STDERR_TAIL_LIMIT} bytes, and mirrors each line into the log. Draining is mandatory: if nobody
   * read it, a chatty controller would eventually block on a full stderr pipe.
   */
  private void drainStderr() {
    try (BufferedReader reader =
        new BufferedReader(
            new InputStreamReader(process.getErrorStream(), StandardCharsets.UTF_8))) {
      String line;
      while ((line = reader.readLine()) != null) {
        logger.at(Level.FINE).log("sandbox backend stderr: %s", line);
        synchronized (stderrTail) {
          stderrTail.append(line).append('\n');
          int overflow = stderrTail.length() - STDERR_TAIL_LIMIT;
          if (overflow > 0) {
            stderrTail.delete(0, overflow);
          }
        }
      }
    } catch (IOException e) {
      logger.at(Level.FINE).withCause(e).log("sandbox backend stderr reader stopped");
    }
  }

  /**
   * Builds the death message for a controller that exited on its own, enriching the bare "exited"
   * with its exit code and the tail of its stderr — the controller's only channel to explain why it
   * died (panic, rejected manifest, missing input). This message becomes the cause of the
   * "Could not copy inputs into sandbox" failure the user sees, so the diagnosis must travel with it.
   */
  private String exitDiagnostic() {
    StringBuilder msg = new StringBuilder("sandbox backend exited");
    // stderr hits EOF when the process exits; wait briefly so the tail is complete before we read
    // it, then reap the process so its exit code is available.
    try {
      stderrThread.join(1000);
      process.waitFor();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
    if (process.finished()) {
      msg.append(" with exit code ").append(process.exitValue());
    }
    return msg.append(stderrSuffix()).toString();
  }

  /**
   * The captured tail of the controller's stderr, formatted for appending to a diagnostic, or the
   * empty string if it produced none. Snapshots {@link #stderrTail} as-is — unlike {@link
   * #exitDiagnostic} it does not wait for EOF, because the timeout path runs while the (wedged)
   * controller is still alive and its stderr stream will never close.
   */
  private String stderrSuffix() {
    synchronized (stderrTail) {
      String stderr = stderrTail.toString().strip();
      return stderr.isEmpty() ? "" : "\nsandbox backend stderr:\n" + stderr;
    }
  }

  /**
   * Forcibly destroys a wedged controller and fails every in-flight request with {@code cause}.
   * Unlike {@link #close}, this kills the process: a controller that blew its read deadline cannot
   * hand off warm state, so there is nothing to preserve. Marking all pending requests dead is what
   * stops a corrupted stream from costing more than one timeout — sibling threads parked in {@link
   * #request} unblock immediately instead of each waiting out the full deadline.
   */
  private void killController(IOException cause) {
    process.destroy();
    markDead(cause);
  }

  private void dispatch(Response resp) {
    CompletableFuture<Response> future = pending.remove(resp.getRid());
    if (future == null) {
      logger.atWarning().log("sandbox backend response for unknown rid %d", resp.getRid());
      return;
    }
    future.complete(resp);
  }

  private void markDead(IOException cause) {
    alive = false;
    for (CompletableFuture<Response> future : pending.values()) {
      future.completeExceptionally(cause);
    }
    pending.clear();
  }

  /**
   * Closes stdin (signalling clean exit) and waits for the process to terminate. NEVER kills it: it
   * must exit on its own so the next controller can reuse its warm state. In-flight requests fail
   * with an IOException.
   */
  void close() {
    try {
      synchronized (writeLock) {
        try {
          out.close();
        } catch (IOException ignored) {
        }
      }
      try {
        process.waitFor();
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        process.destroy();
      }
    } finally {
      markDead(new IOException("sandbox backend closed"));
    }
  }
}
