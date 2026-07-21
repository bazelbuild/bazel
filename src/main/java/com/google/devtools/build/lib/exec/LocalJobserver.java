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

import com.google.common.collect.ImmutableMap;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.UserExecException;
import java.io.IOException;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A GNU make protocol jobserver whose token count continuously tracks Bazel's idle local CPU
 * capacity.
 *
 * <p>Standalone local actions that declare {@link ExecutionRequirements#SUPPORTS_JOBSERVER} get a
 * {@code MAKEFLAGS} environment variable pointing at the jobserver (a fifo on Linux/macOS, a named
 * semaphore on Windows; see {@link Backend}). Jobserver-aware tools such as {@code rustc} and
 * {@code make} take a token for every thread of parallelism beyond their first implicit one, and
 * give it back when the thread finishes. On POSIX the emitted auth style is {@code fifo:}; GNU make
 * < 4.4 aborts rather than ignores it, so tagging an action is a promise about the tool's version
 * too (see {@link ExecutionRequirements#SUPPORTS_JOBSERVER}). Bazel continuously retargets the
 * distributed tokens to the number of CPUs not currently reserved by running actions, so internal
 * tool parallelism expands to use idle cores and contracts when Bazel wants to schedule more
 * actions. This assumes the action reserves the CPU used by its implicit slot.
 *
 * <p>Only held tokens are charged to {@link ResourceManager}; idle tokens in the pool are
 * reclaimable. A tool can take an idle token in the same poll window that Bazel admits another
 * action, causing bounded oversubscription until that token is returned.
 *
 * <p>This class is a <b>platform-agnostic coordinator</b>. It owns the poll loop, publishes the
 * held-token count to {@link ResourceManager}, and injects {@code MAKEFLAGS}; the actual primitive
 * (fifo vs. named semaphore) and all its OS-specific accounting live behind {@link Backend}, which
 * a higher layer ({@code ExecutionTool}) selects and injects via {@link #configure}. The
 * coordinator itself therefore carries no dependency on any platform library.
 *
 * <p>The jobserver's auth string is deliberately not part of any action cache key: it is injected
 * by local spawn runners after cache lookup, like TMPDIR, and never appears in the {@link Spawn}'s
 * declared environment. Remote and persistent-worker spawns never see {@code MAKEFLAGS}.
 *
 * <p>The instance is a process-wide singleton because there is exactly one local token pool per
 * Bazel server.
 */
public final class LocalJobserver {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final LocalJobserver INSTANCE = new LocalJobserver();

  public static LocalJobserver instance() {
    return INSTANCE;
  }

  public static final String MAKEFLAGS = "MAKEFLAGS";

  /** How often the token pool is re-targeted against ResourceManager's idle CPU count. */
  public static final long POLL_MILLIS = 100;

  /**
   * The platform primitive backing the token pool: a fifo on Linux/macOS, a named semaphore on
   * Windows. Owns the shared per-tick token accounting and leaves only the primitive-specific peek
   * and refill to subclasses, which {@code ExecutionTool} selects and injects via {@link
   * #configure}. Both backends use one vocabulary:
   *
   * <ul>
   *   <li><b>issued</b>: tokens this manager has put into the pool and not reclaimed (in
   *       circulation).
   *   <li><b>available</b>: idle tokens currently sitting in the pool, counted by {@link
   *       #drainPool}.
   *   <li><b>held</b>: {@code issued - available}, i.e. tokens taken by tools and not yet returned.
   *   <li><b>desired</b>: {@code max(0, target - held)}, the idle tokens to leave in the pool.
   * </ul>
   */
  public abstract static class Backend {
    // Tokens in circulation: put into the pool by this manager and not yet reclaimed. Bounded by the
    // target (a small CPU count), so unlike a cumulative counter it never grows without bound.
    private int issued = 0;

    /**
     * Sets up the primitive and returns the value that follows {@code --jobserver-auth=} in {@code
     * MAKEFLAGS} (e.g. {@code fifo:/path/to/fifo} on Linux/macOS, or a bare named-semaphore name on
     * Windows). Called once, before any {@link #tick}.
     */
    public abstract String start() throws IOException;

    /**
     * Resizes the token pool toward {@code targetTokens} (growing or shrinking as needed) and
     * returns the current held-token estimate: tokens taken by tools and not yet returned. Called
     * once per poll tick.
     */
    public final int tick(int targetTokens) throws IOException {
      int available = drainPool();
      int held = Math.max(0, issued - available);
      int desired = Math.max(0, targetTokens - held);
      refillPool(desired);
      issued = held + desired;
      return held;
    }

    /** Removes and returns the count of every idle token currently in the pool. */
    protected abstract int drainPool() throws IOException;

    /** Adds {@code count} idle tokens back into the pool; a no-op when {@code count <= 0}. */
    protected abstract void refillPool(int count) throws IOException;

    /**
     * Directory a sandbox must whitelist as writable so a tool can return tokens, or null if the
     * primitive has no filesystem presence (e.g. a named semaphore). Valid only after {@link
     * #start}. Defaults to null; the fifo backend overrides it.
     */
    @Nullable
    public String writableDir() {
      return null;
    }

    /** Releases all resources. */
    public abstract void close();
  }

  // Non-null iff the jobserver is active: the value that follows "--jobserver-auth=" in MAKEFLAGS.
  // Written under `this` lock, read anywhere.
  @Nullable private volatile String jobserverAuth;

  // Directory sandboxes must whitelist as writable so a tool can return tokens. Null when the
  // active backend has no filesystem presence (i.e. the Windows named semaphore).
  @Nullable private volatile String writableDir;

  // Held-token estimate published by the manager thread for ResourceManager's scheduling checks.
  private volatile int outstandingTokens;

  @Nullable private Thread manager;
  @Nullable private Backend backend;
  @Nullable private ResourceManager resourceManager;
  private volatile boolean running;

  private LocalJobserver() {}

  /**
   * Enables the jobserver with the given {@code backend}, or disables it if {@code backend} is
   * null. Called once per build; restarts from a clean slate so tokens leaked by crashed clients
   * of a previous build are reclaimed.
   */
  public synchronized void configure(@Nullable Backend backend, ResourceManager resourceManager)
      throws IOException {
    shutdown();
    if (backend == null) {
      return;
    }
    String auth;
    try {
      auth = backend.start();
    } catch (IOException e) {
      backend.close();
      throw e;
    }
    this.backend = backend;
    this.resourceManager = resourceManager;
    this.writableDir = backend.writableDir();
    this.jobserverAuth = auth;
    this.running = true;
    this.manager = new Thread(this::run, "local-jobserver");
    this.manager.setDaemon(true);
    this.manager.start();
    logger.atInfo().log("Local jobserver started with auth %s", auth);
  }

  /** Stops the manager thread and releases backend resources. No-op if not active. */
  public synchronized void shutdown() {
    if (jobserverAuth == null) {
      return;
    }
    jobserverAuth = null;
    writableDir = null;
    running = false;
    if (manager != null) {
      manager.interrupt();
      try {
        manager.join(1000);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }
    }
    backend.close();
    manager = null;
    backend = null;
    resourceManager = null;
    outstandingTokens = 0;
    logger.atInfo().log("Local jobserver stopped");
  }

  /**
   * Adds the jobserver's MAKEFLAGS to {@code env} if the jobserver is active, the spawn opted in
   * via {@link ExecutionRequirements#SUPPORTS_JOBSERVER}, and the spawn does not already set
   * MAKEFLAGS itself.
   */
  public ImmutableMap<String, String> maybeAddJobserver(
      ImmutableMap<String, String> env, Spawn spawn) {
    String auth = jobserverAuth;
    if (auth == null || env.containsKey(MAKEFLAGS) || !Spawns.supportsJobserver(spawn)) {
      return env;
    }
    return ImmutableMap.<String, String>builderWithExpectedSize(env.size() + 1)
        .putAll(env)
        .put(MAKEFLAGS, "--jobserver-auth=" + auth)
        .buildOrThrow();
  }

  /**
   * Returns the directory sandboxes must whitelist as writable if the given (already rewritten)
   * spawn environment references this jobserver, or null otherwise — including when the active
   * backend has no filesystem presence (i.e. the Windows named semaphore).
   */
  @Nullable
  public String getWritableDirForEnv(Map<String, String> env) {
    String dir = writableDir;
    String auth = jobserverAuth;
    String makeflags = env.get(MAKEFLAGS);
    if (dir == null || auth == null || makeflags == null || !makeflags.contains(auth)) {
      return null;
    }
    return dir;
  }

  /**
   * Estimate of tokens currently held by running tools (updated every {@link #POLL_MILLIS}).
   * {@link ResourceManager} counts these as in-use CPUs when admitting actions.
   */
  public int getOutstandingTokens() {
    return outstandingTokens;
  }

  /** Whether the jobserver is currently active (i.e. {@code --experimental_local_jobserver}). */
  public boolean isActive() {
    return jobserverAuth != null;
  }

  /**
   * The poll loop: re-target the pool against ResourceManager's idle CPU count, publish the
   * held-token estimate, and re-admit waiting actions whenever held tokens drop (a tool returned
   * tokens, freeing CPUs). All platform specifics live in {@link Backend#tick}.
   */
  private void run() {
    ResourceManager rm = resourceManager;
    int lastOutstanding = 0;
    try {
      while (running) {
        // A tool's implicit slot is the CPU already reserved by its Bazel action; tokens represent
        // only additional parallel work.
        int target = Math.max(0, (int) rm.getIdleCpuForJobserver());
        int held = backend.tick(target);
        outstandingTokens = held;
        if (held < lastOutstanding) {
          // Tokens came back; actions waiting on CPU may be admissible now.
          notifyResourcesFreed(rm);
        }
        lastOutstanding = held;
        Thread.sleep(POLL_MILLIS);
      }
    } catch (InterruptedException e) {
      // Shutdown.
    } catch (Exception e) {
      if (running) {
        // The manager thread is gone, so the pool can no longer be resized. Stop charging held
        // tokens against ResourceManager's CPU budget so we don't subtract from available CPU on
        // every admission for the rest of the build (see ResourceManager#isCpuAvailable), silently
        // under-scheduling it.
        outstandingTokens = 0;
        logger.atWarning().withCause(e).log(
            "Local jobserver manager died; tokens no longer tracked");
      }
    }
  }

  /**
   * Re-admits actions waiting on freed CPU, without letting their failures kill the manager.
   * Exceptions here belong to the waiting action being admitted (e.g. borrowing its worker), not
   * to the manager's own work; the request stays queued and is retried on the next resource
   * release, so the pool-resizing loop must keep running.
   */
  private static void notifyResourcesFreed(ResourceManager rm) throws InterruptedException {
    try {
      rm.notifyResourcesFreed();
    } catch (IOException | UserExecException e) {
      logger.atWarning().withCause(e).log("Local jobserver: failed to re-admit waiting actions");
    }
  }
}
