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
package com.google.devtools.build.lib.sandbox.backend;

import com.google.common.collect.ImmutableMap;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.exec.SpawnStrategyRegistry;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Registers the sandbox backends configured with {@code --sandbox_backend} as spawn strategies and
 * keeps their processes.
 *
 * <p>A backend is launched when it is needed for the first time and then lives as long as the
 * Bazel server, unless it dies or its configuration changes, in which case a new one is launched
 * the next time it is needed.
 */
public final class SandboxBackends {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final Object LOCK = new Object();
  // @GuardedBy("LOCK")
  private static final Map<String, Running> running = new HashMap<>();
  // @GuardedBy("LOCK")
  private static boolean shutdownHookInstalled;

  private record Running(BackendConfig config, SandboxBackendSession session) {}

  private SandboxBackends() {}

  /**
   * Registers a strategy for each configured backend.
   *
   * @param backends the values of {@code --sandbox_backend}
   * @param backendOptions the values of {@code --sandbox_backend_opt}
   * @param runners the set that spawn runners are added to
   * @throws IOException if the configuration is invalid
   */
  public static void register(
      CommandEnvironment env,
      List<Map.Entry<String, String>> backends,
      List<Map.Entry<String, String>> backendOptions,
      ExecutionOptions executionOptions,
      SpawnStrategyRegistry.Builder registryBuilder,
      Set<SpawnRunner> runners)
      throws IOException {
    ImmutableMap<String, BackendConfig> configs = BackendConfig.fromFlags(backends, backendOptions);
    // Backends outlive commands, so apply changes and removals now, even if no spawn runs.
    reconcile(configs);
    if (configs.isEmpty()) {
      return;
    }
    DigestUtil digestUtil =
        new DigestUtil(
            env.getSyscallCache(), env.getExecRoot().getFileSystem().getDigestFunction());
    var scratchRoot = env.getOutputBase().getRelative("sandbox-backend");
    scratchRoot.deleteTree();
    scratchRoot.createDirectoryAndParents();
    for (Map.Entry<String, BackendConfig> entry : configs.entrySet()) {
      String name = entry.getKey();
      BackendConfig config = entry.getValue();
      SpawnRunner runner =
          new SandboxBackendSpawnRunner(
              name,
              () -> getOrStart(name, config, env.getClientEnv(), digestUtil),
              () -> config.resolveBinary(env.getClientEnv()) != null,
              env.getExecRoot(),
              scratchRoot.getRelative(name),
              env.getWorkspaceName(),
              env.getBuildRequestId(),
              env.getCommandId().toString());
      runners.add(runner);
      registryBuilder.registerStrategy(
          new SandboxBackendStrategy(name, runner, executionOptions),
          SandboxBackendStrategy.STRATEGY_NAME_PREFIX + name);
    }
  }

  /**
   * Stops the backends that aren't configured any more, or are configured differently than they
   * were when they were launched.
   */
  static void reconcile(ImmutableMap<String, BackendConfig> configs) {
    synchronized (LOCK) {
      Iterator<Map.Entry<String, Running>> it = running.entrySet().iterator();
      while (it.hasNext()) {
        Map.Entry<String, Running> entry = it.next();
        if (!entry.getValue().config().equals(configs.get(entry.getKey()))) {
          logger.atInfo().log("sandbox backend %s changed or was removed, stopping it", entry.getKey());
          it.remove();
          entry.getValue().session().close();
        }
      }
    }
  }

  /** Returns the session of the backend, launching it if it isn't running. */
  static SandboxBackendSession getOrStart(
      String name,
      BackendConfig config,
      ImmutableMap<String, String> clientEnv,
      DigestUtil digestUtil)
      throws IOException, InterruptedException {
    synchronized (LOCK) {
      Running existing = running.get(name);
      if (existing != null && existing.config().equals(config) && existing.session().isAlive()) {
        return existing.session();
      }
      if (existing != null) {
        existing.session().close();
        running.remove(name);
      }
      SandboxBackendSession session = SandboxBackendSession.start(name, config, clientEnv, digestUtil);
      running.put(name, new Running(config, session));
      if (!shutdownHookInstalled) {
        shutdownHookInstalled = true;
        Runtime.getRuntime()
            .addShutdownHook(
                new Thread(
                    () -> {
                      synchronized (LOCK) {
                        running.values().forEach(r -> r.session().close());
                      }
                    },
                    "sandbox-backend-shutdown"));
      }
      return session;
    }
  }

  /** Stops all backends. */
  static void stopAll() {
    synchronized (LOCK) {
      running.values().forEach(r -> r.session().close());
      running.clear();
    }
  }
}
