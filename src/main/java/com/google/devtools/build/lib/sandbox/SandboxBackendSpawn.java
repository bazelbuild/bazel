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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.sandbox.proto.SandboxProto.Confinement;
import com.google.devtools.build.lib.sandbox.proto.SandboxProto.Manifest;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.StringEncoding;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.logging.Level;
import javax.annotation.Nullable;

/**
 * A {@link SandboxedSpawn} that delegates filesystem isolation to the long-lived sandbox backend
 * controller. {@link #createFileSystem} and {@link #delete} round-trip length-prefixed proto frames
 * over the shared {@link SandboxBackendServer}'s stdin/stdout pipe instead of spawning a per-action
 * subprocess.
 *
 * <p>Lifecycle:
 *
 * <ol>
 *   <li>{@link #createFileSystem} sends {@code Request{rid, create: Manifest}} and reads back
 *       {@code Response{created: {path}}}; {@code sandboxExecRoot} is that path plus the workspace
 *       name.
 *   <li>The action runs with {@code cwd = sandboxExecRoot}.
 *   <li>{@link #copyOutputs} sends {@code Request{rid, collect}} so the controller moves the
 *       action's outputs out to the real execroot.
 *   <li>{@link #delete} sends {@code Request{rid, destroy}} and asynchronously removes the
 *       scratch dir.
 * </ol>
 */
final class SandboxBackendSpawn implements SandboxedSpawn {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /** macOS Seatbelt driver. Overridable in tests, since real sandbox-exec can't nest. */
  @VisibleForTesting static String sandboxExecBinary = "/usr/bin/sandbox-exec";

  private final SandboxBackendServer daemon;
  private final Path scratchDir;
  private final Manifest manifest;
  private final String sandboxId;
  private final String workspaceName;
  // The action argv (process-wrapper + action), before any confinement wrapping.
  private final ImmutableList<String> innerArguments;
  private final ImmutableMap<String, String> environment;
  private final TreeDeleter treeDeleter;
  private final String mnemonic;
  @Nullable private final Path statisticsPath;
  // Confinement inputs: host paths the confined action may write, paths it may not read, whether
  // network is allowed, and where to write the generated Seatbelt profile.
  private final ImmutableSet<Path> writableDirs;
  private final ImmutableSet<Path> inaccessiblePaths;
  private final boolean allowNetwork;
  private final Path confinementProfilePath;
  // The linux-sandbox helper binary, used only for CONFINEMENT_LINUX_NAMESPACES.
  private final Path linuxSandboxPath;

  /** Resolved on {@link #createFileSystem}; null before. */
  @Nullable private Path sandboxExecRoot;

  /** The confinement-wrapped argv, resolved on {@link #createFileSystem}; null before. */
  @Nullable private ImmutableList<String> arguments;

  /** Guards against double-destroy when both {@link #copyOutputs} and {@link #delete} run. */
  private boolean slotReleased;

  SandboxBackendSpawn(
      SandboxBackendServer daemon,
      Path scratchDir,
      Manifest manifest,
      String sandboxId,
      String workspaceName,
      ImmutableList<String> innerArguments,
      ImmutableMap<String, String> environment,
      TreeDeleter treeDeleter,
      String mnemonic,
      @Nullable Path statisticsPath,
      ImmutableSet<Path> writableDirs,
      ImmutableSet<Path> inaccessiblePaths,
      boolean allowNetwork,
      Path confinementProfilePath,
      Path linuxSandboxPath) {
    this.daemon = daemon;
    this.scratchDir = scratchDir;
    this.manifest = manifest;
    this.sandboxId = sandboxId;
    this.workspaceName = workspaceName;
    this.innerArguments = innerArguments;
    this.environment = environment;
    this.treeDeleter = treeDeleter;
    this.mnemonic = mnemonic;
    this.statisticsPath = statisticsPath;
    this.writableDirs = writableDirs;
    this.inaccessiblePaths = inaccessiblePaths;
    this.allowNetwork = allowNetwork;
    this.confinementProfilePath = confinementProfilePath;
    this.linuxSandboxPath = linuxSandboxPath;
  }

  @Override
  public Path getSandboxExecRoot() {
    if (sandboxExecRoot == null) {
      throw new IllegalStateException("sandboxExecRoot not resolved; createFileSystem() not called");
    }
    return sandboxExecRoot;
  }

  @Override
  public ImmutableList<String> getArguments() {
    if (arguments == null) {
      throw new IllegalStateException("arguments not resolved; createFileSystem() not called");
    }
    return arguments;
  }

  @Override
  public ImmutableMap<String, String> getEnvironment() {
    return environment;
  }

  @Override
  public String getMnemonic() {
    return mnemonic;
  }

  @Override
  @Nullable
  public Path getSandboxDebugPath() {
    return null;
  }

  @Override
  @Nullable
  public Path getStatisticsPath() {
    return statisticsPath;
  }

  @Override
  public boolean useSubprocessTimeout() {
    // process-wrapper (wrapped around the action argv by the runner) handles the timeout.
    return false;
  }

  @Override
  public void createFileSystem() throws IOException {
    SandboxBackendServer.CreateResponse response;
    try (SilentCloseable c = Profiler.instance().profile("sandbox.createFileSystem.daemonCreate")) {
      response = daemon.createSandbox(sandboxId, manifest);
    }
    // The daemon's stdout is decoded as proper UTF-8 (real Unicode String); convert back to
    // Bazel's internal encoding before constructing a Path, otherwise non-ASCII paths get
    // mishandled in subsequent filesystem ops.
    Path root =
        scratchDir.getFileSystem().getPath(StringEncoding.platformToInternal(response.path()));
    if (!root.exists()) {
      throw new IOException(
          "sandbox backend returned a non-existent sandbox root: " + root.getPathString());
    }
    sandboxExecRoot = root.getRelative(workspaceName);
    // Confine the action with the mechanism the backend chose for this sandbox (or Bazel's platform
    // default), reusing Bazel's existing per-OS confinement. The backend materialized the filesystem
    // view (the mount at `root`); we only add the process jail on top.
    arguments = confine(resolveConfinement(response.confinement()), root);
  }

  /**
   * Resolves the confinement to apply: the backend's choice, or Bazel's platform default when it is
   * unspecified. Fails closed — never silently unconfined — if the backend chose a confinement Bazel
   * did not advertise as supported on this host (the same set sent in {@code Negotiate}).
   */
  private Confinement resolveConfinement(Confinement requested) throws IOException {
    Confinement effective =
        requested == Confinement.CONFINEMENT_UNSPECIFIED
            ? SandboxBackendConfinement.platformDefault()
            : requested;
    if (!SandboxBackendConfinement.supportedOnThisPlatform().contains(effective)) {
      throw new IOException(
          "sandbox backend selected confinement "
              + effective
              + ", which Bazel did not advertise as supported on this host "
              + SandboxBackendConfinement.supportedOnThisPlatform());
    }
    return effective;
  }

  /** Wraps {@link #innerArguments} in the chosen confinement, writing any profile it needs. */
  private ImmutableList<String> confine(Confinement confinement, Path sandboxRoot)
      throws IOException {
    if (confinement == Confinement.CONFINEMENT_NONE) {
      return innerArguments;
    }
    // The action writes into the mount (sandboxRoot) plus the configured writable dirs.
    ImmutableSet<Path> writable =
        ImmutableSet.<Path>builder().addAll(writableDirs).add(sandboxRoot).build();
    return switch (confinement) {
      case CONFINEMENT_SEATBELT -> {
        DarwinSandboxCommandLineBuilder.writeProfile(
            confinementProfilePath, writable, inaccessiblePaths, allowNetwork, statisticsPath);
        yield DarwinSandboxCommandLineBuilder.wrapCommand(
            sandboxExecBinary, confinementProfilePath, innerArguments);
      }
      case CONFINEMENT_LINUX_NAMESPACES ->
          LinuxSandboxCommandLineBuilder.commandLineBuilder(linuxSandboxPath)
              .setWritableFilesAndDirectories(writable)
              .setCreateNetworkNamespace(
                  allowNetwork
                      ? LinuxSandboxCommandLineBuilder.NetworkNamespace.NO_NETNS
                      : LinuxSandboxCommandLineBuilder.NetworkNamespace.NETNS_WITH_LOOPBACK)
              .buildForCommand(innerArguments);
      default -> throw new IOException("unsupported confinement " + confinement);
    };
  }

  @Override
  public void copyOutputs(Path execRoot) throws IOException, InterruptedException {
    try {
      // Delegate output collection to the controller: it owns the sandbox and knows where the
      // action's outputs landed, so it moves them out. We pass the parent of the workspace exec
      // root — the controller anchors each output's sandbox-relative path under it, mirroring how
      // it placed them in the sandbox.
      daemon.collectOutputs(sandboxId, execRoot.getParentDirectory().getPathString());
    } finally {
      // Release the daemon-side slot as soon as outputs are off the mount. Under --sandbox_debug,
      // AbstractSandboxSpawnRunner skips delete() to preserve scratch state, so releasing here
      // keeps the pool from filling up over a long debug build; debug needs the dumped
      // manifest.json + scratch dir, not the live mount/slot.
      releaseSlot();
    }
  }

  @Override
  public void delete() {
    // Idempotent: copyOutputs releases the slot on the happy path; this covers paths that bypass
    // copyOutputs (createFileSystem / run() threw first).
    releaseSlot();
    try {
      treeDeleter.deleteTree(scratchDir);
    } catch (IOException e) {
      // Same reasoning as AbstractContainerizingSandboxedSpawn.delete().
    }
  }

  private void releaseSlot() {
    if (slotReleased) {
      return;
    }
    slotReleased = true;
    try {
      daemon.destroySandbox(sandboxId);
    } catch (IOException e) {
      // The build is done; cleanup failure is annoying but not fatal — server shutdown / unmount
      // reclaims resources.
      logger.at(Level.WARNING).withCause(e).log(
          "sandbox backend failed to destroy sandbox %s", sandboxId);
    }
  }
}
