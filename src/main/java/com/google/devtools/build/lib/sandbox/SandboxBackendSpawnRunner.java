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
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.LostInputsExecException;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileStateType;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.VirtualActionInput;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.remote.merkletree.MerkleTree;
import com.google.devtools.build.lib.remote.merkletree.MerkleTreeComputer;
import com.google.devtools.build.lib.remote.merkletree.MerkleTreeComputer.BlobPolicy;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.ProcessWrapper;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.sandbox.proto.SandboxProto.Manifest;
import com.google.devtools.build.lib.util.StringEncoding;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.io.OutputStream;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Spawn runner that uses an external sandbox controller binary to give each action an isolated
 * filesystem view.
 *
 * <p>Builds no host-side symlink tree. Each action's view is a protobuf {@link Manifest} (schema:
 * {@code src/main/protobuf/sandbox.proto}) passed to the controller, which returns the sandbox
 * root path where the action runs.
 */
final class SandboxBackendSpawnRunner extends AbstractSandboxSpawnRunner {

  private final Path execRoot;
  private final String workspaceName;
  private final String name;
  private final PathFragment sandboxdBinary;
  private final ImmutableList<String> backendArgs;
  private final Path sandboxBase;
  // The linux-sandbox helper, used only when a backend selects CONFINEMENT_LINUX_NAMESPACES.
  private final Path linuxSandboxPath;
  private final TreeDeleter treeDeleter;
  private final LocalEnvProvider localEnvProvider;
  @Nullable private final ProcessWrapper processWrapper;
  // Reused REAPI Merkle hasher, shared with remote execution. We pass a null uploader and call
  // buildForSpawn with BlobPolicy.RETAIN_ALL, which reuses (and populates) the process-wide subtree
  // caches for digests while retaining every directory body so the whole tree can ship to the
  // controller; a null uploader means nothing is ever claimed uploaded, so remote's cache is safe.
  private final MerkleTreeComputer merkleTreeComputer;
  private final boolean available;

  /**
   * Constructs the runner and probes the controller binary once.
   *
   * <p>The strategy is *registered* whenever a controller binary is configured; this probe drives
   * {@link #canExec}, which decides whether Bazel falls through to the next strategy when the
   * binary is missing. A failed probe is silent — chains like {@code mybackend,linux-sandbox,local}
   * on a host without the binary installed are a normal configuration.
   */
  SandboxBackendSpawnRunner(
      CommandEnvironment cmdEnv,
      String name,
      PathFragment sandboxdBinary,
      ImmutableList<String> backendArgs,
      Path sandboxBase,
      TreeDeleter treeDeleter) {
    super(cmdEnv);
    this.execRoot = cmdEnv.getExecRoot();
    this.workspaceName = execRoot.getBaseName();
    this.name = name;
    this.sandboxdBinary = sandboxdBinary;
    this.backendArgs = backendArgs;
    this.sandboxBase = sandboxBase;
    this.linuxSandboxPath = LinuxSandboxUtil.getLinuxSandbox(cmdEnv.getBlazeWorkspace());
    this.treeDeleter = treeDeleter;
    this.localEnvProvider = LocalEnvProvider.forCurrentOs(cmdEnv.getClientEnv());
    this.processWrapper = ProcessWrapper.fromCommandEnvironment(cmdEnv);
    this.merkleTreeComputer =
        new MerkleTreeComputer(
            new DigestUtil(cmdEnv.getSyscallCache(), execRoot.getFileSystem().getDigestFunction()),
            /* remoteExecutionCache= */ null,
            cmdEnv.getBuildRequestId(),
            cmdEnv.getCommandId().toString(),
            workspaceName);
    // Availability is just the cheap filesystem probe. The controller daemon is spawned lazily via
    // getOrSpawn in prepareSpawn and cached as a per-server singleton.
    this.available =
        processWrapper != null
            && !sandboxdBinary.isEmpty()
            && SandboxBackendUtil.isAvailable(sandboxdBinary, cmdEnv.getClientEnv());
  }

  @Override
  public boolean canExec(Spawn spawn) {
    // When the controller binary isn't usable, decline every spawn so Bazel falls through to the
    // next --spawn_strategy. The strategy stays *registered* (whenever a controller binary is
    // configured), which is required for Bazel to accept its name in the strategy list.
    return available && super.canExec(spawn);
  }

  @Override
  protected SandboxedSpawn prepareSpawn(Spawn spawn, SpawnExecutionContext context)
      throws IOException, InterruptedException {
    String sandboxId = getName() + "-" + context.getId();
    Path scratchDir = sandboxBase.getRelative(getName()).getRelative(Integer.toString(context.getId()));
    scratchDir.createDirectoryAndParents();

    Path scratchExecRoot = scratchDir.getRelative(workspaceName);
    scratchExecRoot.createDirectoryAndParents();

    InputMetadataProvider metadataProvider = context.getInputMetadataProvider();
    SandboxOutputs outputs = SandboxHelpers.getOutputs(spawn);

    ImmutableMap<String, String> environment =
        localEnvProvider.rewriteLocalEnv(spawn.getEnvironment(), binTools, "/tmp");

    DigestHashFunction hashFunction = execRoot.getFileSystem().getDigestFunction();

    // Hash the input root with the shared REAPI computer — the same buildForSpawn path remote
    // execution uses, so subtree digests are reused from (and contributed to) the process-wide
    // caches. The sibling layout anchors tree paths under the workspace dir and the input root at its
    // parent, matching how the controller lays out the sandbox. RETAIN_ALL keeps every directory
    // body so the whole v2.Tree ships to the controller instead of being uploaded to a CAS.
    RemotePathResolver pathResolver =
        new RemotePathResolver.SiblingRepositoryLayoutResolver(execRoot);
    MerkleTree.Uploadable built;
    try (com.google.devtools.build.lib.profiler.SilentCloseable c =
        com.google.devtools.build.lib.profiler.Profiler.instance().profile("sandbox.buildTree")) {
      built =
          (MerkleTree.Uploadable)
              merkleTreeComputer.buildForSpawn(
                  spawn,
                  /* toolInputs= */ ImmutableSet.of(),
                  /* scrubber= */ null,
                  context,
                  pathResolver,
                  BlobPolicy.RETAIN_ALL);
    } catch (LostInputsExecException e) {
      // With a null uploader nothing is ever fetched from a remote cache, so this cannot happen.
      throw new IOException(e);
    }

    // locations: the only inputs whose bytes do NOT live at the default exec_root/<tree path> are
    // runfiles entries (bytes at the target's exec path) and virtual inputs (param files etc.,
    // materialized to scratch here). Everything else — including tree artifacts and source dirs — is
    // at the default. Walk the same input mapping the tree was built from; keys are exec paths and
    // tree paths are anchored under the workspace dir, exactly as buildForSpawn anchors them.
    PathFragment wsDir = PathFragment.create(workspaceName);
    LinkedHashMap<String, String> locations = new LinkedHashMap<>();
    Map<PathFragment, ActionInput> inputMap;
    try (com.google.devtools.build.lib.profiler.SilentCloseable c =
        com.google.devtools.build.lib.profiler.Profiler.instance()
            .profile("sandbox.inputMapping")) {
      inputMap =
          context.getInputMapping(PathFragment.EMPTY_FRAGMENT, /* willAccessRepeatedly= */ true);
    }
    for (Map.Entry<PathFragment, ActionInput> entry : inputMap.entrySet()) {
      collectLocation(
          entry.getKey(), entry.getValue(), wsDir, metadataProvider, scratchExecRoot, locations);
    }

    // outputs: each declared output → its kind ("file" or "dir"). The controller materializes
    // outputs in the sandbox (a dir output gets the directory itself so `tar --directory` can chdir
    // in; a file output gets only its parent, pre-created from this map) and Collect moves them out
    // after the action.
    LinkedHashMap<String, String> outputsMap =
        new LinkedHashMap<>(outputs.files().size() + outputs.dirs().size());
    for (PathFragment outputPath : outputs.files().values()) {
      String sandboxPath = "/" + workspaceName + "/" + outputPath.getPathString();
      outputsMap.put(StringEncoding.internalToPlatform(sandboxPath), "file");
    }
    for (PathFragment outputPath : outputs.dirs().values()) {
      String sandboxPath = "/" + workspaceName + "/" + outputPath.getPathString();
      outputsMap.put(StringEncoding.internalToPlatform(sandboxPath), "dir");
    }

    // writable_dirs: ephemeral writable subtrees not tied to action outputs.
    LinkedHashMap<String, String> writableDirs = new LinkedHashMap<>(2);
    Path tmpHost = scratchDir.getRelative("tmp");
    tmpHost.createDirectory();
    writableDirs.put("/tmp", StringEncoding.internalToPlatform(tmpHost.getPathString()));

    String testTmpdir = environment.get("TEST_TMPDIR");
    if (testTmpdir != null && !testTmpdir.isEmpty()) {
      Path testTmpHost = scratchDir.getRelative("test_tmpdir");
      testTmpHost.createDirectory();
      // Anchor an EXECROOT-RELATIVE TEST_TMPDIR under the workspace dir, like outputs
      // (sandboxPath = "/<workspaceName>/<execPath>"). The action's cwd is the execroot
      // (<mount>/<workspaceName>) and test-setup.sh makes TEST_TMPDIR absolute as
      // "$PWD/$TEST_TMPDIR" → <mount>/<workspaceName>/_tmp/<hash>. Without the prefix the symlink
      // lands at the bare mount root, so `mkdir -p "$TEST_TMPDIR"` hits the read-only mount →
      // "Read-only file system". (Absolute TEST_TMPDIR is left as-is.)
      String tmpKey =
          testTmpdir.startsWith("/") ? testTmpdir : "/" + workspaceName + "/" + testTmpdir;
      writableDirs.put(tmpKey, StringEncoding.internalToPlatform(testTmpHost.getPathString()));
    }

    // exec_root is the directory containing the workspace exec root; a tree path's default host is
    // exec_root/<tree path> (= execroot/<workspaceName>/<exec-path>), which is where Bazel already
    // staged it. Only locations (runfiles, virtuals) deviate.
    String execRootPrefix =
        StringEncoding.internalToPlatform(execRoot.getParentDirectory().getPathString());
    Manifest manifest =
        Manifest.newBuilder()
            .setMnemonic(StringEncoding.internalToPlatform(spawn.getMnemonic()))
            .setHashFunction(hashFunction.toString())
            .setInputRootDigest(built.digest())
            .putAllDirectories(SandboxBackendManifest.directoryBlobs(built))
            .setExecRoot(execRootPrefix)
            .putAllLocations(locations)
            .putAllOutputs(outputsMap)
            .putAllWritableDirs(writableDirs)
            .build();

    // With --sandbox_debug, dump a human-readable manifest next to the scratch dir for post-mortem
    // inspection when an action crashes inside the controller.
    if (getSandboxOptions().getSandboxDebug()) {
      Path manifestDump = scratchDir.getRelative("manifest.textproto");
      com.google.devtools.build.lib.vfs.FileSystemUtils.writeContent(
          manifestDump,
          SandboxBackendManifest.toDebugString(manifest)
              .getBytes(java.nio.charset.StandardCharsets.UTF_8));
    }

    // Wrap the action argv in process-wrapper so SubprocessBuilder gets an absolute argv[0] (it
    // rejects relative paths with directory components like `bazel-out/.../my_binary`). The wrapper
    // also enforces the timeout and produces resource-usage stats.
    Path statisticsPath = scratchDir.getRelative("stats.out");
    Duration timeout = context.getTimeout();
    // Debug: SANDBOX_BACKEND_PROFILE_WRAP=<mnemonic>:<wrapper> prepends <wrapper> to that mnemonic's argv
    // to deterministically profile the action process (sample/rusage) — no pid-catching races.
    ImmutableList<String> spawnArgs = spawn.getArguments();
    String profileWrap = clientEnv.get("SANDBOX_BACKEND_PROFILE_WRAP");
    if (profileWrap != null) {
      int sep = profileWrap.indexOf(':');
      if (sep > 0 && spawn.getMnemonic().equals(profileWrap.substring(0, sep))) {
        spawnArgs =
            ImmutableList.<String>builder()
                .add(profileWrap.substring(sep + 1))
                .addAll(spawnArgs)
                .build();
      }
    }
    ProcessWrapper.CommandLineBuilder wrapperBuilder =
        processWrapper
            .commandLineBuilder(spawnArgs)
            .addExecutionInfo(spawn.getExecutionInfo())
            .setStatisticsPath(statisticsPath.asFragment());
    if (!timeout.isZero()) {
      wrapperBuilder.setTimeout(timeout);
    }
    ImmutableList<String> wrappedArguments = wrapperBuilder.build();

    // Acquire this backend's server, keyed by name. getOrSpawn reuses the live server (common path)
    // or re-spawns one if the previous died, relaying the backend's configured options via the
    // Negotiate handshake.
    SandboxBackendServer daemon =
        SandboxBackendServer.getOrSpawn(
            name, sandboxdBinary, backendArgs, clientEnv, execRoot.getPathString());

    // Confinement inputs reused by the OS jail (Seatbelt/namespaces) when the backend requests one.
    // The action's writes land in the mount (added by the spawn) plus scratch and the OS's standard
    // temp dirs; the chosen mechanism is applied on top of the backend's filesystem view.
    var fs = execRoot.getFileSystem();
    ImmutableSet<Path> confinementWritableDirs =
        ImmutableSet.of(
            scratchDir,
            fs.getPath("/dev"),
            fs.getPath("/tmp"),
            fs.getPath("/private/tmp"),
            fs.getPath("/private/var/tmp"));
    boolean allowNetwork =
        Spawns.requiresNetwork(spawn, getSandboxOptions().getDefaultSandboxAllowNetwork());

    return new SandboxBackendSpawn(
        daemon,
        scratchDir,
        manifest,
        sandboxId,
        workspaceName,
        wrappedArguments,
        environment,
        treeDeleter,
        spawn.getMnemonic(),
        statisticsPath,
        confinementWritableDirs,
        getInaccessiblePaths(),
        allowNetwork,
        scratchDir.getRelative("sandbox.sb"),
        linuxSandboxPath);
  }

  @Override
  public String getName() {
    return name;
  }

  /**
   * Records a {@code locations} entry for the one kind of input whose bytes do not live at the
   * default {@code exec_root/<tree path>}: a runfiles entry (bytes at the target's real exec path)
   * or a virtual input (param files etc., materialized to scratch here). Every other input —
   * including tree artifacts and source dirs — is at the default and needs no entry. Tree paths are
   * anchored under the workspace dir, exactly as {@code buildForSpawn} anchors the tree it built.
   */
  private void collectLocation(
      PathFragment execPath,
      ActionInput input,
      PathFragment wsDir,
      InputMetadataProvider metadataProvider,
      Path scratchExecRoot,
      Map<String, String> locations)
      throws IOException {
    PathFragment treePath = wsDir.getRelative(execPath);

    // Virtual inputs (param files, EmptyActionInput runfiles markers) have no on-disk source and
    // EmptyActionInput.getExecPath() throws, so handle them before touching exec paths. Materialize
    // the bytes to scratch and point the controller at the scratch copy.
    if (input instanceof VirtualActionInput virtual) {
      Path scratchPath = scratchExecRoot.getRelative(execPath);
      scratchPath.getParentDirectory().createDirectoryAndParents();
      try (OutputStream os = scratchPath.getOutputStream()) {
        virtual.writeTo(os);
      }
      locations.put(
          StringEncoding.internalToPlatform(treePath.getPathString()),
          StringEncoding.internalToPlatform(scratchPath.getPathString()));
      return;
    }

    FileArtifactValue metadata;
    try {
      metadata = metadataProvider.getInputMetadata(input);
    } catch (IOException e) {
      throw new IOException("failed to fetch input metadata for " + input.getExecPathString(), e);
    }
    if (metadata == null || metadata.getType() == FileStateType.SYMLINK) {
      // Missing inputs are dropped (matching the standard sandbox); symlinks carry their target in
      // the tree and have no bytes, so neither needs a location.
      return;
    }

    // A synthetic tree path — a runfiles/fileset entry, or a displaced source dir — has its bytes at
    // the value's real exec path rather than at exec_root/<tree path>. Inputs at their natural path
    // (plain files, tree artifacts, source dirs) are at the default and need no redirect.
    PathFragment valueExec = input.getExecPath();
    if (!valueExec.equals(execPath)) {
      locations.put(
          StringEncoding.internalToPlatform(treePath.getPathString()),
          StringEncoding.internalToPlatform(wsDir.getRelative(valueExec).getPathString()));
    }
  }
}
