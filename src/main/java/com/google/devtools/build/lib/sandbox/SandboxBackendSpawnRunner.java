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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.LostInputsExecException;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileStateType;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.RunfilesTree;
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
import com.google.devtools.build.lib.sandbox.proto.SandboxProto.ConfinementSetting;
import com.google.devtools.build.lib.sandbox.proto.SandboxProto.Content;
import com.google.devtools.build.lib.sandbox.proto.SandboxProto.Manifest;
import com.google.devtools.build.lib.sandbox.proto.SandboxProto.Output;
import com.google.devtools.build.lib.sandbox.proto.SandboxProto.Retention;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.util.StringEncoding;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.ByteString;
import java.io.IOException;
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
  // The linux-sandbox helper, used only for the LINUX_NAMESPACES built-in jail.
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

    // content: the only inputs whose bytes do NOT live at the default exec_root/<tree path> are
    // runfiles entries (bytes at the target's exec path) and virtual inputs (param files etc.).
    // Everything else — including tree artifacts and source dirs — is at the default. Walk the same
    // input mapping the tree was built from; keys are exec paths and tree paths are anchored under
    // the workspace dir, exactly as buildForSpawn anchors them.
    PathFragment wsDir = PathFragment.create(workspaceName);
    // digest hash -> Content: virtual inputs ship inline (single-use), displaced files by location
    // (reuse). Both go through Push, keyed by digest and deduped in the controller. See
    // collectContent.
    LinkedHashMap<String, Content> content = new LinkedHashMap<>();
    Map<PathFragment, ActionInput> inputMap;
    try (com.google.devtools.build.lib.profiler.SilentCloseable c =
        com.google.devtools.build.lib.profiler.Profiler.instance()
            .profile("sandbox.inputMapping")) {
      inputMap =
          context.getInputMapping(PathFragment.EMPTY_FRAGMENT, /* willAccessRepeatedly= */ true);
    }
    for (Map.Entry<PathFragment, ActionInput> entry : inputMap.entrySet()) {
      collectContent(
          entry.getKey(),
          entry.getValue(),
          wsDir,
          metadataProvider,
          hashFunction,
          content);
    }

    // A runfiles leaf lives in the tree at its runfiles-layout path, never at its exec path, so the
    // default exec_root/<tree path> rule cannot resolve it — it always needs an explicit
    // digest->exec-path location. We drive this from the runfiles metadata rather than the flattened
    // input mapping above, because that mapping's key for a runfiles leaf can coincide with the
    // leaf's own exec path (e.g. a repo-mapped nested binary), making it indistinguishable from a
    // direct input and wrongly skipped by collectContent's default-path check. Keyed by digest, so
    // this is idempotent with any location the flattened walk already emitted.
    for (RunfilesTree runfilesTree : metadataProvider.getRunfilesTrees()) {
      for (Artifact member : runfilesTree.getMapping().values()) {
        collectRunfilesMember(member, wsDir, metadataProvider, content);
      }
    }

    // outputs: keyed by each declared output's in-sandbox path (where the action writes it), valued
    // by its kind ("file" or "dir") and the destination Collect moves it to. The controller
    // materializes outputs in the sandbox (a dir output gets the directory itself so `tar
    // --directory` can chdir in; a file output gets only its parent, pre-created from this map) and
    // Collect moves them out after the action.
    //
    // SandboxOutputs maps the UNMAPPED exec path (where Bazel expects the output) to the MAPPED
    // sandbox path (where the action, running with path-mapped argv, actually writes it). The key is
    // the mapped path so the controller finds the output; Output.dest is the unmapped path so Collect
    // lands it where Bazel looks. With path mapping off the two coincide and dest is left empty.
    LinkedHashMap<String, Output> outputsMap =
        new LinkedHashMap<>(outputs.files().size() + outputs.dirs().size());
    for (Map.Entry<PathFragment, PathFragment> output : outputs.files().entrySet()) {
      outputsMap.put(sandboxOutputPath(output.getValue()), outputValue("file", output));
    }
    for (Map.Entry<PathFragment, PathFragment> output : outputs.dirs().entrySet()) {
      outputsMap.put(sandboxOutputPath(output.getValue()), outputValue("dir", output));
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
    // staged it. Only content entries (runfiles, virtuals) deviate.
    String execRootPrefix =
        StringEncoding.internalToPlatform(execRoot.getParentDirectory().getPathString());

    // Confinement inputs: the host paths writable beyond the mount (scratch + the OS temp dirs),
    // the inaccessible paths, and network policy. Bazel's own built-in jail (the default) uses the
    // Path forms below; the manifest also carries the string forms so a backend returning a
    // CustomConfinement can build its own jail. The mount root is added by the jail itself (built-in)
    // or known to the backend (custom), so it is not listed here.
    var fs = execRoot.getFileSystem();
    ImmutableSet<Path> confinementWritableDirs =
        ImmutableSet.of(
            scratchDir,
            fs.getPath("/dev"),
            fs.getPath("/tmp"),
            fs.getPath("/private/tmp"),
            fs.getPath("/private/var/tmp"));
    ImmutableSet<Path> inaccessiblePaths = getInaccessiblePaths();
    boolean allowNetwork =
        Spawns.requiresNetwork(spawn, getSandboxOptions().getDefaultSandboxAllowNetwork());
    ConfinementSetting.Builder confinementSetting =
        ConfinementSetting.newBuilder().setAllowNetwork(allowNetwork);
    for (Path p : confinementWritableDirs) {
      confinementSetting.addWritablePaths(StringEncoding.internalToPlatform(p.getPathString()));
    }
    for (Path p : inaccessiblePaths) {
      confinementSetting.addInaccessiblePaths(StringEncoding.internalToPlatform(p.getPathString()));
    }

    Manifest manifest =
        Manifest.newBuilder()
            .setMnemonic(StringEncoding.internalToPlatform(spawn.getMnemonic()))
            .setHashFunction(hashFunction.toString())
            .setInputRootDigest(built.digest())
            .setExecRoot(execRootPrefix)
            .putAllOutputs(outputsMap)
            .putAllWritableDirs(writableDirs)
            .setConfinementSetting(confinementSetting)
            .build();

    // The tree's directory blobs travel out of band: the controller keeps a content-addressed store
    // deduped across actions, so the manifest references them only by input_root_digest and the
    // bytes ship via Push. Displaced-content overrides ride the same store. The spawn pushes the ones
    // the controller has not seen before creating.
    ImmutableMap<String, ByteString> blobs = SandboxBackendManifest.directoryBlobs(built);

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

    return new SandboxBackendSpawn(
        daemon,
        scratchDir,
        manifest,
        blobs,
        ImmutableMap.copyOf(content),
        sandboxId,
        workspaceName,
        wrappedArguments,
        environment,
        treeDeleter,
        spawn.getMnemonic(),
        statisticsPath,
        confinementWritableDirs,
        inaccessiblePaths,
        allowNetwork,
        scratchDir.getRelative("sandbox.sb"),
        linuxSandboxPath);
  }

  @Override
  public String getName() {
    return name;
  }

  /**
   * Records the {@link Content} for an input whose bytes do not live at the default {@code
   * exec_root/<tree path>}, keyed by its content digest hash (so the controller resolves a leaf by
   * the digest it reads while walking the tree):
   *
   * <ul>
   *   <li>a virtual input (param file etc.) has no on-disk source, so its bytes ship <em>inline</em>
   *       with {@link Retention#RETENTION_SINGLE_USE} — it is unique per action and not worth
   *       caching.
   *   <li>a runfiles/fileset entry or displaced tree-artifact child sits at a fixed exec path, so it
   *       ships as a <em>location</em> with {@link Retention#RETENTION_REUSE} — the controller
   *       captures the bytes on receipt and reuses them across actions by digest.
   * </ul>
   *
   * <p>Every input at its natural path is at the default and needs no entry. Tree paths are anchored
   * under the workspace dir, exactly as {@code buildForSpawn} anchors the tree it built.
   */
  private void collectContent(
      PathFragment execPath,
      ActionInput input,
      PathFragment wsDir,
      InputMetadataProvider metadataProvider,
      DigestHashFunction hashFunction,
      Map<String, Content> content)
      throws IOException {
    // Virtual inputs (param files, EmptyActionInput runfiles markers) have no on-disk source and
    // EmptyActionInput.getExecPath() throws, so handle them before touching exec paths. Ship the
    // bytes inline — there is no durable host path to point at. The key is the content digest
    // computed from the same bytes the tree hashed, so it matches the FileNode digest the controller
    // resolves against.
    if (input instanceof VirtualActionInput virtual) {
      ByteString.Output bytes = ByteString.newOutput();
      virtual.writeTo(bytes);
      String digestHash = DigestUtil.compute(virtual, hashFunction.getHashFunction()).getHash();
      content.put(
          digestHash,
          Content.newBuilder()
              .setInline(bytes.toByteString())
              .setRetention(Retention.RETENTION_SINGLE_USE)
              .build());
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
      // the tree and have no bytes, so neither needs a content entry.
      return;
    }

    PathFragment valueExec = input.getExecPath();
    if (valueExec.equals(execPath)) {
      // At the default exec_root/<tree path>; no redirect needed.
      return;
    }

    if (metadata.getType() == FileStateType.DIRECTORY) {
      // A displaced tree artifact: the input mapping pre-expands non-empty ones to file leaves
      // (handled above), so what reaches here is an empty tree artifact (no children) or a tree
      // whose children the metadata provider enumerates. Place each child by its own content digest
      // at its real host path. (A source directory has no tree metadata and cannot be placed this
      // way; that rare case is left to the default resolution.)
      TreeArtifactValue tree = metadataProvider.getTreeMetadata(input);
      if (tree != null) {
        for (Map.Entry<TreeFileArtifact, FileArtifactValue> child :
            tree.getChildValues().entrySet()) {
          String childHash = digestHashOf(child.getValue());
          if (childHash != null) {
            content.put(childHash, locationContent(wsDir.getRelative(child.getKey().getExecPath())));
          }
        }
      }
      return;
    }

    // A displaced file (runfiles/fileset entry): its bytes are at valueExec. Key it by the digest
    // carried in the input's own metadata — authoritative, and independent of whether the merkle
    // computer retained this input as a standalone blob (a runfiles-only source file may not).
    String digestHash = digestHashOf(metadata);
    if (digestHash != null) {
      content.put(digestHash, locationContent(wsDir.getRelative(valueExec)));
    }
  }

  /**
   * Emits the {@link Content} location(s) for one runfiles-tree member, keyed by content digest. A
   * runfiles member's bytes live at its artifact's exec path, never at its runfiles-layout tree path,
   * so it always needs an explicit location — unlike a direct input, it can never fall back to the
   * default {@code exec_root/<tree path>} rule. A {@code null} member is an empty-file placeholder
   * (no bytes); a symlink carries its target in the tree; a source directory has no per-child digest
   * and is left to default resolution.
   */
  private void collectRunfilesMember(
      @Nullable Artifact member,
      PathFragment wsDir,
      InputMetadataProvider metadataProvider,
      Map<String, Content> content)
      throws IOException {
    if (member == null) {
      return;
    }
    if (member.isTreeArtifact()) {
      TreeArtifactValue tree = metadataProvider.getTreeMetadata(member);
      if (tree != null) {
        for (Map.Entry<TreeFileArtifact, FileArtifactValue> child :
            tree.getChildValues().entrySet()) {
          String childHash = digestHashOf(child.getValue());
          if (childHash != null) {
            content.put(childHash, locationContent(wsDir.getRelative(child.getKey().getExecPath())));
          }
        }
      }
      return;
    }
    FileArtifactValue metadata = metadataProvider.getInputMetadata(member);
    if (metadata == null
        || metadata.getType() == FileStateType.SYMLINK
        || metadata.getType() == FileStateType.DIRECTORY) {
      return;
    }
    String digestHash = digestHashOf(metadata);
    if (digestHash != null) {
      content.put(digestHash, locationContent(wsDir.getRelative(member.getExecPath())));
    }
  }

  /**
   * The sandbox-root-relative path of an output at {@code execPath} (already path-mapped for the
   * key, unmapped for a dest): {@code /<workspaceName>/<execPath>}, in platform encoding. This is
   * both the manifest-map key and, for {@link Output#getDest}, the Collect destination.
   */
  private String sandboxOutputPath(PathFragment execPath) {
    return StringEncoding.internalToPlatform("/" + workspaceName + "/" + execPath.getPathString());
  }

  /**
   * The {@link Output} value for a declared output of the given {@code type}. {@code output} maps the
   * unmapped exec path (key) to the mapped sandbox path (value). {@link Output#getDest} is set to the
   * unmapped destination only when path mapping made it differ from the mapped in-sandbox path (the
   * manifest-map key); otherwise it is left empty, meaning "same as the key".
   */
  private Output outputValue(String type, Map.Entry<PathFragment, PathFragment> output) {
    Output.Builder builder = Output.newBuilder().setType(type);
    if (!output.getKey().equals(output.getValue())) {
      builder.setDest(sandboxOutputPath(output.getKey()));
    }
    return builder.build();
  }

  /**
   * A {@link Content} that reads its bytes from {@code hostPath} — reusable across actions. The
   * controller captures the bytes on receipt of the Push, so a later change to {@code hostPath} does
   * not matter (see {@code Content.location}).
   */
  private static Content locationContent(PathFragment hostPath) {
    return Content.newBuilder()
        .setLocation(StringEncoding.internalToPlatform(hostPath.getPathString()))
        .setRetention(Retention.RETENTION_REUSE)
        .build();
  }

  /**
   * The content digest hash of {@code metadata}, formatted exactly as the tree's {@code
   * FileNode.digest.hash} (so the daemon resolves against it), or {@code null} when the metadata
   * carries no digest.
   */
  @Nullable
  private static String digestHashOf(FileArtifactValue metadata) {
    byte[] digest = metadata.getDigest();
    return digest == null ? null : DigestUtil.buildDigest(digest, metadata.getSize()).getHash();
  }
}
