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

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;
import static org.junit.Assume.assumeTrue;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.LocalHostCapacity;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.actions.ParamFileActionInput;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.sandbox.SpawnRunnerTestUtil.SpawnExecutionContextForTesting;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.time.Duration;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link SandboxBackendSpawnRunner}.
 *
 * <p>Uses a stub controller binary that returns the host scratch tree as the sandbox root, so the
 * "sandbox view" is the scratch tree directly — enough to verify Bazel's side of the contract
 * without a real FSKit mount. No macOS required: the runner is constructed directly, bypassing
 * {@code SandboxModule}'s platform check.
 */
@RunWith(JUnit4.class)
public final class SandboxBackendSpawnRunnerTest extends SandboxedSpawnRunnerTestCase {

  private static final TreeDeleter treeDeleter = new SynchronousTreeDeleter();

  /**
   * A minimal {@link PathMapper} that strips a {@code cfg/} configuration segment right after {@code
   * bazel-out/} ({@code bazel-out/cfg/bin/x} -> {@code bazel-out/bin/x}), mimicking {@code
   * StrippingPathMapper} without the full analysis machinery.
   */
  private static final PathMapper STRIP_CFG =
      new PathMapper() {
        @Override
        public PathFragment map(PathFragment execPath) {
          String s = execPath.getPathString();
          String prefix = "bazel-out/cfg/";
          return s.startsWith(prefix)
              ? PathFragment.create("bazel-out/" + s.substring(prefix.length()))
              : execPath;
        }
      };

  private CommandEnvironment commandEnvironment;
  private Path sandboxBase;
  private PathFragment stubBinary;

  @Before
  public void setUp() throws Exception {
    commandEnvironment = runtimeWrapper.newCommand();
    commandEnvironment
        .getLocalResourceManager()
        .setAvailableResources(LocalHostCapacity.getLocalHostCapacity());

    Path execRoot = commandEnvironment.getExecRoot();
    execRoot.createDirectory();
    // The runner wraps action argv in process-wrapper; ProcessWrapper.fromCommandEnvironment needs
    // the binary on disk.
    SpawnRunnerTestUtil.copyProcessWrapperIntoPath(execRoot);

    sandboxBase = execRoot.getRelative("sandbox");
    sandboxBase.createDirectory();

    // Stub implements `<binary> serve` over the real wire protocol: varint length-delimited
    // Request/Response protos on stdin/stdout (see proto/sandbox.proto). It acts as a minimal
    // controller: it captures Push blobs/content into a store (reading a content.location's bytes on
    // receipt, as the contract requires), and on Create walks the input tree from input_root_digest,
    // materializing files into scratch — which it returns as the sandbox root, so the action (cwd at
    // the sandbox exec root) sees its inputs. Python because POSIX sh can't parse varints.
    Path stub = execRoot.getRelative("sandbox-fs-stub");
    FileSystemUtils.writeContentAsLatin1(
        stub,
        "#!/usr/bin/env python3\n"
            + "import sys, os\n"
            + "inp, out = sys.stdin.buffer, sys.stdout.buffer\n"
            + "def rd_varint(b, i):\n"
            + "    v = s = 0\n"
            + "    while True:\n"
            + "        x = b[i]; i += 1; v |= (x & 0x7f) << s\n"
            + "        if not x & 0x80: return v, i\n"
            + "        s += 7\n"
            + "def fields(b):\n"
            + "    i = 0\n"
            + "    while i < len(b):\n"
            + "        k, i = rd_varint(b, i)\n"
            + "        if k & 7 == 0: v, i = rd_varint(b, i)\n"
            + "        else:\n"
            + "            n, i = rd_varint(b, i); v = b[i:i+n]; i += n\n"
            + "        yield k >> 3, v\n"
            + "def varint(v):\n"
            + "    o = b''\n"
            + "    while True:\n"
            + "        x = v & 0x7f; v >>= 7\n"
            + "        o += bytes([x | 0x80] if v else [x])\n"
            + "        if not v: return o\n"
            + "def ld(f, p): return varint(f << 3 | 2) + varint(len(p)) + p\n"
            + "def uv(f, v): return varint(f << 3) + varint(v)\n"
            + "def reply(payload):\n"
            + "    out.write(varint(len(payload)) + payload); out.flush()\n"
            // Content store populated by Push: digest hash -> Directory bytes (blobstore) and digest
            // hash -> leaf bytes (contentstore).
            + "blobstore = {}\n"
            + "contentstore = {}\n"
            // sandbox_id -> (scratch root bytes, [(in-sandbox key, dest, type)]) captured on Create,
            // used by Collect to move each declared output to its destination.
            + "sandboxes = {}\n"
            + "def dhash(d):\n"  // Digest.hash (field 1)
            + "    for f, v in fields(d):\n"
            + "        if f == 1: return v\n"
            + "    return b''\n"
            // Walk a Directory (by digest) writing its files under `base`; `rel` is the tree path so
            // far, used to fall back to the default host location execroot/<tree path>.
            + "def materialize(dirhash, base, rel, execroot):\n"
            + "    d = blobstore.get(dirhash)\n"
            + "    if d is None: return\n"
            + "    for f, v in fields(d):\n"
            + "        if f == 1:\n"  // FileNode { name=1, digest=2 }
            + "            fn = dict(fields(v)); name = fn.get(1, b'').decode()\n"
            + "            treepath = (rel + '/' + name) if rel else name\n"
            + "            data = contentstore.get(dhash(fn.get(2, b'')))\n"
            + "            if data is None and execroot:\n"
            + "                try: data = open(execroot + '/' + treepath, 'rb').read()\n"
            + "                except OSError: data = b''\n"
            + "            open(os.path.join(base, name), 'wb').write(data or b'')\n"
            + "        elif f == 2:\n"  // DirectoryNode { name=1, digest=2 }
            + "            dn = dict(fields(v)); name = dn.get(1, b'').decode()\n"
            + "            sub = os.path.join(base, name); os.makedirs(sub, exist_ok=True)\n"
            + "            materialize(dhash(dn.get(2, b'')), sub, (rel + '/' + name) if rel else name, execroot)\n"
            + "if sys.argv[1:2] != ['serve']: sys.exit(1)\n"
            + "ws = None\n"  // capture --workspace <ws> to record negotiate options under it
            + "for i, a in enumerate(sys.argv):\n"
            + "    if a == '--workspace' and i + 1 < len(sys.argv): ws = sys.argv[i + 1]\n"
            // Confinement mode returned in Create.Result.Ok, chosen by a relayed negotiate option:
            //   default            -> 'unconfined' arm (tests run bare, no OS jail to nest)
            //   'builtin'          -> leave the oneof unset -> Bazel applies its platform default
            //   'custom-wrapper:P' -> 'custom' arm with wrapper=[P]
            + "confmode = 'unconfined'; custom_wrapper = None; custom_env = None\n"
            + "def rd_len(f):\n"
            + "    v = s = 0\n"
            + "    while True:\n"
            + "        c = f.read(1)\n"
            + "        if not c: return None\n"
            + "        x = c[0]; v |= (x & 0x7f) << s\n"
            + "        if not x & 0x80: return v\n"
            + "        s += 7\n"
            + "while True:\n"
            + "    n = rd_len(inp)\n"
            + "    if n is None: break\n"
            + "    req = inp.read(n)\n"
            + "    rid = 0; sid = b''; create = None; destroy = None; collect = None; negotiate = None; push = None\n"
            + "    for f, v in fields(req):\n"
            + "        if f == 1: rid = v\n"
            + "        elif f == 2: sid = v\n"  // Request.sandbox_id
            + "        elif f == 3: create = v\n"
            + "        elif f == 4: destroy = v\n"
            + "        elif f == 5: collect = v\n"
            + "        elif f == 6: negotiate = v\n"
            + "        elif f == 7: push = v\n"  // Request.push
            + "    if push is not None:\n"  // fire-and-forget store write; no reply
            + "        for pf, pv in fields(push):\n"
            + "            e = dict(fields(pv))\n"  // map entry { key=1, value=2 }
            + "            if pf == 1: blobstore[e.get(1, b'')] = e.get(2, b'')\n"  // Push.blobs
            + "            elif pf == 2:\n"  // Push.content { key=1 -> Content=2 }
            + "                c = dict(fields(e.get(2, b'')))\n"  // Content { inline=1 | location=2 }
            + "                if 1 in c: contentstore[e.get(1, b'')] = c[1]\n"
            + "                elif 2 in c: contentstore[e.get(1, b'')] = open(c[2].decode(), 'rb').read()\n"
            + "        continue\n"
            + "    if negotiate is not None:\n"
            // Record Negotiate.options (field 1, repeated string) to <ws>/negotiate.opts so the
            // test can assert backend args were relayed over the protocol, not argv.
            + "        opts = [x for fo, x in fields(negotiate) if fo == 1]\n"
            + "        if b'builtin' in opts: confmode = 'builtin'\n"
            + "        if b'custom-empty' in opts: confmode = 'custom-empty'\n"
            + "        for o in opts:\n"
            + "            if o.startswith(b'custom-wrapper:'):\n"
            + "                confmode = 'custom'; custom_wrapper = o[len(b'custom-wrapper:'):]\n"
            + "            elif o.startswith(b'custom-env:'):\n"
            + "                kv = o[len(b'custom-env:'):].split(b'=', 1); custom_env = (kv[0], kv[1])\n"
            + "        if ws is not None:\n"
            + "            open(ws + '/negotiate.opts', 'wb').write(b'\\n'.join(opts))\n"
            + "        reply(uv(1, rid) + ld(5, ld(1, b''))); continue\n"  // negotiate=5 { Result.ok=1 {} }
            + "    if destroy is not None:\n"
            + "        reply(uv(1, rid) + ld(3, ld(1, b''))); continue\n"  // destroy=3 { Result.ok=1 {} }
            + "    if collect is not None:\n"
            // Collect { exec_root = 1 }: the dir the workspace exec root lives in. Move each declared
            // output from its in-sandbox path (scratch + key) to its destination under exec_root
            // (Output.dest when set — the unmapped path under path mapping — else the key itself).
            + "        cexec = b''\n"
            + "        for cf, cv in fields(collect):\n"
            + "            if cf == 1: cexec = cv\n"
            + "        st = sandboxes.get(sid)\n"
            + "        if st is not None:\n"
            + "            scr, outs = st\n"
            + "            for okey, odest, otype in outs:\n"
            + "                src = scr.decode() + okey\n"
            + "                dst = cexec.decode() + (odest if odest else okey)\n"
            + "                os.makedirs(os.path.dirname(dst), exist_ok=True)\n"
            + "                try: os.replace(src, dst)\n"
            + "                except OSError: pass\n"
            + "        reply(uv(1, rid) + ld(4, ld(1, b''))); continue\n"  // collect=4 { Result.ok=1 {} }
            + "    if create is None: continue\n"  // unknown op; nothing to reply
            + "    manifest = None\n"
            + "    for f, v in fields(create):\n"  // Create { manifest = 1 }
            + "        if f == 1: manifest = v\n"
            + "    tmphost = None; execroot = None; rootdig = None\n"
            + "    cinputs = None; outs = []\n"
            + "    if manifest is not None:\n"
            + "        for f, v in fields(manifest):\n"
            + "            if f == 2: execroot = v\n"  // Manifest.exec_root
            + "            elif f == 4: rootdig = v\n"  // Manifest.input_root_digest
            + "            elif f == 5:\n"  // Manifest.outputs map entry { key=1 string -> Output=2 }
            + "                oe = dict(fields(v))\n"
            + "                ov = dict(fields(oe.get(2, b'')))\n"  // Output { type=1, dest=2 }
            + "                outs.append((oe.get(1, b'').decode(), ov.get(2, b'').decode(), ov.get(1, b'').decode()))\n"
            + "            elif f == 6:\n"  // Manifest.writable_dirs
            + "                kv = dict(fields(v))\n"
            + "                if kv.get(1) == b'/tmp': tmphost = kv.get(2)\n"
            + "            elif f == 7: cinputs = v\n"  // Manifest.confinement_setting
            // Record ConfinementSetting{ writable_paths=1, inaccessible_paths=2, allow_network=3 } so
            // a test can assert Bazel delivered it to the backend.
            + "    if ws is not None and cinputs is not None:\n"
            + "        wp = [x for cf, x in fields(cinputs) if cf == 1]\n"
            + "        ip = [x for cf, x in fields(cinputs) if cf == 2]\n"
            + "        an = next((x for cf, x in fields(cinputs) if cf == 3), 0)\n"
            + "        open(ws + '/confinement.inputs', 'wb').write("
            + "b'W:' + b','.join(wp) + b'\\nI:' + b','.join(ip) + b'\\nN:' + str(an).encode())\n"
            + "    if not tmphost:\n"
            + "        reply(uv(1, rid) + ld(2, ld(3, ld(1, b'no /tmp in manifest')))); continue\n"  // create=2 { error=3 { msg=1 } }
            + "    scratch = tmphost[: -len(b'/tmp')]\n"
            // The controller owns the sandbox mount, so it (not the runner) creates the exec-root
            // dir the action's cwd derives from: <root>/<workspaceName>. basename(ws) is the
            // workspace name (ws is the real exec root path passed via --workspace).
            + "    if ws is not None:\n"
            + "        os.makedirs(os.path.join(scratch.decode(), os.path.basename(ws)), exist_ok=True)\n"
            // Materialize the input tree into scratch so the action can read its inputs.
            + "    if rootdig is not None:\n"
            + "        materialize(dhash(rootdig), scratch.decode(), '', execroot.decode() if execroot else '')\n"
            // Pre-create each output's parent (a dir output: the directory itself), mirroring what a
            // real controller does so the action can write into it. Remember the outputs for Collect.
            + "    for okey, odest, otype in outs:\n"
            + "        p = scratch.decode() + okey\n"
            + "        os.makedirs(p if otype == 'dir' else os.path.dirname(p), exist_ok=True)\n"
            + "    sandboxes[sid] = (scratch, outs)\n"
            // Ok{ path=1, oneof confinement { custom=2 (CustomConfinement{wrapper=2}) | unconfined=3 } }.
            // 'builtin' leaves the oneof unset so Bazel applies its platform default.
            + "    ok = ld(1, scratch)\n"
            + "    if confmode == 'unconfined': ok += ld(3, b'')\n"
            + "    elif confmode == 'custom-empty': ok += ld(2, b'')\n"  // CustomConfinement with no wrapper
            + "    elif confmode == 'custom':\n"
            + "        cp = ld(2, custom_wrapper)\n"  // CustomConfinement.wrapper=2
            + "        if custom_env is not None:\n"  // CustomConfinement.env=3 (map entry {1:k, 2:v})
            + "            cp += ld(3, ld(1, custom_env[0]) + ld(2, custom_env[1]))\n"
            + "        ok += ld(2, cp)\n"
            + "    reply(uv(1, rid) + ld(2, ld(1, ok)))\n");  // Response{create=2: Create.Result{ok=1: Ok}}
    stub.setExecutable(true);
    stubBinary = stub.asFragment();
  }

  /** A backend named "sandbox-backend" with no options, backed by {@code binary}. */
  private SandboxBackendSpawnRunner newRunner(PathFragment binary) {
    return newRunner("sandbox-backend", binary, ImmutableList.of());
  }

  private SandboxBackendSpawnRunner newRunner(
      String name, PathFragment binary, ImmutableList<String> args) {
    return new SandboxBackendSpawnRunner(
        commandEnvironment, name, binary, args, sandboxBase, treeDeleter);
  }

  @Test
  public void absolutePathBinary_executesAction() throws Exception {
    // Confirms the runner accepts an absolute path to the controller binary.
    assertThat(stubBinary.isAbsolute()).isTrue();
    SandboxBackendSpawnRunner runner =
        newRunner(stubBinary);

    Spawn spawn = new SpawnBuilder("/bin/sh", "-c", "exit 42").build();
    FileOutErr fileOutErr =
        new FileOutErr(testRoot.getChild("stdout"), testRoot.getChild("stderr"));
    SpawnExecutionContextForTesting policy =
        new SpawnExecutionContextForTesting(spawn, fileOutErr, Duration.ofMinutes(1));

    SpawnResult result = runner.exec(spawn, policy);

    assertThat(result.status()).isEqualTo(SpawnResult.Status.NON_ZERO_EXIT);
    assertThat(result.exitCode()).isEqualTo(42);
  }

  @Test
  public void virtualInput_isShippedInlineAndReadableByAction() throws Exception {
    // A virtual input (param file) has no on-disk source: buildForSpawn digests its bytes directly
    // and the runner ships them inline in Push.content (single-use). The stub (acting as controller)
    // captures the inline bytes and materializes the file into scratch, which it returns as the
    // sandbox root, so the action — cwd at the sandbox exec root — reads it. Exercises the
    // virtual-input inline branch of collectContent.
    SandboxBackendSpawnRunner runner =
        newRunner(stubBinary);

    ParamFileActionInput paramFile =
        new ParamFileActionInput(
            PathFragment.create("params.txt"),
            ImmutableList.of("hello-from-param-file"),
            ParameterFileType.UNQUOTED);
    Spawn spawn =
        new SpawnBuilder("/bin/sh", "-c", "cat params.txt").withInput(paramFile).build();
    FileOutErr fileOutErr =
        new FileOutErr(testRoot.getChild("stdout"), testRoot.getChild("stderr"));
    SpawnExecutionContextForTesting policy =
        new SpawnExecutionContextForTesting(spawn, fileOutErr, Duration.ofMinutes(1));

    SpawnResult result = runner.exec(spawn, policy);

    assertThat(result.status()).isEqualTo(SpawnResult.Status.SUCCESS);
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(fileOutErr.outAsLatin1()).contains("hello-from-param-file");
  }

  @Test
  public void virtualInputInSubdirectory_isReadableByAction() throws Exception {
    // The virtual input lives under a nested tree path; walking the tree creates the parent dirs
    // before writing the leaf. Reading it back confirms the whole prefix was materialized.
    SandboxBackendSpawnRunner runner =
        newRunner(stubBinary);

    ParamFileActionInput paramFile =
        new ParamFileActionInput(
            PathFragment.create("nested/dir/params.txt"),
            ImmutableList.of("nested-content"),
            ParameterFileType.UNQUOTED);
    Spawn spawn =
        new SpawnBuilder("/bin/sh", "-c", "cat nested/dir/params.txt").withInput(paramFile).build();
    FileOutErr fileOutErr =
        new FileOutErr(testRoot.getChild("stdout"), testRoot.getChild("stderr"));
    SpawnExecutionContextForTesting policy =
        new SpawnExecutionContextForTesting(spawn, fileOutErr, Duration.ofMinutes(1));

    SpawnResult result = runner.exec(spawn, policy);

    assertThat(result.status()).isEqualTo(SpawnResult.Status.SUCCESS);
    assertThat(fileOutErr.outAsLatin1()).contains("nested-content");
  }

  @Test
  public void declaredOutput_isCollectedToExecRoot() throws Exception {
    // Baseline (no path mapping): the action writes its declared output inside the sandbox and
    // Collect moves it to the same exec path under the exec root. Manifest.outputs carries no dest,
    // so the in-sandbox path and the destination coincide.
    SandboxBackendSpawnRunner runner = newRunner(stubBinary);

    Spawn spawn =
        new SpawnBuilder("/bin/sh", "-c", "echo out-content > bazel-out/bin/out.txt")
            .withOutput("bazel-out/bin/out.txt")
            .build();
    FileOutErr fileOutErr =
        new FileOutErr(testRoot.getChild("stdout"), testRoot.getChild("stderr"));
    SpawnExecutionContextForTesting policy =
        new SpawnExecutionContextForTesting(spawn, fileOutErr, Duration.ofMinutes(1));

    SpawnResult result = runner.exec(spawn, policy);

    assertThat(result.status()).isEqualTo(SpawnResult.Status.SUCCESS);
    Path collected = commandEnvironment.getExecRoot().getRelative("bazel-out/bin/out.txt");
    assertThat(collected.exists()).isTrue();
    assertThat(FileSystemUtils.readContent(collected, UTF_8)).contains("out-content");
  }

  @Test
  public void pathMappedOutput_isCollectedToUnmappedExecPath() throws Exception {
    // With path mapping, the action's argv is rewritten to the MAPPED path (config segment
    // stripped), so it writes bazel-out/bin/out.txt inside the sandbox — but Bazel expects the
    // output at the UNMAPPED exec path bazel-out/cfg/bin/out.txt. The runner must key Manifest.outputs
    // by the mapped path (so the controller finds it) and set Output.dest to the unmapped path (so
    // Collect lands it where Bazel looks). Verify it arrives at the unmapped path, not the mapped one.
    SandboxBackendSpawnRunner runner = newRunner(stubBinary);

    Spawn spawn =
        new SpawnBuilder("/bin/sh", "-c", "echo mapped-content > bazel-out/bin/out.txt")
            .withOutput("bazel-out/cfg/bin/out.txt")
            .setPathMapper(STRIP_CFG)
            .build();
    FileOutErr fileOutErr =
        new FileOutErr(testRoot.getChild("stdout"), testRoot.getChild("stderr"));
    SpawnExecutionContextForTesting policy =
        new SpawnExecutionContextForTesting(spawn, fileOutErr, Duration.ofMinutes(1));

    SpawnResult result = runner.exec(spawn, policy);

    assertThat(result.status()).isEqualTo(SpawnResult.Status.SUCCESS);
    Path unmapped = commandEnvironment.getExecRoot().getRelative("bazel-out/cfg/bin/out.txt");
    assertThat(unmapped.exists()).isTrue();
    assertThat(FileSystemUtils.readContent(unmapped, UTF_8)).contains("mapped-content");
    // The mapped path is where the action wrote it inside the sandbox; nothing should be collected
    // there under the exec root.
    assertThat(commandEnvironment.getExecRoot().getRelative("bazel-out/bin/out.txt").exists())
        .isFalse();
  }

  @Test
  public void backendArgs_areRelayedToBackendViaNegotiate() throws Exception {
    // --sandbox_backend_opt options travel over the Negotiate handshake, not as process argv. The
    // stub records the Negotiate.options it received; assert our args arrived in order.
    SandboxBackendSpawnRunner runner =
        newRunner("acme", stubBinary, ImmutableList.of("--backend=fskit", "--cache-dir=/x"));
    Spawn spawn = new SpawnBuilder("/bin/sh", "-c", "exit 0").build();
    FileOutErr fileOutErr =
        new FileOutErr(testRoot.getChild("stdout"), testRoot.getChild("stderr"));
    SpawnExecutionContextForTesting policy =
        new SpawnExecutionContextForTesting(spawn, fileOutErr, Duration.ofMinutes(1));

    SpawnResult result = runner.exec(spawn, policy);

    assertThat(result.status()).isEqualTo(SpawnResult.Status.SUCCESS);
    Path recorded = commandEnvironment.getExecRoot().getRelative("negotiate.opts");
    assertThat(new String(FileSystemUtils.readContentAsLatin1(recorded)))
        .isEqualTo("--backend=fskit\n--cache-dir=/x");
  }

  @Test
  public void configuredName_isReportedAsStrategyName() {
    // The registered backend name (what --strategy selects) is the strategy's name.
    assertThat(newRunner("acme", stubBinary, ImmutableList.of()).getName()).isEqualTo("acme");
  }

  @Test
  public void defaultConfinement_onDarwin_wrapsActionInSeatbelt() throws Exception {
    // When the backend leaves confinement to Bazel (the 'builtin' stub mode returns an unset
    // confinement oneof), Bazel applies its platform default — Seatbelt on macOS — wrapping the argv
    // as `sandbox-exec -f <profile> -- <argv>`. Real sandbox-exec can't nest (this test already runs
    // under darwin-sandbox), so point it at a passthrough fake that drops `-f <profile>` and execs
    // the inner argv. The action runs end to end, proving the built-in path produced a valid wrapper.
    assumeTrue(OS.getCurrent() == OS.DARWIN);
    Path fakeSandboxExec = commandEnvironment.getExecRoot().getRelative("fake-sandbox-exec");
    FileSystemUtils.writeContentAsLatin1(fakeSandboxExec, "#!/bin/sh\nshift 2\nexec \"$@\"\n");
    fakeSandboxExec.setExecutable(true);
    String oldSandboxExec = SandboxBackendSpawn.sandboxExecBinary;
    SandboxBackendSpawn.sandboxExecBinary = fakeSandboxExec.getPathString();
    try {
      SandboxBackendSpawnRunner runner =
          newRunner("acme-builtin", stubBinary, ImmutableList.of("builtin"));
      Spawn spawn = new SpawnBuilder("/bin/sh", "-c", "exit 0").build();
      FileOutErr fileOutErr =
          new FileOutErr(testRoot.getChild("stdout"), testRoot.getChild("stderr"));
      SpawnExecutionContextForTesting policy =
          new SpawnExecutionContextForTesting(spawn, fileOutErr, Duration.ofMinutes(1));

      SpawnResult result = runner.exec(spawn, policy);

      assertThat(result.status()).isEqualTo(SpawnResult.Status.SUCCESS);
    } finally {
      SandboxBackendSpawn.sandboxExecBinary = oldSandboxExec;
    }
  }

  @Test
  public void customConfinement_wrapsActionInBackendSuppliedWrapper() throws Exception {
    // The backend brings its own jail: it returns a CustomConfinement whose wrapper Bazel prepends
    // to the action, outermost. Here the wrapper is a script that touches a marker then execs its
    // args; the marker's existence after the action proves Bazel applied the backend's wrapper.
    Path marker = testRoot.getChild("custom-wrapper-ran");
    Path wrapper = commandEnvironment.getExecRoot().getRelative("custom-wrapper");
    FileSystemUtils.writeContentAsLatin1(
        wrapper, "#!/bin/sh\ntouch " + marker.getPathString() + "\nexec \"$@\"\n");
    wrapper.setExecutable(true);

    SandboxBackendSpawnRunner runner =
        newRunner(
            "acme-custom",
            stubBinary,
            ImmutableList.of("custom-wrapper:" + wrapper.getPathString()));
    Spawn spawn = new SpawnBuilder("/bin/sh", "-c", "exit 7").build();
    FileOutErr fileOutErr =
        new FileOutErr(testRoot.getChild("stdout"), testRoot.getChild("stderr"));
    SpawnExecutionContextForTesting policy =
        new SpawnExecutionContextForTesting(spawn, fileOutErr, Duration.ofMinutes(1));

    SpawnResult result = runner.exec(spawn, policy);

    assertThat(result.exitCode()).isEqualTo(7);
    assertThat(marker.exists()).isTrue();
  }

  @Test
  public void customConfinement_mergesEnvIntoAction() throws Exception {
    // A CustomConfinement's env is merged over the action's env. The wrapper writes the injected var
    // to a marker; its value proves Bazel applied the merge before launching.
    Path marker = testRoot.getChild("custom-env-value");
    Path wrapper = commandEnvironment.getExecRoot().getRelative("custom-env-wrapper");
    FileSystemUtils.writeContentAsLatin1(
        wrapper,
        "#!/bin/sh\nprintf '%s' \"$SANDBOX_CUSTOM\" > " + marker.getPathString() + "\nexec \"$@\"\n");
    wrapper.setExecutable(true);

    SandboxBackendSpawnRunner runner =
        newRunner(
            "acme-custom-env",
            stubBinary,
            ImmutableList.of(
                "custom-wrapper:" + wrapper.getPathString(), "custom-env:SANDBOX_CUSTOM=hello-env"));
    Spawn spawn = new SpawnBuilder("/bin/sh", "-c", "exit 0").build();
    FileOutErr fileOutErr =
        new FileOutErr(testRoot.getChild("stdout"), testRoot.getChild("stderr"));
    SpawnExecutionContextForTesting policy =
        new SpawnExecutionContextForTesting(spawn, fileOutErr, Duration.ofMinutes(1));

    SpawnResult result = runner.exec(spawn, policy);

    assertThat(result.status()).isEqualTo(SpawnResult.Status.SUCCESS);
    assertThat(new String(FileSystemUtils.readContentAsLatin1(marker))).isEqualTo("hello-env");
  }

  @Test
  public void emptyCustomWrapper_failsClosed() throws Exception {
    // A custom confinement with an empty wrapper is no jail; that must be requested via the
    // Unconfined arm, so Bazel rejects it rather than run silently unconfined.
    SandboxBackendSpawnRunner runner =
        newRunner("acme-empty", stubBinary, ImmutableList.of("custom-empty"));
    Spawn spawn = new SpawnBuilder("/bin/sh", "-c", "exit 0").build();
    FileOutErr fileOutErr =
        new FileOutErr(testRoot.getChild("stdout"), testRoot.getChild("stderr"));
    SpawnExecutionContextForTesting policy =
        new SpawnExecutionContextForTesting(spawn, fileOutErr, Duration.ofMinutes(1));

    Exception e = assertThrows(Exception.class, () -> runner.exec(spawn, policy));
    assertThat(e).hasMessageThat().contains("empty wrapper");
  }

  @Test
  public void unconfined_appliesNoJail_onDarwin() throws Exception {
    // The backend returns the Unconfined arm (the stub's default mode). On macOS this must NOT wrap
    // the action in Seatbelt: run without the fake sandbox-exec, so if a real jail were applied it
    // would fail to nest under the outer darwin-sandbox. Success proves no jail was added.
    assumeTrue(OS.getCurrent() == OS.DARWIN);
    SandboxBackendSpawnRunner runner =
        newRunner("acme-unconfined", stubBinary, ImmutableList.of());
    Spawn spawn = new SpawnBuilder("/bin/sh", "-c", "exit 0").build();
    FileOutErr fileOutErr =
        new FileOutErr(testRoot.getChild("stdout"), testRoot.getChild("stderr"));
    SpawnExecutionContextForTesting policy =
        new SpawnExecutionContextForTesting(spawn, fileOutErr, Duration.ofMinutes(1));

    SpawnResult result = runner.exec(spawn, policy);

    assertThat(result.status()).isEqualTo(SpawnResult.Status.SUCCESS);
  }

  @Test
  public void confinementInputs_areSentToBackend() throws Exception {
    // Bazel populates Manifest.confinement_setting every Create so a custom backend can build its own
    // jail. The stub records them; assert a stable writable host path and the network flag arrived.
    SandboxBackendSpawnRunner runner =
        newRunner("acme-cinputs", stubBinary, ImmutableList.of());
    Spawn spawn = new SpawnBuilder("/bin/sh", "-c", "exit 0").build();
    FileOutErr fileOutErr =
        new FileOutErr(testRoot.getChild("stdout"), testRoot.getChild("stderr"));
    SpawnExecutionContextForTesting policy =
        new SpawnExecutionContextForTesting(spawn, fileOutErr, Duration.ofMinutes(1));

    var unused = runner.exec(spawn, policy);

    Path recorded = commandEnvironment.getExecRoot().getRelative("confinement.inputs");
    String text = new String(FileSystemUtils.readContentAsLatin1(recorded));
    // The writable-path list reached the backend intact (two stable entries), and the network flag
    // line is present (its value follows the host's --sandbox_default_allow_network policy).
    assertThat(text).contains("/dev");
    assertThat(text).contains("/private/tmp");
    assertThat(text).contains("N:");
  }

  @Test
  public void getName_isStable() {
    SandboxBackendSpawnRunner runner =
        newRunner(stubBinary);
    assertThat(runner.getName()).isEqualTo("sandbox-backend");
  }

  @Test
  public void canExec_trueWhenBinaryProbeSucceeds() {
    SandboxBackendSpawnRunner runner =
        newRunner(stubBinary);
    Spawn spawn = new SpawnBuilder("/bin/true").build();
    assertThat(runner.canExec(spawn)).isTrue();
  }

  @Test
  public void canExec_falseWhenBinaryPathEmpty() {
    SandboxBackendSpawnRunner runner =
        newRunner(PathFragment.EMPTY_FRAGMENT);
    Spawn spawn = new SpawnBuilder("/bin/true").build();
    // No binary configured: canExec must decline so Bazel falls through to the next strategy.
    assertThat(runner.canExec(spawn)).isFalse();
  }

  @Test
  public void canExec_falseWhenAbsolutePathDoesNotExist() {
    SandboxBackendSpawnRunner runner =
        newRunner(PathFragment.create("/definitely/not/a/real/binary"));
    Spawn spawn = new SpawnBuilder("/bin/true").build();
    assertThat(runner.canExec(spawn)).isFalse();
  }

  @Test
  public void canExec_falseWhenBareNameNotInPath() {
    // Bare names are resolved against $PATH at probe time; an unfindable name fails the probe and
    // the strategy declines.
    SandboxBackendSpawnRunner runner =
        newRunner(PathFragment.create("definitely-not-a-real-binary-xyz-9999"));
    Spawn spawn = new SpawnBuilder("/bin/true").build();
    assertThat(runner.canExec(spawn)).isFalse();
  }
}
