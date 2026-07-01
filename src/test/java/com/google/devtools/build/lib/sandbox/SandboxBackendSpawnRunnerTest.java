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
import static org.junit.Assert.assertThrows;
import static org.junit.Assume.assumeTrue;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.LocalHostCapacity;
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

    // Stub implements `<binary> serve` over the real wire protocol: length-prefixed (u32 LE)
    // Request/Response protos on stdin/stdout (see proto/sandbox.proto). It derives the scratch
    // root from /tmp's host target in writable_dirs (its parent) so the returned root matches what
    // the runner pre-created. Python because POSIX sh can't parse varints.
    Path stub = execRoot.getRelative("sandbox-fs-stub");
    FileSystemUtils.writeContentAsLatin1(
        stub,
        "#!/usr/bin/env python3\n"
            + "import sys\n"
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
            + "if sys.argv[1:2] != ['serve']: sys.exit(1)\n"
            + "ws = None\n"  // capture --workspace <ws> to record negotiate options under it
            + "for i, a in enumerate(sys.argv):\n"
            + "    if a == '--workspace' and i + 1 < len(sys.argv): ws = sys.argv[i + 1]\n"
            // Confinement returned in Created: NONE (1) by default, SEATBELT (2) if a backend option
            // 'seatbelt' was relayed at negotiate.
            + "confine = 1\n"
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
            + "    rid = 0; create = None; destroy = None; collect = None; negotiate = None\n"
            + "    for f, v in fields(req):\n"
            + "        if f == 1: rid = v\n"
            + "        elif f == 3: create = v\n"
            + "        elif f == 4: destroy = v\n"
            + "        elif f == 5: collect = v\n"
            + "        elif f == 6: negotiate = v\n"
            + "    if negotiate is not None:\n"
            // Record Negotiate.options (field 1, repeated string) to <ws>/negotiate.opts so the
            // test can assert backend args were relayed over the protocol, not argv.
            + "        opts = [x for fo, x in fields(negotiate) if fo == 1]\n"
            + "        if b'seatbelt' in opts: confine = 2\n"  // CONFINEMENT_SEATBELT
            + "        if b'linux-namespaces' in opts: confine = 3\n"  // CONFINEMENT_LINUX_NAMESPACES
            // Record Bazel's advertised supported_confinements (field 3, packed enum varints).
            + "        sup = next((x for fo, x in fields(negotiate) if fo == 3), b'')\n"
            + "        vals = []\n"
            + "        j = 0\n"
            + "        while j < len(sup):\n"
            + "            v, j = rd_varint(sup, j); vals.append(v)\n"
            + "        if ws is not None:\n"
            + "            open(ws + '/negotiate.opts', 'wb').write(b'\\n'.join(opts))\n"
            + "            open(ws + '/negotiate.supported', 'wb').write(b','.join(str(v).encode() for v in vals))\n"
            + "        reply(uv(1, rid) + ld(6, b'')); continue\n"  // Response.negotiated = 6
            + "    if destroy is not None:\n"
            + "        reply(uv(1, rid) + ld(3, b'')); continue\n"  // Response.destroyed = 3
            + "    if collect is not None:\n"
            + "        reply(uv(1, rid) + ld(5, b'')); continue\n"  // Response.collected = 5
            + "    if create is None:\n"
            + "        reply(uv(1, rid) + ld(4, ld(1, b'unknown op'))); continue\n"  // error
            + "    manifest = None\n"
            + "    for f, v in fields(create):\n"  // Create { manifest = 1 }
            + "        if f == 1: manifest = v\n"
            + "    tmphost = None\n"
            + "    if manifest is not None:\n"
            + "        for f, v in fields(manifest):\n"
            + "            if f == 18:\n"  // Manifest.writable_dirs
            + "                kv = dict(fields(v))\n"
            + "                if kv.get(1) == b'/tmp': tmphost = kv.get(2)\n"
            + "    if not tmphost:\n"
            + "        reply(uv(1, rid) + ld(4, ld(1, b'no /tmp in manifest'))); continue\n"
            + "    scratch = tmphost[: -len(b'/tmp')]\n"
            + "    # Response{created=2: Created{path=1, confinement=2}}\n"
            + "    reply(uv(1, rid) + ld(2, ld(1, scratch) + uv(2, confine)))\n");
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
  public void virtualInput_isMaterializedToScratchAndReadableByAction() throws Exception {
    // A virtual input (param file) has no on-disk source: buildForSpawn digests its bytes directly
    // and the runner materializes them to scratch (recording a `locations` redirect). The stub
    // exposes scratch as the sandbox root, so the action — run with cwd at the sandbox exec root —
    // can read the file. Exercises the virtual-input branch of the manifest's tree + locations.
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
  public void virtualInputInSubdirectory_isMaterializedWithParentDirs() throws Exception {
    // The virtual input lives under a nested path; collectLocation must create the parent dirs in
    // scratch before writing it. Reading it back confirms the whole prefix was materialized.
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
  public void advertisesSupportedConfinementsToBackend() throws Exception {
    // Bazel sends the confinements it can enforce on this host in the Negotiate handshake. On macOS
    // that's {NONE=1, SEATBELT=2}; the stub records the list it received.
    assumeTrue(OS.getCurrent() == OS.DARWIN);
    SandboxBackendSpawnRunner runner = newRunner("acme-advert", stubBinary, ImmutableList.of());
    Spawn spawn = new SpawnBuilder("/bin/sh", "-c", "exit 0").build();
    FileOutErr fileOutErr =
        new FileOutErr(testRoot.getChild("stdout"), testRoot.getChild("stderr"));
    SpawnExecutionContextForTesting policy =
        new SpawnExecutionContextForTesting(spawn, fileOutErr, Duration.ofMinutes(1));

    var unused = runner.exec(spawn, policy);

    Path recorded = commandEnvironment.getExecRoot().getRelative("negotiate.supported");
    assertThat(new String(FileSystemUtils.readContentAsLatin1(recorded))).isEqualTo("1,2");
  }

  @Test
  public void seatbeltConfinement_wrapsActionInSandboxExec() throws Exception {
    // The backend selects CONFINEMENT_SEATBELT (via the 'seatbelt' option); the runner reuses
    // Bazel's macOS Seatbelt confinement, wrapping the argv as `sandbox-exec -f <profile> -- <argv>`.
    // Real sandbox-exec can't nest (this test already runs under darwin-sandbox), so point it at a
    // passthrough fake that drops `-f <profile>` and execs the inner argv. The action still runs end
    // to end, proving the SEATBELT branch produced a valid wrapper and wrote a profile.
    assumeTrue(OS.getCurrent() == OS.DARWIN);
    Path fakeSandboxExec = commandEnvironment.getExecRoot().getRelative("fake-sandbox-exec");
    FileSystemUtils.writeContentAsLatin1(fakeSandboxExec, "#!/bin/sh\nshift 2\nexec \"$@\"\n");
    fakeSandboxExec.setExecutable(true);
    String oldSandboxExec = SandboxBackendSpawn.sandboxExecBinary;
    SandboxBackendSpawn.sandboxExecBinary = fakeSandboxExec.getPathString();
    try {
      SandboxBackendSpawnRunner runner =
          newRunner("acme-seatbelt", stubBinary, ImmutableList.of("seatbelt"));
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
  public void backendSelectingUnadvertisedConfinement_failsClosed() throws Exception {
    // The backend picks CONFINEMENT_LINUX_NAMESPACES, which Bazel did not advertise on macOS. The
    // action must fail closed rather than run unconfined.
    assumeTrue(OS.getCurrent() == OS.DARWIN);
    SandboxBackendSpawnRunner runner =
        newRunner("acme-bad", stubBinary, ImmutableList.of("linux-namespaces"));
    Spawn spawn = new SpawnBuilder("/bin/sh", "-c", "exit 0").build();
    FileOutErr fileOutErr =
        new FileOutErr(testRoot.getChild("stdout"), testRoot.getChild("stderr"));
    SpawnExecutionContextForTesting policy =
        new SpawnExecutionContextForTesting(spawn, fileOutErr, Duration.ofMinutes(1));

    Exception e = assertThrows(Exception.class, () -> runner.exec(spawn, policy));
    assertThat(e).hasMessageThat().contains("did not advertise");
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
