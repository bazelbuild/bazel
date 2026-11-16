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

package com.google.devtools.build.lib.remote.merkletree;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.exec.util.FakeActionInputFileCache;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.FakeSpawnExecutionContext;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

/**
 * Benchmark for computing the root digest of the Merkle tree of a spawn, as is done for every spawn
 * to check for a remote cache hit.
 *
 * <p>The spawns model actions of a single rule: all of them share a large "toolchain" nested set
 * made up of one nested set per package (think of an unpacked {@code node_modules} tree), each
 * depends on a few of a pool of "library" nested sets and each has a few inputs of its own.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 2, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 2, timeUnit = TimeUnit.SECONDS)
@Fork(1)
@State(Scope.Benchmark)
public class MerkleTreeComputerBenchmark {
  private static final int SPAWNS = 64;
  private static final int FILES_PER_PACKAGE = 20;
  private static final int LIBRARIES = 100;
  private static final int LIBRARIES_PER_SPAWN = 10;
  private static final int FILES_PER_LIBRARY = 10;
  private static final int OWN_FILES_PER_SPAWN = 10;

  /** The number of inputs in the nested set shared by all spawns. */
  @Param({"200", "2000", "20000"})
  public int sharedInputs;

  private final Random random = new Random(42);
  private final FakeActionInputFileCache metadataProvider = new FakeActionInputFileCache();
  private final List<Spawn> spawns = new ArrayList<>();
  private Path execRoot;
  private ArtifactRoot sourceRoot;
  private ArtifactRoot outputRoot;
  private MerkleTreeComputer computer;
  private int nextSpawn;

  @Setup(Level.Trial)
  public void setUp() throws Exception {
    var fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);
    execRoot = fs.getPath("/execroot/_main");
    execRoot.createDirectoryAndParents();
    sourceRoot = ArtifactRoot.asSourceRoot(Root.fromPath(execRoot));
    outputRoot = ArtifactRoot.asDerivedRoot(execRoot, ArtifactRoot.RootType.OUTPUT, "bazel-out");
    computer =
        new MerkleTreeComputer(
            new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256),
            /* remoteExecutionCache= */ null,
            "buildRequestId",
            "commandId",
            "_main");

    var toolchain = NestedSetBuilder.<ActionInput>stableOrder();
    for (int pkg = 0; pkg < sharedInputs / FILES_PER_PACKAGE; pkg++) {
      var files = NestedSetBuilder.<ActionInput>stableOrder();
      for (int file = 0; file < FILES_PER_PACKAGE; file++) {
        files.add(
            createArtifact(
                sourceRoot,
                "external/npm/node_modules/package%d/%s/file%d.js"
                    .formatted(pkg, file % 2 == 0 ? "lib" : "dist/esm", file)));
      }
      toolchain.addTransitive(files.build());
    }
    NestedSet<ActionInput> toolchainSet = toolchain.build();

    var libraries = new ArrayList<NestedSet<ActionInput>>();
    for (int lib = 0; lib < LIBRARIES; lib++) {
      var files = NestedSetBuilder.<ActionInput>stableOrder();
      for (int file = 0; file < FILES_PER_LIBRARY; file++) {
        files.add(
            createArtifact(
                outputRoot,
                "bazel-out/k8-fastbuild/bin/src/library%d/file%d.d.ts".formatted(lib, file)));
      }
      // Libraries depend on earlier libraries.
      for (int dep = 0; dep < 2 && lib > 0; dep++) {
        files.addTransitive(libraries.get(random.nextInt(lib)));
      }
      libraries.add(files.build());
    }

    for (int i = 0; i < SPAWNS; i++) {
      var inputs = NestedSetBuilder.<ActionInput>stableOrder();
      inputs.addTransitive(toolchainSet);
      for (int lib = 0; lib < LIBRARIES_PER_SPAWN; lib++) {
        inputs.addTransitive(libraries.get(random.nextInt(LIBRARIES)));
      }
      for (int file = 0; file < OWN_FILES_PER_SPAWN; file++) {
        inputs.add(createArtifact(sourceRoot, "src/target%d/file%d.ts".formatted(i, file)));
      }
      spawns.add(new SpawnBuilder().withInputs(inputs.build()).build());
    }
  }

  private Artifact createArtifact(ArtifactRoot root, String execPath) {
    var artifact = ActionsTestUtil.createArtifactWithExecPath(root, PathFragment.create(execPath));
    var digest = new byte[32];
    random.nextBytes(digest);
    metadataProvider.put(
        artifact,
        FileArtifactValue.createForNormalFile(digest, /* proxy= */ null, random.nextInt(1 << 20)));
    return artifact;
  }

  @Benchmark
  public Digest buildForSpawn() throws Exception {
    Spawn spawn = spawns.get(nextSpawn);
    nextSpawn = (nextSpawn + 1) % SPAWNS;
    return computer
        .buildForSpawn(
            spawn,
            ImmutableSet.of(),
            /* scrubber= */ null,
            new FakeSpawnExecutionContext(
                spawn,
                metadataProvider,
                execRoot,
                new FileOutErr(),
                ImmutableClassToInstanceMap.of(),
                /* actionFileSystem= */ null),
            MerkleTreeComputer.BlobPolicy.DISCARD)
        .digest();
  }
}
