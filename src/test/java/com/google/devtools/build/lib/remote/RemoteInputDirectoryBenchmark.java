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
package com.google.devtools.build.lib.remote;

import static org.mockito.Mockito.mock;

import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
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
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/** Cold index construction and warm lookup costs with 250,000 checked remote inputs. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 1, time = 1)
@Measurement(iterations = 2, time = 2)
@Fork(1)
public class RemoteInputDirectoryBenchmark {
  private static final int INPUT_COUNT = 250_000;

  @Benchmark
  public void firstOutputStat(ColdInputs inputs, Blackhole blackhole) throws IOException {
    blackhole.consume(inputs.cold.statIfFound(inputs.outputFile.getPath().asFragment(), false));
  }

  @Benchmark
  public void firstOutputReadlink(ColdInputs inputs, Blackhole blackhole) throws IOException {
    try {
      inputs.cold.readSymbolicLink(inputs.outputFile.getPath().asFragment());
      throw new AssertionError("Output should be absent");
    } catch (FileNotFoundException e) {
      blackhole.consume(e);
    }
  }

  @Benchmark
  public void outputCacheCheck(WarmInputs inputs, Blackhole blackhole) throws IOException {
    RemoteActionFileSystem actionFs = inputs.newFileSystem();
    blackhole.consume(actionFs.statIfFound(inputs.outputFile.getPath().asFragment(), false));
  }

  @Benchmark
  public void firstLocalDigest(ColdInputs inputs, Blackhole blackhole) throws IOException {
    blackhole.consume(inputs.cold.getFastDigest(inputs.localFile));
  }

  @Benchmark
  public void warmLocalDigest(WarmPhysicalInputs inputs, Blackhole blackhole) throws IOException {
    blackhole.consume(inputs.warm.getFastDigest(inputs.localFile));
  }

  @Benchmark
  public void firstUnrelatedMiss(ColdInputs inputs, Blackhole blackhole) throws IOException {
    blackhole.consume(inputs.cold.statIfFound(inputs.missing, /* followSymlinks= */ false));
  }

  @Benchmark
  public void twoUnrelatedMisses(ColdInputs inputs, Blackhole blackhole) throws IOException {
    blackhole.consume(inputs.cold.statIfFound(inputs.missing, /* followSymlinks= */ false));
    blackhole.consume(inputs.cold.statIfFound(inputs.secondMissing, /* followSymlinks= */ false));
  }

  @Benchmark
  public void firstSparseInput(ColdInputs inputs, Blackhole blackhole) throws IOException {
    blackhole.consume(inputs.cold.stat(inputs.sparseInput, /* followSymlinks= */ true));
  }

  @Benchmark
  public void firstPhysicalReaddir(ColdInputs inputs, Blackhole blackhole) throws IOException {
    blackhole.consume(inputs.cold.getDirectoryEntries(inputs.physicalDirectory));
  }

  @Benchmark
  public void twoPhysicalReaddirs(ColdInputs inputs, Blackhole blackhole) throws IOException {
    blackhole.consume(inputs.cold.getDirectoryEntries(inputs.physicalDirectory));
    blackhole.consume(inputs.cold.getDirectoryEntries(inputs.physicalDirectory));
  }

  @Benchmark
  public void firstVirtualReaddir(ColdInputs inputs, Blackhole blackhole) throws IOException {
    blackhole.consume(inputs.cold.getDirectoryEntries(inputs.virtualDirectory));
  }

  @Benchmark
  public void warmSparseInput(WarmInputs inputs, Blackhole blackhole) throws IOException {
    blackhole.consume(inputs.warm.stat(inputs.sparseInput, /* followSymlinks= */ true));
  }

  @Benchmark
  public void warmPhysicalReaddir(WarmInputs inputs, Blackhole blackhole) throws IOException {
    blackhole.consume(inputs.warm.getDirectoryEntries(inputs.physicalDirectory));
  }

  @Benchmark
  public void warmVirtualReaddir(WarmInputs inputs, Blackhole blackhole) throws IOException {
    blackhole.consume(inputs.warm.getDirectoryEntries(inputs.virtualDirectory));
  }

  @Benchmark
  public void warmInputWithPhysicalParent(WarmPhysicalInputs inputs, Blackhole blackhole)
      throws IOException {
    blackhole.consume(
        inputs.warm.stat(inputs.remoteInputWithPhysicalParent, /* followSymlinks= */ true));
  }

  @State(Scope.Thread)
  public abstract static class Inputs {
    @Param({"flat", "distinct"})
    public String layout;

    private FileSystem localFs;
    private Path root;
    private PathFragment execRoot;
    private ActionInputMap checked;
    private RemoteActionInputFetcher fetcher;
    protected PathFragment missing;
    protected PathFragment secondMissing;
    protected PathFragment sparseInput;
    protected PathFragment physicalDirectory;
    protected PathFragment virtualDirectory;
    protected PathFragment remoteInputWithPhysicalParent;
    protected PathFragment localFile;
    protected Artifact outputFile;

    @Setup(Level.Trial)
    public void setup() throws IOException {
      localFs = FileSystems.getNativeFileSystem(DigestHashFunction.SHA256);
      root = TestUtils.createUniqueTmpDir(localFs);
      execRoot = root.asFragment();
      ArtifactRoot output = ArtifactRoot.asDerivedRoot(root, RootType.OUTPUT, "out");
      output.getRoot().asPath().createDirectoryAndParents();
      physicalDirectory = output.getRoot().asPath().getRelative("unrelated-physical").asFragment();
      virtualDirectory = output.getRoot().asPath().getRelative("inputs").asFragment();
      localFs.getPath(physicalDirectory).createDirectory();
      localFile = physicalDirectory.getChild("local-file");
      localFs.getPath(localFile).getOutputStream().close();
      Path remoteInputParent = output.getRoot().asPath().getRelative("materialized");
      remoteInputParent.createDirectory();
      missing = output.getRoot().asPath().getRelative("unrelated").asFragment();
      secondMissing = output.getRoot().asPath().getRelative("also-unrelated").asFragment();
      checked = new ActionInputMap(INPUT_COUNT);
      FileArtifactValue metadata =
          FileArtifactValue.createForRemoteFileWithMaterializationData(
              DigestHashFunction.SHA256.getHashFunction().hashBytes(new byte[0]).asBytes(),
              /* size= */ 0,
              /* locationIndex= */ 1,
              /* expirationTime= */ null,
              /* inMemoryOutput= */ false);
      for (int i = 0; i < INPUT_COUNT; i++) {
        Artifact artifact =
            ActionsTestUtil.createArtifact(
                output,
                layout.equals("flat") ? "inputs/dir/file-" + i : "inputs/dir-" + i + "/file");
        checked.put(artifact, metadata);
        if (i == 0) {
          sparseInput = artifact.getPath().asFragment();
        }
      }
      Artifact physicalParentInput = ActionsTestUtil.createArtifact(output, "materialized/file");
      remoteInputWithPhysicalParent = physicalParentInput.getPath().asFragment();
      checked.put(physicalParentInput, metadata);
      outputFile = ActionsTestUtil.createArtifact(output, "output-file");
      fetcher = mock(RemoteActionInputFetcher.class);
    }

    @TearDown(Level.Trial)
    public void tearDown() throws IOException {
      root.deleteTree();
    }

    protected RemoteActionFileSystem newFileSystem() {
      return new RemoteActionFileSystem(
          localFs, execRoot, "out", checked, checked, List.of(outputFile), fetcher);
    }
  }

  @State(Scope.Thread)
  public static class ColdInputs extends Inputs {
    private RemoteActionFileSystem cold;

    @Setup(Level.Invocation)
    public void resetCold() {
      cold = newFileSystem();
    }
  }

  @State(Scope.Thread)
  public static class WarmPhysicalInputs extends Inputs {
    private RemoteActionFileSystem warm;

    @Setup(Level.Iteration)
    public void warmFileSystem() throws IOException {
      warm = newFileSystem();
      warm.stat(remoteInputWithPhysicalParent, /* followSymlinks= */ true);
      warm.getFastDigest(localFile);
    }
  }

  @State(Scope.Thread)
  public static class WarmInputs extends Inputs {
    private RemoteActionFileSystem warm;

    @Setup(Level.Iteration)
    public void warmFileSystem() throws IOException {
      warm = newFileSystem();
      warm.statIfFound(missing, /* followSymlinks= */ false);
      warm.statIfFound(secondMissing, /* followSymlinks= */ false);
      warm.stat(remoteInputWithPhysicalParent, /* followSymlinks= */ true);
      warm.getDirectoryEntries(physicalDirectory);
      warm.getFastDigest(localFile);
    }
  }
}
