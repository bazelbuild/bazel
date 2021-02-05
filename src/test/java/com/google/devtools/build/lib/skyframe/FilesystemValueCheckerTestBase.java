// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArchivedTreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileStatusWithDigestAdapter;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.SkyFunctionName;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Helper class to allow sharing test helpers between {@link FilesystemValueCheckerTest} and {@link
 * FilesystemValueCheckerParameterizedTest}.
 */
public class FilesystemValueCheckerTestBase {

  static final int FSVC_THREADS_FOR_TEST = 200;
  static final ActionLookupKey ACTION_LOOKUP_KEY =
      new ActionLookupKey() {
        @Override
        public SkyFunctionName functionName() {
          return SkyFunctionName.FOR_TESTING;
        }

        @Nullable
        @Override
        public Label getLabel() {
          return null;
        }
      };

  final MockFileSystem fs = new MockFileSystem();

  SpecialArtifact createTreeArtifact(String relPath) throws IOException {
    String outSegment = "bin";
    Path outputDir = fs.getPath("/" + outSegment);
    Path outputPath = outputDir.getRelative(relPath);
    outputDir.createDirectory();
    ArtifactRoot derivedRoot =
        ArtifactRoot.asDerivedRoot(fs.getPath("/"), RootType.Output, outSegment);
    return ActionsTestUtil.createTreeArtifactWithGeneratingAction(
        derivedRoot,
        derivedRoot.getExecPath().getRelative(derivedRoot.getRoot().relativize(outputPath)));
  }

  static ActionExecutionValue actionValueWithTreeArtifacts(List<TreeFileArtifact> contents)
      throws IOException {
    return actionValueWithTreeArtifacts(contents, ImmutableList.of());
  }

  static ActionExecutionValue actionValueWithTreeArtifacts(
      Iterable<TreeFileArtifact> contents, Iterable<ArchivedTreeArtifact> archivedTreeArtifacts)
      throws IOException {
    TreeArtifactValue.MultiBuilder treeArtifacts = TreeArtifactValue.newMultiBuilder();

    for (TreeFileArtifact output : contents) {
      treeArtifacts.putChild(output, createMetadataFromFileSystem(output));
    }

    for (ArchivedTreeArtifact archivedTreeArtifact : archivedTreeArtifacts) {
      treeArtifacts.setArchivedRepresentation(
          archivedTreeArtifact, createMetadataFromFileSystem(archivedTreeArtifact));
    }

    Map<Artifact, TreeArtifactValue> treeArtifactData = new HashMap<>();
    treeArtifacts.injectTo(treeArtifactData::put);

    return ActionExecutionValue.create(
        /*artifactData=*/ ImmutableMap.of(),
        treeArtifactData,
        /*outputSymlinks=*/ null,
        /*discoveredModules=*/ null,
        /*actionDependsOnBuildId=*/ false);
  }

  private static FileArtifactValue createMetadataFromFileSystem(Artifact artifact)
      throws IOException {
    Path path = artifact.getPath();
    FileArtifactValue noDigest =
        ActionMetadataHandler.fileArtifactValueFromArtifact(
            artifact, FileStatusWithDigestAdapter.adapt(path.statIfFound(Symlinks.NOFOLLOW)), null);
    return FileArtifactValue.createFromInjectedDigest(
        noDigest, path.getDigest(), !artifact.isConstantMetadata());
  }

  void writeFile(Path path, String... lines) throws IOException {
    // Make sure we advance the clock to detect modifications which do not change the size, which
    // rely on ctime.
    fs.advanceClockMillis(1);
    FileSystemUtils.writeIsoLatin1(path, lines);
  }

  static final class MockFileSystem extends InMemoryFileSystem {
    boolean statThrowsRuntimeException;
    boolean readlinkThrowsIoException;

    MockFileSystem() {
      this(new ManualClock());
    }

    private MockFileSystem(ManualClock clock) {
      super(clock, DigestHashFunction.SHA256);
    }

    @Override
    public FileStatus statIfFound(Path path, boolean followSymlinks) throws IOException {
      if (statThrowsRuntimeException) {
        throw new RuntimeException("bork");
      }
      return super.statIfFound(path, followSymlinks);
    }

    @Override
    protected PathFragment readSymbolicLink(Path path) throws IOException {
      if (readlinkThrowsIoException) {
        throw new IOException("readlink failed");
      }
      return super.readSymbolicLink(path);
    }

    void advanceClockMillis(int millis) {
      ((ManualClock) clock).advanceMillis(millis);
    }
  }
}
