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
package com.google.devtools.build.lib.analysis;

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.NULL_ACTION_OWNER;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.SourceManifestAction.ManifestType;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.io.Writer;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link SourceManifestAction}.
 */
@RunWith(JUnit4.class)
public final class SourceManifestActionTest extends BuildViewTestCase {

  private Map<PathFragment, Artifact> fakeManifest;
  private Artifact buildFile;
  private Artifact relativeSymlink;
  private Artifact absoluteSymlink;
  private Artifact manifestOutputFile;

  @Before
  public void createFiles() throws Exception {
    analysisMock.pySupport().setup(mockToolsConfig);
    // Test with a raw manifest Action.
    fakeManifest = new LinkedHashMap<>();
    ArtifactRoot trivialRoot =
        ArtifactRoot.asSourceRoot(Root.fromPath(rootDirectory.getRelative("trivial")));
    Path buildFilePath =
        scratch.file("trivial/BUILD", "py_binary(name='trivial', srcs =['trivial.py'])");
    buildFile = ActionsTestUtil.createArtifact(trivialRoot, buildFilePath);

    Path pythonSourcePath =
        scratch.file("trivial/trivial.py", "#!/usr/bin/python \n print 'Hello World'");
    Artifact pythonSourceFile = ActionsTestUtil.createArtifact(trivialRoot, pythonSourcePath);
    fakeManifest.put(buildFilePath.relativeTo(rootDirectory), buildFile);
    fakeManifest.put(pythonSourcePath.relativeTo(rootDirectory), pythonSourceFile);
    ArtifactRoot outputDir =
        ArtifactRoot.asDerivedRoot(rootDirectory, RootType.Output, "blaze-output");
    outputDir.getRoot().asPath().createDirectoryAndParents();
    manifestOutputFile =
        ActionsTestUtil.createArtifact(
            outputDir, rootDirectory.getRelative("blaze-output/trivial.runfiles_manifest"));

    Path relativeSymlinkPath = outputDir.getRoot().asPath().getChild("relative_symlink");
    relativeSymlinkPath.createSymbolicLink(PathFragment.create("../some/relative/path"));
    relativeSymlink =
        SpecialArtifact.create(
            outputDir,
            outputDir.getExecPath().getChild("relative_symlink"),
            ActionsTestUtil.NULL_ARTIFACT_OWNER,
            SpecialArtifactType.UNRESOLVED_SYMLINK);
    Path absoluteSymlinkPath = outputDir.getRoot().asPath().getChild("absolute_symlink");
    absoluteSymlinkPath.createSymbolicLink(PathFragment.create("/absolute/path"));
    absoluteSymlink =
        SpecialArtifact.create(
            outputDir,
            outputDir.getExecPath().getChild("absolute_symlink"),
            ActionsTestUtil.NULL_ARTIFACT_OWNER,
            SpecialArtifactType.UNRESOLVED_SYMLINK);
  }

  private SourceManifestAction createSymlinkAction() {
    return createAction(ManifestType.SOURCE_SYMLINKS, true);
  }

  private SourceManifestAction createSourceOnlyAction() {
    return createAction(ManifestType.SOURCES_ONLY, true);
  }

  private SourceManifestAction createAction(ManifestType type, boolean addInitPy) {
    Runfiles.Builder builder = new Runfiles.Builder("TESTING", false);
    builder.addSymlinks(fakeManifest);
    if (addInitPy) {
      builder.setEmptyFilesSupplier(analysisMock.pySupport().getEmptyRunfilesSupplier());
    }
    return new SourceManifestAction(type, NULL_ACTION_OWNER, manifestOutputFile, builder.build());
  }

  /** Manifest writer that validates an expected call sequence. */
  private final class MockManifestWriter implements SourceManifestAction.ManifestWriter {
    private final List<Map.Entry<PathFragment, Artifact>> expectedSequence = new ArrayList<>();

    MockManifestWriter() {
      expectedSequence.addAll(fakeManifest.entrySet());
    }

    @Override
    public void writeEntry(
        Writer manifestWriter,
        PathFragment rootRelativePath,
        @Nullable PathFragment symlinkTarget) {
      assertWithMessage("Expected manifest input to be exhausted")
          .that(expectedSequence)
          .isNotEmpty();
      Map.Entry<PathFragment, Artifact> expectedEntry = expectedSequence.remove(0);
      assertThat(rootRelativePath)
          .isEqualTo(PathFragment.create("TESTING").getRelative(expectedEntry.getKey()));
      assertThat(symlinkTarget).isEqualTo(expectedEntry.getValue().getPath().asFragment());
    }

    int unconsumedInputs() {
      return expectedSequence.size();
    }

    @Override
    public String getMnemonic() {
      return null;
    }

    @Override
    public String getRawProgressMessage() {
      return null;
    }

    @Override
    public boolean isRemotable() {
      return false;
    }

    @Override
    public boolean emitsAbsolutePaths() {
      return false;
    }
  }

  /**
   * Tests that SourceManifestAction calls its manifest writer with the expected call sequence.
   */
  @Test
  public void testManifestWriterIntegration() throws Exception {
    MockManifestWriter mockWriter = new MockManifestWriter();
    String manifestContents =
        new SourceManifestAction(
                mockWriter,
                NULL_ACTION_OWNER,
                manifestOutputFile,
                new Runfiles.Builder("TESTING", false).addSymlinks(fakeManifest).build())
            .getFileContents(reporter);
    assertThat(mockWriter.unconsumedInputs()).isEqualTo(0);
    assertThat(manifestContents).isEmpty();
  }

  @Test
  public void testSimpleFileWriting() throws Exception {
    String manifestContents = createSymlinkAction().getFileContents(reporter);
    assertThat(manifestContents)
        .isEqualTo(
            """
            TESTING/trivial/BUILD /workspace/trivial/BUILD
            TESTING/trivial/__init__.py\s
            TESTING/trivial/trivial.py /workspace/trivial/trivial.py
            """);
  }

  /**
   * Tests that the source-only formatting strategy includes relative paths only
   * (i.e. not symlinks).
   */
  @Test
  public void testSourceOnlyFormatting() throws Exception {
    String manifestContents = createSourceOnlyAction().getFileContents(reporter);
    assertThat(manifestContents)
        .isEqualTo(
            """
            TESTING/trivial/BUILD
            TESTING/trivial/__init__.py
            TESTING/trivial/trivial.py
            """);
  }

  /**
   * Test that a directory which has only a .so file in the manifest triggers
   * the inclusion of a __init__.py file for that directory.
   */
  @Test
  public void testSwigLibrariesTriggerInitDotPyInclusion() throws Exception {
    ArtifactRoot swiggedLibPath =
        ArtifactRoot.asSourceRoot(Root.fromPath(rootDirectory.getRelative("swig")));
    Path swiggedFile = scratch.file("swig/fakeLib.so");
    Artifact swigDotSO = ActionsTestUtil.createArtifact(swiggedLibPath, swiggedFile);
    fakeManifest.put(swiggedFile.relativeTo(rootDirectory), swigDotSO);
    String manifestContents = createSymlinkAction().getFileContents(reporter);
    assertThat(manifestContents).containsMatch(".*TESTING/swig/__init__.py .*");
    assertThat(manifestContents).containsMatch("fakeLib.so");
  }

  @Test
  public void testNoPythonOrSwigLibrariesDoNotTriggerInitDotPyInclusion() throws Exception {
    ArtifactRoot nonPythonPath =
        ArtifactRoot.asSourceRoot(Root.fromPath(rootDirectory.getRelative("not_python")));
    Path nonPythonFile = scratch.file("not_python/blob_of_data");
    Artifact nonPython = ActionsTestUtil.createArtifact(nonPythonPath, nonPythonFile);
    fakeManifest.put(nonPythonFile.relativeTo(rootDirectory), nonPython);
    String manifestContents = createSymlinkAction().getFileContents(reporter);
    assertThat(manifestContents).doesNotContain("not_python/__init__.py \n");
    assertThat(manifestContents).containsMatch("blob_of_data");
  }

  @Test
  public void testGetMnemonic() {
    assertThat(createSymlinkAction().getMnemonic()).isEqualTo("SourceSymlinkManifest");
    assertThat(createAction(ManifestType.SOURCE_SYMLINKS, false).getMnemonic())
        .isEqualTo("SourceSymlinkManifest");
    assertThat(createSourceOnlyAction().getMnemonic()).isEqualTo("PackagingSourcesManifest");
  }

  @Test
  public void testSymlinkProgressMessage() {
    String progress = createSymlinkAction().getProgressMessage();
    assertWithMessage("null action not found in " + progress)
        .that(progress.contains("//null/action:owner"))
        .isTrue();
  }

  @Test
  public void testSymlinkProgressMessageNoPyInitFiles() {
    String progress = createAction(ManifestType.SOURCE_SYMLINKS, false).getProgressMessage();
    assertWithMessage("null action not found in " + progress)
        .that(progress.contains("//null/action:owner"))
        .isTrue();
  }

  @Test
  public void testSourceOnlyProgressMessage() {
    SourceManifestAction action =
        new SourceManifestAction(
            ManifestType.SOURCES_ONLY,
            NULL_ACTION_OWNER,
            getBinArtifactWithNoOwner("trivial.runfiles_manifest"),
            Runfiles.EMPTY);
    String progress = action.getProgressMessage();
    assertWithMessage("null action not found in " + progress)
        .that(progress.contains("//null/action:owner"))
        .isTrue();
  }

  @Test
  public void testRootSymlinksAffectKey() {
    Artifact manifest1 = getBinArtifactWithNoOwner("manifest1");
    Artifact manifest2 = getBinArtifactWithNoOwner("manifest2");

    SourceManifestAction action1 =
        new SourceManifestAction(
            ManifestType.SOURCE_SYMLINKS,
            NULL_ACTION_OWNER,
            manifest1,
            new Runfiles.Builder("TESTING", false)
                .addRootSymlinks(ImmutableMap.of(PathFragment.create("a"), buildFile))
                .build());

    SourceManifestAction action2 =
        new SourceManifestAction(
            ManifestType.SOURCE_SYMLINKS,
            NULL_ACTION_OWNER,
            manifest2,
            new Runfiles.Builder("TESTING", false)
                .addRootSymlinks(ImmutableMap.of(PathFragment.create("b"), buildFile))
                .build());

    assertThat(computeKey(action2)).isNotEqualTo(computeKey(action1));
  }

  // Regression test for b/116254698.
  @Test
  public void testEmptyFilesAffectKey() {
    Artifact manifest1 = getBinArtifactWithNoOwner("manifest1");
    Artifact manifest2 = getBinArtifactWithNoOwner("manifest2");

    SourceManifestAction action1 =
        new SourceManifestAction(
            ManifestType.SOURCE_SYMLINKS,
            NULL_ACTION_OWNER,
            manifest1,
            new Runfiles.Builder("TESTING", false)
                .addSymlink(PathFragment.create("a"), buildFile)
                .setEmptyFilesSupplier(
                    new Runfiles.EmptyFilesSupplier() {
                      @Override
                      public ImmutableSet<PathFragment> getExtraPaths(
                          Set<PathFragment> manifestPaths) {
                        return manifestPaths.stream()
                            .map(p -> p.replaceName(p.getBaseName() + "~"))
                            .collect(toImmutableSet());
                      }

                      @Override
                      public void fingerprint(Fingerprint fingerprint) {
                        fingerprint.addInt(1);
                      }
                    })
                .build());

    SourceManifestAction action2 =
        new SourceManifestAction(
            ManifestType.SOURCE_SYMLINKS,
            NULL_ACTION_OWNER,
            manifest2,
            new Runfiles.Builder("TESTING", false)
                .addSymlink(PathFragment.create("a"), buildFile)
                .setEmptyFilesSupplier(
                    new Runfiles.EmptyFilesSupplier() {
                      @Override
                      public ImmutableSet<PathFragment> getExtraPaths(
                          Set<PathFragment> manifestPaths) {
                        return manifestPaths.stream()
                            .map(p -> p.replaceName(p.getBaseName() + "~~"))
                            .collect(toImmutableSet());
                      }

                      @Override
                      public void fingerprint(Fingerprint fingerprint) {
                        fingerprint.addInt(2);
                      }
                    })
                .build());

    assertThat(computeKey(action2)).isNotEqualTo(computeKey(action1));
  }

  @Test
  public void testUnresolvedSymlink() throws Exception {
    Artifact manifest = getBinArtifactWithNoOwner("manifest1");

    SourceManifestAction action =
        new SourceManifestAction(
            ManifestType.SOURCE_SYMLINKS,
            NULL_ACTION_OWNER,
            manifest,
            new Runfiles.Builder("TESTING", false)
                .addArtifact(absoluteSymlink)
                .addArtifact(buildFile)
                .addArtifact(relativeSymlink)
                .build());

    NestedSet<Artifact> inputs = action.getInputs();
    assertThat(inputs.toList()).containsExactly(absoluteSymlink, relativeSymlink);

    // Verify that the return value of getInputs is cached.
    assertThat(inputs).isEqualTo(action.getInputs());
    assertThat(inputs.toList()).isEqualTo(action.getInputs().toList());

    assertThat(action.getFileContents(reporter))
        .isEqualTo(
            """
            TESTING/BUILD /workspace/trivial/BUILD
            TESTING/absolute_symlink /absolute/path
            TESTING/relative_symlink ../some/relative/path
            """);
  }

  @Test
  public void testEscaping() throws Exception {
    Artifact manifest = getBinArtifactWithNoOwner("manifest1");

    ArtifactRoot trivialRoot =
        ArtifactRoot.asSourceRoot(Root.fromPath(rootDirectory.getRelative("trivial")));
    Path fileWithSpacePath = scratch.file("trivial/file with sp\\ace", "foo");
    Artifact fileWithSpaceAndBackslash = ActionsTestUtil.createArtifact(trivialRoot, fileWithSpacePath);

    SourceManifestAction action =
        new SourceManifestAction(
            ManifestType.SOURCE_SYMLINKS,
            NULL_ACTION_OWNER,
            manifest,
            new Runfiles.Builder("TESTING", false)
                .addSymlink(PathFragment.create("no/sp\\ace"), buildFile)
                .addSymlink(PathFragment.create("also/no/sp\\ace"), fileWithSpaceAndBackslash)
                .addSymlink(PathFragment.create("with sp\\ace"), buildFile)
                .addSymlink(PathFragment.create("also/with sp\\ace"), fileWithSpaceAndBackslash)
                .build());

    assertThat(action.getFileContents(reporter))
        .isEqualTo(
            """
            TESTING/also/no/sp\\ace /workspace/trivial/file with sp\\ace
             TESTING/also/with\\ssp\\bace /workspace/trivial/file with sp\\ace
            TESTING/no/sp\\ace /workspace/trivial/BUILD
             TESTING/with\\ssp\\bace /workspace/trivial/BUILD
            """);
  }

  private String computeKey(SourceManifestAction action) {
    Fingerprint fp = new Fingerprint();
    action.computeKey(actionKeyContext, /* artifactExpander= */ null, fp);
    return fp.hexDigestAndReset();
  }
}
