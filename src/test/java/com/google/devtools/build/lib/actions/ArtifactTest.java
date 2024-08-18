// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.mock;

import com.google.common.base.Equivalence;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.Lists;
import com.google.common.testing.EqualsTester;
import com.google.common.testing.EquivalenceTester;
import com.google.devtools.build.lib.actions.Artifact.ArchivedTreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactSerializationContext;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.OwnerlessArtifactWrapper;
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.LabelArtifactOwner;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.rules.cpp.CppFileTypes;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.serialization.AutoRegistry;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueStore;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationDepsUtils;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.starlarkbuildapi.FileRootApi;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.Root.RootCodecDependencies;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link Artifact}. */
@RunWith(TestParameterInjector.class)
public final class ArtifactTest {

  private final Scratch scratch = new Scratch();
  private Path execDir;
  private ArtifactRoot rootDir;

  @Before
  public void setRootDir() throws Exception {
    execDir = scratch.dir("/base/exec");
    rootDir = ArtifactRoot.asDerivedRoot(execDir, RootType.Output, "root");
  }

  @Test
  public void testConstruction_badRootDir() throws IOException {
    Path f1 = scratch.file("/exec/dir/file.ext");
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ActionsTestUtil.createArtifactWithExecPath(
                    ArtifactRoot.asDerivedRoot(execDir, RootType.Output, "bogus"),
                    f1.relativeTo(execDir))
                .getRootRelativePath());
  }

  private static long getUsedMemory() {
    System.gc();
    System.gc();
    System.runFinalization();
    System.gc();
    System.gc();
    return Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
  }

  @Test
  public void testMemoryUsage() throws IOException {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.dir("/foo")));
    PathFragment aPath = PathFragment.create("src/a");
    int arrSize = 1 << 20;
    Object[] arr = new Object[arrSize];
    long usedMemory = getUsedMemory();
    for (int i = 0; i < arrSize; i++) {
      arr[i] = ActionsTestUtil.createArtifactWithExecPath(root, aPath);
    }
    assertThat((getUsedMemory() - usedMemory) / arrSize).isAtMost(34L);
  }

  @Test
  public void testEquivalenceRelation() throws Exception {
    PathFragment aPath = PathFragment.create("src/a");
    PathFragment bPath = PathFragment.create("src/b");
    assertThat(ActionsTestUtil.createArtifactWithRootRelativePath(rootDir, aPath))
        .isEqualTo(ActionsTestUtil.createArtifactWithRootRelativePath(rootDir, aPath));
    assertThat(ActionsTestUtil.createArtifactWithRootRelativePath(rootDir, bPath))
        .isEqualTo(ActionsTestUtil.createArtifactWithRootRelativePath(rootDir, bPath));
    assertThat(
            ActionsTestUtil.createArtifactWithRootRelativePath(rootDir, aPath)
                .equals(ActionsTestUtil.createArtifactWithRootRelativePath(rootDir, bPath)))
        .isFalse();
  }

  @Test
  public void testComparison() {
    PathFragment aPath = PathFragment.create("src/a");
    PathFragment bPath = PathFragment.create("src/b");
    Artifact aArtifact = ActionsTestUtil.createArtifactWithRootRelativePath(rootDir, aPath);
    Artifact bArtifact = ActionsTestUtil.createArtifactWithRootRelativePath(rootDir, bPath);
    assertThat(Artifact.EXEC_PATH_COMPARATOR.compare(aArtifact, bArtifact)).isEqualTo(-1);
    assertThat(Artifact.EXEC_PATH_COMPARATOR.compare(aArtifact, aArtifact)).isEqualTo(0);
    assertThat(Artifact.EXEC_PATH_COMPARATOR.compare(bArtifact, bArtifact)).isEqualTo(0);
    assertThat(Artifact.EXEC_PATH_COMPARATOR.compare(bArtifact, aArtifact)).isEqualTo(1);
  }

  @Test
  public void testGetFilename() throws Exception {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.dir("/foo")));
    Artifact javaFile = ActionsTestUtil.createArtifact(root, scratch.file("/foo/Bar.java"));
    Artifact generatedHeader =
        ActionsTestUtil.createArtifact(root, scratch.file("/foo/bar.proto.h"));
    Artifact generatedCc = ActionsTestUtil.createArtifact(root, scratch.file("/foo/bar.proto.cc"));
    Artifact aCPlusPlusFile = ActionsTestUtil.createArtifact(root, scratch.file("/foo/bar.cc"));
    assertThat(JavaSemantics.JAVA_SOURCE.matches(javaFile.getFilename())).isTrue();
    assertThat(CppFileTypes.CPP_HEADER.matches(generatedHeader.getFilename())).isTrue();
    assertThat(CppFileTypes.CPP_SOURCE.matches(generatedCc.getFilename())).isTrue();
    assertThat(CppFileTypes.CPP_SOURCE.matches(aCPlusPlusFile.getFilename())).isTrue();
  }

  @Test
  public void testGetExtension() throws Exception {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.dir("/foo")));
    Artifact javaFile = ActionsTestUtil.createArtifact(root, scratch.file("/foo/Bar.java"));
    assertThat(javaFile.getExtension()).isEqualTo("java");
  }

  @Test
  public void testIsFileType() throws Exception {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.dir("/foo")));
    Artifact javaFile = ActionsTestUtil.createArtifact(root, scratch.file("/foo/Bar.java"));
    assertThat(javaFile.isFileType(FileType.of("java"))).isTrue();
    assertThat(javaFile.isFileType(FileType.of("cc"))).isFalse();
  }

  @Test
  public void testIsFileTypeSet() throws Exception {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.dir("/foo")));
    Artifact javaFile = ActionsTestUtil.createArtifact(root, scratch.file("/foo/Bar.java"));
    assertThat(javaFile.isFileType(FileTypeSet.of(FileType.of("cc"), FileType.of("java"))))
        .isTrue();
    assertThat(javaFile.isFileType(FileTypeSet.of(FileType.of("py"), FileType.of("js")))).isFalse();
    assertThat(javaFile.isFileType(FileTypeSet.of())).isFalse();
  }

  @Test
  public void testMangledPath() {
    String path = "dir/sub_dir/name:end";
    assertThat(Actions.escapedPath(path)).isEqualTo("dir_Ssub_Udir_Sname_Cend");
  }

  private List<Artifact> getFooBarArtifacts() throws Exception {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.dir("/foo")));
    Artifact aHeader1 = ActionsTestUtil.createArtifact(root, scratch.file("/foo/bar1.h"));
    Artifact aHeader2 = ActionsTestUtil.createArtifact(root, scratch.file("/foo/bar2.h"));
    return Lists.newArrayList(aHeader1, aHeader2);
  }

  @Test
  public void testAddExecPaths() throws Exception {
    List<String> paths = new ArrayList<>();
    Artifact.addExecPaths(getFooBarArtifacts(), paths);
    assertThat(paths).containsExactly("bar1.h", "bar2.h");
  }

  @Test
  public void testAddExecPathsNewActionGraph() throws Exception {
    List<String> paths = new ArrayList<>();
    Artifact.addExecPaths(getFooBarArtifacts(), paths);
    assertThat(paths).containsExactly("bar1.h", "bar2.h");
  }

  @Test
  public void testRootRelativePathIsSameAsExecPath() throws Exception {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.dir("/foo")));
    Artifact a = ActionsTestUtil.createArtifact(root, scratch.file("/foo/bar1.h"));
    assertThat(a.getRootRelativePath()).isSameInstanceAs(a.getExecPath());
  }

  @Test
  public void testToDetailString() throws Exception {
    Path execRoot = scratch.getFileSystem().getPath("/execroot/workspace");
    Artifact a =
        ActionsTestUtil.createArtifact(
            ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, "b"), "c");
    assertThat(a.toDetailString()).isEqualTo("[[<execution_root>]b]c");
  }

  @Test
  public void testWeirdArtifact() {
    Path execRoot = scratch.getFileSystem().getPath("/");
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ActionsTestUtil.createArtifactWithExecPath(
                    ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, "a"),
                    PathFragment.create("c"))
                .getRootRelativePath());
  }

  @Test
  public void derivedArtifactCodecs(
      @TestParameter boolean includeGeneratingActionKey, @TestParameter boolean useSharedValues)
      throws Exception {
    ArtifactSerializationContext artifactContext =
        new ArtifactSerializationContext() {
          @Override
          public SourceArtifact getSourceArtifact(
              PathFragment execPath, Root root, ArtifactOwner owner) {
            throw new UnsupportedOperationException();
          }

          @Override
          public boolean includeGeneratingActionKey(DerivedArtifact artifact) {
            return includeGeneratingActionKey;
          }
        };

    DerivedArtifact artifact =
        (DerivedArtifact) ActionsTestUtil.createArtifact(rootDir, "dir/out.txt");
    artifact.setGeneratingActionKey(ActionsTestUtil.NULL_ACTION_LOOKUP_DATA);

    ArtifactRoot anotherRoot =
        ArtifactRoot.asDerivedRoot(scratch.getFileSystem().getPath("/"), RootType.Output, "other");
    DerivedArtifact anotherArtifact =
        DerivedArtifact.create(
            anotherRoot,
            anotherRoot.getExecPath().getRelative("dir/out.txt"),
            ActionsTestUtil.NULL_ARTIFACT_OWNER);
    anotherArtifact.setGeneratingActionKey(ActionsTestUtil.NULL_ACTION_LOOKUP_DATA);

    SpecialArtifact tree =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(
            rootDir, rootDir.getExecPath().getRelative("tree"));
    TreeFileArtifact treeChild = TreeFileArtifact.createTreeOutput(tree, "child");
    ArchivedTreeArtifact archivedTree = ArchivedTreeArtifact.createForTree(tree);
    ArchivedTreeArtifact customArchivedTree =
        ArchivedTreeArtifact.createWithCustomDerivedTreeRoot(
            tree, PathFragment.create("custom"), PathFragment.create("archived.zip"));

    SpecialArtifact templateExpansionTree =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(
            rootDir, rootDir.getExecPath().getRelative("template"));
    TreeFileArtifact expansionOutput =
        TreeFileArtifact.createTemplateExpansionOutput(
            templateExpansionTree,
            "output",
            ActionsTestUtil.NULL_TEMPLATE_EXPANSION_ARTIFACT_OWNER);
    expansionOutput.setGeneratingActionKey(
        ActionLookupData.create(ActionsTestUtil.NULL_TEMPLATE_EXPANSION_ARTIFACT_OWNER, 0));

    SerializationTester tester =
        new SerializationTester(
                artifact,
                anotherArtifact,
                tree,
                treeChild,
                archivedTree,
                customArchivedTree,
                expansionOutput)
            .addDependency(FileSystem.class, scratch.getFileSystem())
            .addDependency(
                RootCodecDependencies.class, new RootCodecDependencies(anotherRoot.getRoot()))
            .addDependency(ArtifactSerializationContext.class, artifactContext);

    if (useSharedValues) {
      for (ObjectCodec<? extends Artifact> codec : ArtifactCodecs.VALUE_SHARING_CODECS) {
        tester.addCodec(codec);
      }
      tester.makeMemoizingAndAllowFutureBlocking(/* allowFutureBlocking= */ true);
    }

    if (!includeGeneratingActionKey) {
      tester.<DerivedArtifact>setVerificationFunction(
          (original, deserialized) -> {
            String debug =
                String.format(
                    "original=%s\ndeseriaized=%s",
                    original.toDebugString(), deserialized.toDebugString());
            assertWithMessage(debug).that(deserialized.hasGeneratingActionKey()).isFalse();
            assertWithMessage(debug).that(deserialized.equalsWithoutOwner(original)).isTrue();
            assertThat(new OwnerlessArtifactWrapper(deserialized))
                .isEqualTo(new OwnerlessArtifactWrapper(original));

            assertThrows(debug, RuntimeException.class, deserialized::getArtifactOwner);
            assertThrows(debug, RuntimeException.class, deserialized::getGeneratingActionKey);
            assertThrows(debug, RuntimeException.class, deserialized::getOwner);
            assertThrows(debug, RuntimeException.class, deserialized::getOwnerLabel);
            assertThrows(debug, RuntimeException.class, () -> deserialized.equals(original));
            assertThrows(
                debug,
                RuntimeException.class,
                () -> deserialized.setGeneratingActionKey(original.getGeneratingActionKey()));
          });
    }

    tester.runTests();
  }

  @Test
  public void sourceArtifactCodecRecyclesSourceArtifactInstances(
      @TestParameter boolean useSharedValues) throws Exception {
    Root root = Root.fromPath(scratch.dir("/"));
    ArtifactRoot artifactRoot = ArtifactRoot.asSourceRoot(root);
    ArtifactFactory artifactFactory =
        new ArtifactFactory(execDir.getParentDirectory(), "blaze-out");

    ObjectCodecs objectCodecs =
        new ObjectCodecs(
            AutoRegistry.get()
                .getBuilder()
                .addReferenceConstant(scratch.getFileSystem())
                .setAllowDefaultCodec(true)
                .build(),
            ImmutableClassToInstanceMap.builder()
                .put(FileSystem.class, scratch.getFileSystem())
                .put(ArtifactSerializationContext.class, artifactFactory::getSourceArtifact)
                .put(RootCodecDependencies.class, new RootCodecDependencies(artifactRoot.getRoot()))
                .build());

    FingerprintValueService service = null;
    if (useSharedValues) {
      service = FingerprintValueService.createForTesting(FingerprintValueStore.inMemoryStore());
      for (ObjectCodec<? extends Artifact> codec : ArtifactCodecs.VALUE_SHARING_CODECS) {
        objectCodecs = objectCodecs.withCodecOverridesForTesting(codec);
      }
    }

    PathFragment pathFragment = PathFragment.create("src/foo.cc");
    ArtifactOwner owner = new LabelArtifactOwner(Label.parseCanonicalUnchecked("//foo:bar"));
    SourceArtifact sourceArtifact = new SourceArtifact(artifactRoot, pathFragment, owner);

    SourceArtifact deserialized1;
    SourceArtifact deserialized2;
    if (useSharedValues) {
      deserialized1 =
          (SourceArtifact)
              objectCodecs.deserializeMemoizedAndBlocking(
                  service,
                  objectCodecs
                      .serializeMemoizedAndBlocking(
                          service, sourceArtifact, /* profileCollector= */ null)
                      .getObject());
      deserialized2 =
          (SourceArtifact)
              objectCodecs.deserializeMemoizedAndBlocking(
                  service,
                  objectCodecs
                      .serializeMemoizedAndBlocking(
                          service, sourceArtifact, /* profileCollector= */ null)
                      .getObject());
    } else {
      deserialized1 =
          (SourceArtifact) objectCodecs.deserialize(objectCodecs.serialize(sourceArtifact));
      deserialized2 =
          (SourceArtifact) objectCodecs.deserialize(objectCodecs.serialize(sourceArtifact));
    }
    assertThat(deserialized1).isSameInstanceAs(deserialized2);

    Artifact sourceArtifactFromFactory =
        artifactFactory.getSourceArtifact(pathFragment, root, owner);
    Artifact deserialized;
    if (useSharedValues) {
      deserialized =
          (Artifact)
              objectCodecs.deserializeMemoizedAndBlocking(
                  service,
                  objectCodecs
                      .serializeMemoizedAndBlocking(
                          service, sourceArtifactFromFactory, /* profileCollector= */ null)
                      .getObject());
    } else {
      deserialized =
          (Artifact) objectCodecs.deserialize(objectCodecs.serialize(sourceArtifactFromFactory));
    }
    assertThat(sourceArtifactFromFactory).isSameInstanceAs(deserialized);
  }

  @Test
  public void testLongDirname() throws Exception {
    String dirName = createDirNameArtifact().getDirname();

    assertThat(dirName).isEqualTo("aaa/bbb/ccc");
  }

  @Test
  public void testDirnameInExecutionDir() throws Exception {
    Artifact artifact =
        ActionsTestUtil.createArtifact(
            ArtifactRoot.asSourceRoot(Root.fromPath(scratch.dir("/foo"))),
            scratch.file("/foo/bar.txt"));

    assertThat(artifact.getDirname()).isEqualTo(".");
  }

  @Test
  public void testCanConstructPathFromDirAndFilename() throws Exception {
    Artifact artifact = createDirNameArtifact();
    String constructed = String.format("%s/%s", artifact.getDirname(), artifact.getFilename());

    assertThat(constructed).isEqualTo("aaa/bbb/ccc/ddd");
  }

  @Test
  public void testIsSourceArtifact() throws Exception {
    assertThat(
            new Artifact.SourceArtifact(
                    ArtifactRoot.asSourceRoot(Root.fromPath(scratch.dir("/"))),
                    PathFragment.create("src/foo.cc"),
                    ArtifactOwner.NULL_OWNER)
                .isSourceArtifact())
        .isTrue();
    assertThat(
            ActionsTestUtil.createArtifact(
                    ArtifactRoot.asDerivedRoot(scratch.dir("/genfiles"), RootType.Output, "aaa"),
                    scratch.file("/genfiles/aaa/bar.out"))
                .isSourceArtifact())
        .isFalse();
  }

  @Test
  public void testGetRoot() throws Exception {
    Path execRoot = scratch.getFileSystem().getPath("/");
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, "newRoot");
    assertThat(ActionsTestUtil.createArtifact(root, scratch.file("/newRoot/foo")).getRoot())
        .isEqualTo(root);
  }

  @Test
  public void hashCodeAndEquals() {
    Path execRoot = scratch.getFileSystem().getPath("/");
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, "newRoot");
    ActionLookupKey firstOwner =
        new ActionLookupKey() {
          @Override
          public Label getLabel() {
            return null;
          }

          @Override
          public BuildConfigurationKey getConfigurationKey() {
            return null;
          }

          @Override
          public SkyFunctionName functionName() {
            return null;
          }
        };
    ActionLookupKey secondOwner =
        new ActionLookupKey() {
          @Override
          public Label getLabel() {
            return null;
          }

          @Override
          public BuildConfigurationKey getConfigurationKey() {
            return null;
          }

          @Override
          public SkyFunctionName functionName() {
            return null;
          }
        };
    DerivedArtifact derived1 =
        DerivedArtifact.create(root, PathFragment.create("newRoot/shared"), firstOwner);
    derived1.setGeneratingActionKey(ActionLookupData.create(firstOwner, 0));
    DerivedArtifact derived2 =
        DerivedArtifact.create(root, PathFragment.create("newRoot/shared"), secondOwner);
    derived2.setGeneratingActionKey(ActionLookupData.create(secondOwner, 0));
    ArtifactRoot sourceRoot = ArtifactRoot.asSourceRoot(Root.fromPath(root.getRoot().asPath()));
    Artifact source1 = new SourceArtifact(sourceRoot, PathFragment.create("shared"), firstOwner);
    Artifact source2 = new SourceArtifact(sourceRoot, PathFragment.create("shared"), secondOwner);
    new EqualsTester()
        .addEqualityGroup(derived1)
        .addEqualityGroup(derived2)
        .addEqualityGroup(source1, source2)
        .testEquals();
    assertThat(derived1.hashCode()).isNotEqualTo(derived2.hashCode());
    assertThat(derived1.hashCode()).isNotEqualTo(source1.hashCode());
    assertThat(source1.hashCode()).isEqualTo(source2.hashCode());
    Artifact.OwnerlessArtifactWrapper wrapper1 = new Artifact.OwnerlessArtifactWrapper(derived1);
    Artifact.OwnerlessArtifactWrapper wrapper2 = new Artifact.OwnerlessArtifactWrapper(derived2);
    Artifact.OwnerlessArtifactWrapper wrapper3 = new Artifact.OwnerlessArtifactWrapper(source1);
    Artifact.OwnerlessArtifactWrapper wrapper4 = new Artifact.OwnerlessArtifactWrapper(source2);
    new EqualsTester()
        .addEqualityGroup(wrapper1, wrapper2)
        .addEqualityGroup(wrapper3, wrapper4)
        .testEquals();
    Path path1 = derived1.getPath();
    Path path2 = derived2.getPath();
    Path path3 = source1.getPath();
    Path path4 = source2.getPath();
    new EqualsTester().addEqualityGroup(path1, path2, path3, path4).testEquals();
  }

  private Artifact createDirNameArtifact() throws Exception {
    return ActionsTestUtil.createArtifact(
        ArtifactRoot.asSourceRoot(Root.fromPath(scratch.dir("/"))),
        scratch.file("/aaa/bbb/ccc/ddd"));
  }

  @Test
  public void canDeclareContentBasedOutput() {
    Path execRoot = scratch.getFileSystem().getPath("/");
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, "newRoot");
    assertThat(
            DerivedArtifact.create(
                    root,
                    PathFragment.create("newRoot/my.output"),
                    ActionsTestUtil.NULL_ARTIFACT_OWNER,
                    /*contentBasedPath=*/ true)
                .contentBasedPath())
        .isTrue();
  }

  @Test
  public void testGetRepositoryRelativePathExternalSourceArtifacts() throws IOException {
    ArtifactRoot externalRoot =
        ArtifactRoot.asExternalSourceRoot(
            Root.fromPath(
                scratch
                    .dir("/output_base")
                    .getRelative(LabelConstants.EXTERNAL_REPOSITORY_LOCATION)
                    .getRelative("foo")));

    // --experimental_sibling_repository_layout not set
    assertThat(
            new Artifact.SourceArtifact(
                    externalRoot,
                    LabelConstants.EXTERNAL_PATH_PREFIX.getRelative("foo/bar/baz.cc"),
                    ArtifactOwner.NULL_OWNER)
                .getRepositoryRelativePath())
        .isEqualTo(PathFragment.create("bar/baz.cc"));

    // --experimental_sibling_repository_layout set
    assertThat(
            new Artifact.SourceArtifact(
                    externalRoot,
                    LabelConstants.EXPERIMENTAL_EXTERNAL_PATH_PREFIX.getRelative("foo/bar/baz.cc"),
                    ArtifactOwner.NULL_OWNER)
                .getRepositoryRelativePath())
        .isEqualTo(PathFragment.create("bar/baz.cc"));
  }

  @Test
  public void archivedTreeArtifact_create_returnsArtifactInArchivedRoot() {
    ArtifactRoot root =
        ArtifactRoot.asDerivedRoot(execDir, RootType.Output, "blaze-out", "fastbuild");
    SpecialArtifact tree = createTreeArtifact(root, "tree");

    ArchivedTreeArtifact archivedTreeArtifact = ArchivedTreeArtifact.createForTree(tree);

    assertThat(archivedTreeArtifact.getParent()).isSameInstanceAs(tree);
    assertThat(archivedTreeArtifact.getArtifactOwner())
        .isSameInstanceAs(ActionsTestUtil.NULL_ARTIFACT_OWNER);
    assertThat(archivedTreeArtifact.getExecPathString())
        .isEqualTo("blaze-out/:archived_tree_artifacts/fastbuild/tree.zip");
    assertThat(archivedTreeArtifact.getRoot().getExecPathString())
        .isEqualTo("blaze-out/:archived_tree_artifacts/fastbuild");
  }

  @Test
  public void archivedTreeArtifact_create_returnsArtifactWithGeneratingActionFromParent() {
    ActionLookupKey actionLookupKey = mock(ActionLookupKey.class);
    ActionLookupData actionLookupData = ActionLookupData.create(actionLookupKey, 0);
    SpecialArtifact tree = createTreeArtifact(rootDir, "tree", actionLookupData);

    ArchivedTreeArtifact archivedTreeArtifact = ArchivedTreeArtifact.createForTree(tree);

    assertThat(archivedTreeArtifact.getExecPathString())
        .isEqualTo("root/:archived_tree_artifacts/tree.zip");
    assertThat(archivedTreeArtifact.getArtifactOwner()).isSameInstanceAs(actionLookupKey);
    assertThat(archivedTreeArtifact.getGeneratingActionKey()).isSameInstanceAs(actionLookupData);
  }

  @Test
  public void archivedTreeArtifact_createWithCustomDerivedTreeRoot_returnsArtifactWithCustomRoot() {
    ArtifactRoot root =
        ArtifactRoot.asDerivedRoot(execDir, RootType.Output, "blaze-out", "fastbuild");
    SpecialArtifact tree = createTreeArtifact(root, "dir/tree");

    ArchivedTreeArtifact archivedTreeArtifact =
        ArchivedTreeArtifact.createWithCustomDerivedTreeRoot(
            tree, PathFragment.create("custom/custom2"), PathFragment.create("treePath/file.xyz"));

    assertThat(archivedTreeArtifact.getParent()).isSameInstanceAs(tree);
    assertThat(archivedTreeArtifact.getExecPathString())
        .isEqualTo("blaze-out/custom/custom2/fastbuild/treePath/file.xyz");
    assertThat(archivedTreeArtifact.getRoot().getExecPathString())
        .isEqualTo("blaze-out/custom/custom2/fastbuild");
  }

  @Test
  public void archivedTreeArtifact_codec_roundTripsArchivedArtifact(
      @TestParameter boolean useSharedValues) throws Exception {
    ArchivedTreeArtifact artifact1 = createArchivedTreeArtifact(rootDir, "tree1");
    ArtifactRoot anotherRoot =
        ArtifactRoot.asDerivedRoot(scratch.getFileSystem().getPath("/"), RootType.Output, "src");
    ArchivedTreeArtifact artifact2 = createArchivedTreeArtifact(anotherRoot, "tree2");
    SerializationTester tester =
        new SerializationTester(artifact1, artifact2)
            .addDependency(FileSystem.class, scratch.getFileSystem())
            .addDependency(
                RootCodecDependencies.class, new RootCodecDependencies(anotherRoot.getRoot()))
            .addDependencies(SerializationDepsUtils.SERIALIZATION_DEPS_FOR_TEST)
            .<ArchivedTreeArtifact>setVerificationFunction(
                (original, deserialized) -> {
                  assertThat(original).isEqualTo(deserialized);
                  assertThat(original.getGeneratingActionKey())
                      .isEqualTo(deserialized.getGeneratingActionKey());
                });
    if (useSharedValues) {
      for (ObjectCodec<? extends Artifact> codec : ArtifactCodecs.VALUE_SHARING_CODECS) {
        tester.addCodec(codec);
      }
      tester.makeMemoizingAndAllowFutureBlocking(/* allowFutureBlocking= */ true);
    }
    tester.runTests();
  }

  @Test
  public void archivedTreeArtifact_getExecPathWithinArchivedArtifactsTree_returnsCorrectPath() {
    assertThat(
            ArchivedTreeArtifact.getExecPathWithinArchivedArtifactsTree(
                PathFragment.create("bazel-out/k8-fastbuild/bin/dir/subdir")))
        .isEqualTo(
            PathFragment.create("bazel-out/:archived_tree_artifacts/k8-fastbuild/bin/dir/subdir"));
  }

  private static final PathMapper PATH_MAPPER =
      new PathMapper() {
        @Override
        public PathFragment map(PathFragment execPath) {
          if (execPath.startsWith(PathFragment.create("output"))) {
            // output/k8-opt/bin/path/to/pkg/file --> output/<hash>/path/to/pkg/file
            return execPath
                .subFragment(0, 1)
                .getRelative(Integer.toUnsignedString(execPath.subFragment(3).hashCode()))
                .getRelative(execPath.subFragment(3));
          } else {
            return execPath;
          }
        }
      };

  @Test
  public void mappedArtifact() {
    StarlarkSemantics semantics = PATH_MAPPER.storeIn(StarlarkSemantics.DEFAULT);

    Root sourceRoot = Root.fromPath(scratch.getFileSystem().getPath("/some/path"));
    ArtifactRoot sourceArtifactRoot = ArtifactRoot.asSourceRoot(sourceRoot);
    Artifact sourceArtifact1 =
        ActionsTestUtil.createArtifactWithExecPath(
            sourceArtifactRoot, PathFragment.create("path/to/pkg/file1"));
    Artifact sourceArtifact2 =
        ActionsTestUtil.createArtifactWithExecPath(
            sourceArtifactRoot, PathFragment.create("path/to/pkg/file2"));

    Path execRoot = scratch.getFileSystem().getPath("/some/path");
    ArtifactRoot outputArtifactRoot =
        ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, "output", "k8-opt", "bin");
    Artifact outputArtifact1 =
        ActionsTestUtil.createArtifactWithExecPath(
            outputArtifactRoot, PathFragment.create("output/k8-opt/bin/path/to/pkg/file1"));
    Artifact outputArtifact2 =
        ActionsTestUtil.createArtifactWithExecPath(
            outputArtifactRoot, PathFragment.create("output/k8-opt/bin/path/to/pkg/file2"));

    assertThat(sourceArtifact1.getExecPathStringForStarlark(semantics))
        .isEqualTo("path/to/pkg/file1");
    assertThat(sourceArtifact1.getDirnameForStarlark(semantics)).isEqualTo("path/to/pkg");

    FileRootApi mappedSourceRoot1 = sourceArtifact1.getRootForStarlark(semantics);
    assertThat(mappedSourceRoot1.getExecPathString()).isEqualTo("");

    assertThat(sourceArtifact2.getExecPathStringForStarlark(semantics))
        .isEqualTo("path/to/pkg/file2");
    assertThat(sourceArtifact2.getDirnameForStarlark(semantics)).isEqualTo("path/to/pkg");

    FileRootApi mappedSourceRoot2 = sourceArtifact1.getRootForStarlark(semantics);
    assertThat(mappedSourceRoot2.getExecPathString()).isEqualTo("");

    assertThat(outputArtifact1.getExecPathStringForStarlark(semantics))
        .isEqualTo("output/3540078408/path/to/pkg/file1");
    assertThat(outputArtifact1.getDirnameForStarlark(semantics))
        .isEqualTo("output/3540078408/path/to/pkg");

    FileRootApi mappedOutputRoot1 = outputArtifact1.getRootForStarlark(semantics);
    assertThat(mappedOutputRoot1.getExecPathString()).isEqualTo("output/3540078408");

    assertThat(outputArtifact2.getExecPathStringForStarlark(semantics))
        .isEqualTo("output/3540078409/path/to/pkg/file2");
    assertThat(outputArtifact2.getDirnameForStarlark(semantics))
        .isEqualTo("output/3540078409/path/to/pkg");

    FileRootApi mappedOutputRoot2 = outputArtifact2.getRootForStarlark(semantics);
    assertThat(mappedOutputRoot2.getExecPathString()).isEqualTo("output/3540078409");

    // Starlark equality uses Object#equals.
    // Mapped roots are always distinct from non-mapped roots, even if their paths are equal.
    new EqualsTester()
        .addEqualityGroup(mappedSourceRoot1, mappedSourceRoot2)
        .addEqualityGroup(mappedOutputRoot1)
        .addEqualityGroup(mappedOutputRoot2)
        .addEqualityGroup(sourceRoot)
        .addEqualityGroup(outputArtifactRoot)
        .testEquals();

    var starlarkCompare =
        new Equivalence<Comparable<?>>() {
          @Override
          protected boolean doEquivalent(Comparable<?> a, Comparable<?> b) {
            // Compare a and b in both directions as the implementations of compareTo may be
            // different.
            return Starlark.ORDERING.compare(a, b) == 0 && Starlark.ORDERING.compare(b, a) == 0;
          }

          @Override
          protected int doHash(Comparable<?> comparable) {
            return 0;
          }
        };

    ClassCastException e =
        assertThrows(
            ClassCastException.class,
            () -> Starlark.ORDERING.compare(mappedOutputRoot1, outputArtifactRoot));
    assertThat(e).hasMessageThat().isEqualTo("unsupported comparison: mapped_root <=> root");

    EquivalenceTester.of(starlarkCompare)
        .addEquivalenceGroup((Comparable) mappedSourceRoot1, (Comparable) mappedSourceRoot2)
        .addEquivalenceGroup((Comparable) mappedOutputRoot1)
        .addEquivalenceGroup((Comparable) mappedOutputRoot2)
        .test();
  }

  private static SpecialArtifact createTreeArtifact(ArtifactRoot root, String relativePath) {
    return createTreeArtifact(root, relativePath, ActionsTestUtil.NULL_ACTION_LOOKUP_DATA);
  }

  private static SpecialArtifact createTreeArtifact(
      ArtifactRoot root, String relativePath, ActionLookupData actionLookupData) {
    SpecialArtifact treeArtifact =
        SpecialArtifact.create(
            root,
            root.getExecPath().getRelative(relativePath),
            actionLookupData.getActionLookupKey(),
            SpecialArtifactType.TREE);
    treeArtifact.setGeneratingActionKey(actionLookupData);
    return treeArtifact;
  }

  private static ArchivedTreeArtifact createArchivedTreeArtifact(
      ArtifactRoot root, String treeRelativePath) {
    return ArchivedTreeArtifact.createForTree(createTreeArtifact(root, treeRelativePath));
  }
}
