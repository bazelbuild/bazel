// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.util.ScratchAttributeWriter;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for header thinning. */
@RunWith(JUnit4.class)
public class HeaderThinningTest extends ObjcRuleTestCase {

  private static final String CPP_COMPILE_ACTION_RULE_TYPE = "objc_library";

  @Before
  public void beforeEach() throws Exception {
    MockObjcSupport.setupCcToolchainConfig(mockToolsConfig);
    MockObjcSupport.setupAppleSdks(mockToolsConfig);
    useConfiguration(
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL,
        "--experimental_objc_header_thinning",
        "--objc_use_dotd_pruning",
        "--xcode_version=" + MockObjcSupport.DEFAULT_XCODE_VERSION,
        "--ios_sdk_version=" + MockObjcSupport.DEFAULT_IOS_SDK_VERSION);
  }

  @Test
  public void testCppCompileActionHeaderThinningCanDetermineAdditionalInputs() throws Exception {
    Artifact sourceFile = getSourceArtifact("objc/a.m");
    CppCompileAction action = createCppCompileAction(sourceFile);
    List<Artifact> expectedHeaders =
        ImmutableList.of(
            getSourceArtifact("objc/a.pch"),
            getSourceArtifact("objc/b.h"),
            getSourceArtifact("objc/c"),
            getSourceArtifact("objc/d.hpp"));
    HeaderThinning headerThinning = new HeaderThinning(getPotentialHeaders(expectedHeaders));
    writeToHeadersListFile(action, "objc/a.pch", "objc/b.h", "objc/c", "objc/d.hpp");

    Iterable<Artifact> headersFound =
        headerThinning.determineAdditionalInputs(null, action, null, null);
    assertThat(headersFound).containsExactlyElementsIn(expectedHeaders);
  }

  @Test
  public void testCppCompileActionHeaderThinningThrowsWhenUnknownHeaderFound() throws Exception {
    Artifact sourceFile = getSourceArtifact("objc/a.m");
    CppCompileAction action = createCppCompileAction(sourceFile);
    List<Artifact> expectedHeaders =
        ImmutableList.of(getSourceArtifact("objc/a.h"), getSourceArtifact("objc/b.h"));
    HeaderThinning headerThinning = new HeaderThinning(getPotentialHeaders(expectedHeaders));
    writeToHeadersListFile(action, "objc/a.h", "objc/b.h", "objc/c.h");

    try {
      headerThinning.determineAdditionalInputs(null, action, null, null);
      fail("Exception was not thrown");
    } catch (ExecException e) {
      assertThat(e).hasMessageThat().containsMatch("(objc/c.h)");
      assertThat(e).isInstanceOf(UserExecException.class);
    }
  }

  @Test
  public void testCppCompileActionHeaderThinningFindsHeadersInTreeArtifacts() throws Exception {
    Artifact sourceFile = getSourceArtifact("objc/a.m");
    CppCompileAction action = createCppCompileAction(sourceFile);
    List<Artifact> expectedHeaders =
        ImmutableList.of(getSourceArtifact("objc/a.h"), getTreeArtifact("tree/dir"));
    HeaderThinning headerThinning = new HeaderThinning(getPotentialHeaders(expectedHeaders));
    writeToHeadersListFile(action, "objc/a.h", "tree/dir/c.h");

    Iterable<Artifact> headersFound =
        headerThinning.determineAdditionalInputs(null, action, null, null);
    assertThat(headersFound).containsExactlyElementsIn(expectedHeaders);
  }

  @Test
  public void testObjcCompileActionHeaderThinningCanFindRequiredHeaderInputs() throws Exception {
    Artifact sourceFile = getSourceArtifact("objc/a.m");
    Artifact headersListFile = getHeadersListArtifact(sourceFile);
    scratch.file(
        headersListFile.getExecPathString(), "objc/a.pch", "objc/b.h", "objc/c", "objc/d.hpp");
    List<Artifact> expectedHeaders =
        ImmutableList.of(
            getSourceArtifact("objc/a.pch"),
            getSourceArtifact("objc/b.h"),
            getSourceArtifact("objc/c"),
            getSourceArtifact("objc/d.hpp"));

    Iterable<Artifact> headersFound =
        HeaderThinning.findRequiredHeaderInputs(
            sourceFile,
            headersListFile,
            createHeaderFilesMap(getPotentialHeaders(expectedHeaders)),
            ArtifactPathResolver.IDENTITY);
    assertThat(headersFound).containsExactlyElementsIn(expectedHeaders);
  }

  private void writeToHeadersListFile(CppCompileAction action, String... lines) throws Exception {
    Artifact headersListFile = null;
    for (Artifact input : action.getMandatoryInputs()) {
      if (input.getExtension().equals("headers_list")) {
        headersListFile = input;
        break;
      }
    }
    assertThat(headersListFile).isNotNull();
    scratch.file(headersListFile.getPath().getPathString(), lines);
  }

  /**
   * Simple utility to populate some unused header Artifacts along with the ones we expect to find.
   */
  private Iterable<Artifact> getPotentialHeaders(List<Artifact> expectedHeaders) {
    return Iterables.concat(
        ImmutableList.of(
            getSourceArtifact("objc/unused.h"),
            getSourceArtifact("some/other.h"),
            getSourceArtifact("objc/foo.pch")),
        expectedHeaders);
  }

  private CppCompileAction createCppCompileAction(Artifact sourceFile) throws Exception {
    String ownerLabel = "//objc:lib";
    ScratchAttributeWriter.fromLabelString(this, CPP_COMPILE_ACTION_RULE_TYPE, ownerLabel)
        .setList("srcs", sourceFile.getFilename())
        .write();
    ConfiguredTarget target = getConfiguredTarget(ownerLabel);
    List<CppCompileAction> allActions =
        actionsTestUtil()
            .findTransitivePrerequisitesOf(
                ActionsTestUtil.getFirstArtifactEndingWith(getFilesToBuild(target), ".a"),
                CppCompileAction.class);
    for (CppCompileAction action : allActions) {
      if (action.getSourceFile().getExecPath().equals(sourceFile.getExecPath())) {
        return action;
      }
    }
    return null;
  }

  private Artifact getHeadersListArtifact(Artifact sourceFile) {
    return getSourceArtifact(
        FileSystemUtils.replaceExtension(sourceFile.getExecPath(), ".headers_list"),
        sourceFile.getRoot().getRoot());
  }

  private static Map<PathFragment, Artifact> createHeaderFilesMap(Iterable<Artifact> artifacts) {
    ImmutableMap.Builder<PathFragment, Artifact> headerFilesMapBuilder = ImmutableMap.builder();
    for (Artifact artifact : artifacts) {
      headerFilesMapBuilder.put(artifact.getExecPath(), artifact);
    }
    return headerFilesMapBuilder.build();
  }

  private Artifact getTreeArtifact(String name) {
    Artifact treeArtifactBase = getSourceArtifact(name);
    return new SpecialArtifact(
        treeArtifactBase.getRoot(),
        treeArtifactBase.getExecPath(),
        treeArtifactBase.getArtifactOwner(),
        SpecialArtifactType.TREE);
  }

  @Test
  public void generatesHeaderScanningAction() throws Exception {
    Set<SpawnAction> scanningActions =
        createTargetAndGetHeaderScanningActions(ImmutableList.of("one.m", "two.m"));
    assertThat(scanningActions).hasSize(1);
  }

  @Test
  public void generates2HeaderScanningActionsWhenObjcAndCppSources() throws Exception {
    Set<SpawnAction> scanningActions =
        createTargetAndGetHeaderScanningActions(ImmutableList.of("one.m", "two.cc"));
    assertThat(scanningActions).hasSize(2);
  }

  @Test
  public void generatesMultipleHeaderScanningActionsForLargeTargets2() throws Exception {
    validateGeneratesMultipleHeaderScanningActionsForLargeTargets(
        2, targetConfig.getFragment(ObjcConfiguration.class).objcHeaderThinningPartitionSize());
  }

  @Test
  public void generatesMultipleHeaderScanningActionsForLargeTargets4() throws Exception {
    validateGeneratesMultipleHeaderScanningActionsForLargeTargets(
        4, targetConfig.getFragment(ObjcConfiguration.class).objcHeaderThinningPartitionSize());
  }

  @Test
  public void generatesMultipleHeaderScanningActionsForLargeTargets8() throws Exception {
    validateGeneratesMultipleHeaderScanningActionsForLargeTargets(
        8, targetConfig.getFragment(ObjcConfiguration.class).objcHeaderThinningPartitionSize());
  }

  @Test
  public void generatesMultipleHeaderScanningActionsForLargeTargetsCustomPartition()
      throws Exception {
    int partitionSize = 5;
    useConfiguration(
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL,
        "--experimental_objc_header_thinning",
        "--objc_header_thinning_partition_size=" + 5,
        "--objc_use_dotd_pruning",
        "--xcode_version=" + MockObjcSupport.DEFAULT_XCODE_VERSION,
        "--ios_sdk_version=" + MockObjcSupport.DEFAULT_IOS_SDK_VERSION);
    validateGeneratesMultipleHeaderScanningActionsForLargeTargets(12, partitionSize);
  }

  private void validateGeneratesMultipleHeaderScanningActionsForLargeTargets(
      int actionCount, int partitionSize) throws Exception {
    ImmutableList.Builder<String> sourcesBuilder = ImmutableList.builder();
    for (int i = 0; i < partitionSize * actionCount; ++i) {
      sourcesBuilder.add(String.format("source_%d.m", i));
    }
    Set<SpawnAction> scanningActions =
        createTargetAndGetHeaderScanningActions(sourcesBuilder.build());
    assertThat(scanningActions).hasSize(actionCount);
  }

  private Set<SpawnAction> createTargetAndGetHeaderScanningActions(Iterable<String> sources)
      throws Exception {
    String ownerLabel = "//objc:lib";
    ScratchAttributeWriter.fromLabelString(this, CPP_COMPILE_ACTION_RULE_TYPE, ownerLabel)
        .setList("srcs", sources)
        .write();
    ConfiguredTarget target = getConfiguredTarget(ownerLabel);
    List<SpawnAction> spawnActions =
        actionsTestUtil()
            .findTransitivePrerequisitesOf(
                ActionsTestUtil.getFirstArtifactEndingWith(getFilesToBuild(target), ".a"),
                SpawnAction.class);
    return Sets.newHashSet(
        Iterables.filter(spawnActions, a -> a.getMnemonic().equals("ObjcHeaderScanning")));
  }
}
