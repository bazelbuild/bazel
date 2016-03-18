// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.ideinfo;

import static com.google.common.collect.Iterables.transform;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.BuildView;
import com.google.devtools.build.lib.analysis.BuildView.AnalysisResult;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.actions.BinaryFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.ArtifactLocation;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.LibraryArtifact;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.RuleIdeInfo;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.RuleIdeInfo.Kind;
import com.google.devtools.build.lib.skyframe.AspectValue;
import com.google.protobuf.TextFormat;

import org.junit.Before;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import javax.annotation.Nullable;

/**
 * Provides utils for AndroidStudioInfoAspectTest.
 */
abstract class AndroidStudioInfoAspectTestBase extends BuildViewTestCase {

  protected static final Function<ArtifactLocation, String> ARTIFACT_TO_RELATIVE_PATH =
      new Function<ArtifactLocation, String>() {
        @Nullable
        @Override
        public String apply(ArtifactLocation artifactLocation) {
          return artifactLocation.getRelativePath();
        }
      };
  protected static final Function<LibraryArtifact, String> LIBRARY_ARTIFACT_TO_STRING =
      new Function<LibraryArtifact, String>() {
        @Override
        public String apply(LibraryArtifact libraryArtifact) {
          StringBuilder stringBuilder = new StringBuilder();
          if (libraryArtifact.hasJar()) {
            stringBuilder.append("<jar:");
            stringBuilder.append(libraryArtifact.getJar().getRelativePath());
            stringBuilder.append(">");
          }
          if (libraryArtifact.hasInterfaceJar()) {
            stringBuilder.append("<ijar:");
            stringBuilder.append(libraryArtifact.getInterfaceJar().getRelativePath());
            stringBuilder.append(">");
          }
          if (libraryArtifact.hasSourceJar()) {
            stringBuilder.append("<source:");
            stringBuilder.append(libraryArtifact.getSourceJar().getRelativePath());
            stringBuilder.append(">");
          }

          return stringBuilder.toString();
        }
      };

  protected ConfiguredAspect configuredAspect;

  /**
   * Constructs a string that matches OutputJar#toString for comparison testing.
   */
  protected static String jarString(String base, String jar, String iJar, String sourceJar) {
    StringBuilder sb = new StringBuilder();
    if (jar != null) {
      sb.append("<jar:" + base + "/" + jar + ">");
    }
    if (iJar != null) {
      sb.append("<ijar:" + base + "/" + iJar + ">");
    }
    if (sourceJar != null) {
      sb.append("<source:" + base + "/" + sourceJar + ">");
    }
    return sb.toString();
  }

  protected static Iterable<String> relativePathsForJavaSourcesOf(RuleIdeInfo ruleIdeInfo) {
    return relativePathsForSources(ruleIdeInfo.getJavaRuleIdeInfo().getSourcesList());
  }

  protected static Iterable<String> relativePathsForCSourcesOf(RuleIdeInfo ruleIdeInfo) {
    return relativePathsForSources(ruleIdeInfo.getCRuleIdeInfo().getSourceList());
  }

  protected static Iterable<String> relativePathsForExportedCHeadersOf(RuleIdeInfo ruleIdeInfo) {
    return relativePathsForSources(ruleIdeInfo.getCRuleIdeInfo().getExportedHeaderList());
  }

  private static Iterable<String> relativePathsForSources(List<ArtifactLocation> sourcesList) {
    return transform(sourcesList, ARTIFACT_TO_RELATIVE_PATH);
  }

  protected RuleIdeInfo getRuleInfoAndVerifyLabel(
      String target, Map<String, RuleIdeInfo> ruleIdeInfos) {
    RuleIdeInfo ruleIdeInfo = ruleIdeInfos.get(target);
    assertThat(ruleIdeInfo).named(target).isNotNull();
    assertThat(ruleIdeInfo.getLabel()).isEqualTo(target);
    return ruleIdeInfo;
  }

  protected Entry<String, RuleIdeInfo> getCcToolchainRuleAndVerifyThereIsOnlyOne(
      Map<String, RuleIdeInfo> ruleIdeInfos) {
    Entry<String, RuleIdeInfo> toolchainInfo = null;
    for (Entry<String, RuleIdeInfo> entry : ruleIdeInfos.entrySet()) {
      if (entry.getValue().getKind() == Kind.CC_TOOLCHAIN) {
        // Make sure we only have 1.
        assertThat(toolchainInfo).isNull();
        assertThat(entry.getValue().hasCToolchainIdeInfo()).isTrue();
        toolchainInfo = entry;
      }
    }
    assertThat(toolchainInfo).isNotNull();
    return toolchainInfo;
  }

  protected void buildTarget(String target) throws Exception {
    AnalysisResult analysisResult =
        update(
            ImmutableList.of(target),
            ImmutableList.of(AndroidStudioInfoAspect.NAME),
            false,
            LOADING_PHASE_THREADS,
            true,
            new EventBus());
    Collection<AspectValue> aspects = analysisResult.getAspects();
    assertThat(aspects.size()).isEqualTo(1);
    AspectValue value = aspects.iterator().next();
    this.configuredAspect = value.getConfiguredAspect();
    assertThat(configuredAspect.getName()).isEqualTo(AndroidStudioInfoAspect.NAME);
  }

  @Before
  public void setupBzl() throws Exception {
    if (isNativeTest()) {
      return;
    }

    InputStream stream = AndroidStudioInfoAspectTestBase.class
        .getResourceAsStream("intellij_info.bzl");
    BufferedReader reader =
        new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8));
    String line;
    ArrayList<String> contents = new ArrayList<>();
    while ((line = reader.readLine()) != null) {
      contents.add(line);
    }

    scratch.file("intellij_tools/BUILD", "# empty");
    scratch.file("intellij_tools/intellij_info.bzl", contents.toArray(new String[0]));
  }


  /**
   * Returns a map of (label as string) -> RuleIdeInfo for each rule in the transitive
   * closure of the passed target.
   */
  protected Map<String, RuleIdeInfo> buildRuleIdeInfo(String target) throws Exception {
    if (isNativeTest()) {
      buildTarget(target);
      AndroidStudioInfoFilesProvider provider =
          configuredAspect.getProvider(AndroidStudioInfoFilesProvider.class);
      Iterable<Artifact> artifacts = provider.getIdeInfoFiles();
      ImmutableMap.Builder<String, RuleIdeInfo> builder = ImmutableMap.builder();
      for (Artifact artifact : artifacts) {
        Action generatingAction = getGeneratingAction(artifact);
        if (generatingAction instanceof BinaryFileWriteAction) {
          BinaryFileWriteAction writeAction = (BinaryFileWriteAction) generatingAction;
          RuleIdeInfo ruleIdeInfo = RuleIdeInfo.parseFrom(writeAction.getSource().openStream());
          builder.put(ruleIdeInfo.getLabel(), ruleIdeInfo);
        } else {
          verifyPackageManifestSpawnAction(generatingAction);
        }
      }
      return builder.build();
    } else {
      BuildView.AnalysisResult analysisResult = update(
          ImmutableList.of(target),
          ImmutableList.of("intellij_tools/intellij_info.bzl%intellij_info_aspect"),
          false,
          LOADING_PHASE_THREADS,
          true,
          new EventBus()
      );
      Collection<AspectValue> aspects = analysisResult.getAspects();
      assertThat(aspects).hasSize(1);
      AspectValue aspectValue = aspects.iterator().next();
      this.configuredAspect = aspectValue.getConfiguredAspect();
      OutputGroupProvider provider = configuredAspect.getProvider(OutputGroupProvider.class);
      NestedSet<Artifact> outputGroup = provider.getOutputGroup("ide-info-text");
      Map<String, RuleIdeInfo> ruleIdeInfos = new HashMap<>();
      for (Artifact artifact : outputGroup) {
        Action generatingAction = getGeneratingAction(artifact);
        assertThat(generatingAction).isInstanceOf(FileWriteAction.class);
        String fileContents = ((FileWriteAction) generatingAction).getFileContents();
        RuleIdeInfo.Builder builder = RuleIdeInfo.newBuilder();
        TextFormat.getParser().merge(fileContents, builder);
        RuleIdeInfo ruleIdeInfo = builder.build();
        ruleIdeInfos.put(ruleIdeInfo.getLabel(), ruleIdeInfo);
      }
      return ruleIdeInfos;

    }
  }
  
  protected final void verifyPackageManifestSpawnAction(Action genAction) {
    assertEquals(genAction.getMnemonic(), "JavaPackageManifest");
    SpawnAction action = (SpawnAction) genAction;
    assertFalse(action.isShellCommand());
  }
  
  protected List<String> getOutputGroupResult(String outputGroup) {
    OutputGroupProvider outputGroupProvider =
        this.configuredAspect.getProvider(OutputGroupProvider.class);
    assert outputGroupProvider != null;
    NestedSet<Artifact> artifacts = outputGroupProvider.getOutputGroup(outputGroup);

    for (Artifact artifact : artifacts) {
      assertWithMessage("Artifact %s has no generating action", artifact)
          .that(getGeneratingAction(artifact))
          .isNotNull();
    }

    List<String> artifactRelativePaths = Lists.newArrayList();
    for (Artifact artifact : artifacts) {
      artifactRelativePaths.add(artifact.getRootRelativePathString());
    }
    return artifactRelativePaths;
  }

  protected List<String> getIdeResolveFiles() {
    return getOutputGroupResult(AndroidStudioInfoAspect.IDE_RESOLVE);
  }

  protected abstract boolean isNativeTest();
}
