// Copyright 2014 Google Inc. All rights reserved.
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

import static com.google.devtools.build.lib.rules.objc.XcodeProductType.LIBRARY_STATIC;

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.DependencyControl;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.XcodeprojBuildSetting;

/**
 * Implementation for {@code objc_library}.
 */
public class ObjcLibrary implements RuleConfiguredTargetFactory {
  static final class InfoplistsFromRule extends IterableWrapper<Artifact> {
    InfoplistsFromRule(Iterable<Artifact> infoplists) {
      super(infoplists);
    }

    InfoplistsFromRule(Artifact... infoplists) {
      super(infoplists);
    }
  }

  static OptionsProvider optionsProvider(
      RuleContext ruleContext, InfoplistsFromRule infoplistsFromRule) {
    return new OptionsProvider.Builder()
        .addCopts(ruleContext.getTokenizedStringListAttr("copts"))
        .addInfoplists(infoplistsFromRule)
        .addTransitive(Optional.fromNullable(
            ruleContext.getPrerequisite("options", Mode.TARGET, OptionsProvider.class)))
        .build();
  }

  /**
   * Constructs an {@link ObjcCommon} instance based on the attributes of the given rule. The rule
   * should inherit from {@link ObjcLibraryRule}. This method automatically calls
   * {@link ObjcCommon#reportErrors()}.
   */
  static ObjcCommon common(RuleContext ruleContext, Iterable<SdkFramework> extraSdkFrameworks) {
    CompilationArtifacts compilationArtifacts = new CompilationArtifacts.Builder()
        .addSrcs(ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET))
        .addNonArcSrcs(ruleContext.getPrerequisiteArtifacts("non_arc_srcs", Mode.TARGET))
        .setIntermediateArtifacts(ObjcBase.intermediateArtifacts(ruleContext))
        .setPchFile(Optional.fromNullable(ruleContext.getPrerequisiteArtifact("pch", Mode.TARGET)))
        .build();

    ObjcCommon common = new ObjcCommon.Builder(ruleContext)
        .addExtraSdkFrameworks(extraSdkFrameworks)
        .addAssetCatalogs(ruleContext.getPrerequisiteArtifacts("asset_catalogs", Mode.TARGET))
        .addSdkDylibs(ruleContext.attributes().get("sdk_dylibs", Type.STRING_LIST))
        .setCompilationArtifacts(compilationArtifacts)
        .addHdrs(ruleContext.getPrerequisiteArtifacts("hdrs", Mode.TARGET))
        .addDepObjcProviders(ruleContext.getPrerequisites("deps", Mode.TARGET, ObjcProvider.class))
        .build();
    common.reportErrors();

    return common;
  }

  static void registerActions(RuleContext ruleContext, ObjcCommon common,
      XcodeProvider xcodeProvider, OptionsProvider optionsProvider) {
    for (CompilationArtifacts compilationArtifacts : common.getCompilationArtifacts().asSet()) {
      ObjcBase.actionsBuilder(ruleContext)
          .registerCompileAndArchiveActions(
              compilationArtifacts, common.getObjcProvider(), optionsProvider);
    }
    ObjcBase.registerActions(ruleContext, xcodeProvider);
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    ObjcCommon common = common(ruleContext, ImmutableList.<SdkFramework>of());
    OptionsProvider optionsProvider = optionsProvider(ruleContext, new InfoplistsFromRule());

    Iterable<XcodeProvider> depXcodeProviders =
        ruleContext.getPrerequisites("deps", Mode.TARGET, XcodeProvider.class);
    XcodeProvider xcodeProvider = common.xcodeProvider(Optional.<Artifact>absent(),
        ImmutableList.<DependencyControl>of(), ImmutableList.<XcodeprojBuildSetting>of(),
        optionsProvider.getCopts(),
        LIBRARY_STATIC,
        depXcodeProviders);
    registerActions(ruleContext, common, xcodeProvider, optionsProvider);
    return common.configuredTarget(
        NestedSetBuilder.<Artifact>stableOrder()
            .addAll(common.getCompiledArchive().asSet())
            .add(ruleContext.getImplicitOutputArtifact(ObjcRuleClasses.PBXPROJ))
            .build(),
        Optional.of(xcodeProvider),
        Optional.of(common.getObjcProvider()));
  }
}
