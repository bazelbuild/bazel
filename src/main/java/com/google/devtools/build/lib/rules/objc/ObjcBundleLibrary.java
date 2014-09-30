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

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.HEADER;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.INCLUDE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.NESTED_BUNDLE;

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.objc.ObjcActionsBuilder.ExtraActoolArgs;
import com.google.devtools.build.lib.rules.objc.ObjcActionsBuilder.ExtraLinkArgs;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.DependencyControl;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.XcodeprojBuildSetting;

/**
 * Implementation for {@code objc_bundle_library}.
 */
public class ObjcBundleLibrary implements RuleConfiguredTargetFactory {
  static void registerActions(
      RuleContext ruleContext, Bundling bundling,
      ObjcCommon common, XcodeProvider xcodeProvider, OptionsProvider optionsProvider,
      ExtraLinkArgs extraLinkArgs, ExtraActoolArgs extraActoolArgs) {
    ObjcConfiguration objcConfiguration = ObjcActionsBuilder.objcConfiguration(ruleContext);
    InfoplistMerging infoplistMerging = bundling.getInfoplistMerging();
    ObjcProvider objcProvider = common.getObjcProvider();

    for (Artifact linkedBinary : bundling.getLinkedBinary().asSet()) {
      ruleContext.getAnalysisEnvironment().registerAction(
          ObjcActionsBuilder.linkAction(
              ruleContext, linkedBinary, objcConfiguration, objcProvider, extraLinkArgs));
    }

    for (Artifact actoolzipOutput : bundling.getActoolzipOutput().asSet()) {
      ruleContext.registerAction(
          ObjcActionsBuilder.actoolzipAction(
              ruleContext, objcConfiguration,
              ruleContext.getPrerequisiteArtifact("$actoolzip_deploy", Mode.HOST),
              common.getObjcProvider(), actoolzipOutput,
              extraActoolArgs));
    }

    ObjcActionsBuilder.registerAll(ruleContext, infoplistMerging.getMergeAction().asSet());

    ObjcActionsBuilder.registerAll(
        ruleContext,
        ObjcActionsBuilder.baseActions(
            ruleContext, common.getCompilationArtifacts(), objcProvider, xcodeProvider,
            optionsProvider));
  }

  // TODO(bazel-team): Factor out the logic this method has in common with ObjcLibrary.create into
  // shared methods.
  @Override
  public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    IntermediateArtifacts intermediateArtifacts = new IntermediateArtifacts(
        ruleContext.getAnalysisEnvironment(), ruleContext.getBinOrGenfilesDirectory(),
        ruleContext.getLabel());

    CompilationArtifacts compilationArtifacts = new CompilationArtifacts.Builder()
        .addSrcs(ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET))
        .addNonArcSrcs(ruleContext.getPrerequisiteArtifacts("non_arc_srcs", Mode.TARGET))
        .setIntermediateArtifacts(intermediateArtifacts)
        .setPchFile(Optional.fromNullable(ruleContext.getPrerequisiteArtifact("pch", Mode.TARGET)))
        .build();

    ObjcCommon common = new ObjcCommon.Builder(ruleContext)
        .addAssetCatalogs(ruleContext.getPrerequisiteArtifacts("asset_catalogs", Mode.TARGET))
        .addSdkDylibs(ruleContext.attributes().get("sdk_dylibs", Type.STRING_LIST))
        .setCompilationArtifacts(compilationArtifacts)
        .addHdrs(ruleContext.getPrerequisiteArtifacts("hdrs", Mode.TARGET))
        .build();
    common.reportErrors();

    OptionsProvider optionsProvider = new OptionsProvider.Builder()
        .addCopts(ruleContext.getTokenizedStringListAttr("copts"))
        .addInfoplists(ruleContext.getPrerequisiteArtifacts("infoplist", Mode.TARGET))
        .addTransitive(Optional.fromNullable(
            ruleContext.getPrerequisite("options", Mode.TARGET, OptionsProvider.class)))
        .build();

    InfoplistMerging infoplistMerging = new InfoplistMerging.Builder(ruleContext)
        .setIntermediateArtifacts(intermediateArtifacts)
        .setInputPlists(optionsProvider.getInfoplists())
        .setPlmerge(ruleContext.getExecutablePrerequisite("$plmerge", Mode.HOST))
        .build();

    Bundling bundling = new Bundling.Builder()
        .setName(ruleContext.getLabel().getName())
        .setBundleDirSuffix(".bundle")
        .setExtraBundleFiles(ImmutableList.<BundleableFile>of())
        .setObjcProvider(common.getObjcProvider())
        .setInfoplistMerging(infoplistMerging)
        .setIntermediateArtifacts(intermediateArtifacts)
        .build();

    // TODO(bazel-team): Add support to xcodegen for objc_bundle_library targets.
    XcodeProvider xcodeProvider = common.xcodeProvider(
        infoplistMerging.getPlistWithEverything(),
        ImmutableList.<DependencyControl>of(),
        ImmutableList.<XcodeprojBuildSetting>of(),
        optionsProvider.getCopts());

    ObjcProvider nestedBundleProvider = new ObjcProvider.Builder()
        .add(NESTED_BUNDLE, bundling)
        .addTransitive(HEADER, common.getObjcProvider().get(HEADER))
        .addTransitive(INCLUDE, common.getObjcProvider().get(INCLUDE))
        .build();

    registerActions(ruleContext, bundling, common, xcodeProvider, optionsProvider,
        new ExtraLinkArgs("-bundle"), new ExtraActoolArgs());

    return common.configuredTarget(
        NestedSetBuilder.<Artifact>stableOrder()
            .add(ruleContext.getImplicitOutputArtifact(ObjcRuleClasses.PBXPROJ))
            .build(),
        Optional.of(xcodeProvider),
        Optional.of(nestedBundleProvider));
  }
}
