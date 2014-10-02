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
import static com.google.devtools.build.lib.rules.objc.XcodeProductType.BUNDLE;

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.objc.ObjcActionsBuilder.ExtraActoolArgs;
import com.google.devtools.build.lib.rules.objc.ObjcActionsBuilder.ExtraLinkArgs;
import com.google.devtools.build.lib.rules.objc.ObjcLibrary.InfoplistsFromRule;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.DependencyControl;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.TargetControl;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.XcodeprojBuildSetting;

/**
 * Implementation for {@code objc_bundle_library}.
 */
public class ObjcBundleLibrary implements RuleConfiguredTargetFactory {
  static Iterable<DependencyControl> targetDependenciesTransitive(
      Iterable<XcodeProvider> depXcodeProviders) {
    ImmutableSet.Builder<DependencyControl> result = new ImmutableSet.Builder<>();
    for (XcodeProvider provider : depXcodeProviders) {
      for (TargetControl targetDependency : provider.getTargets()) {
        // Only add a target to a binary's dependencies if it has source files to compile. Xcode
        // cannot build targets without a source file in the PBXSourceFilesBuildPhase, so if such a
        // target is present in the control file, it is only to get Xcodegen to put headers and
        // resources not used by the final binary in the Project Navigator.
        if (!targetDependency.getSourceFileList().isEmpty()
            || !targetDependency.getNonArcSourceFileList().isEmpty()) {
          result.add(DependencyControl.newBuilder()
              .setTargetLabel(targetDependency.getLabel())
              .build());
        }
      }
    }
    return result.build();
  }

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

  static Bundling bundling(RuleContext ruleContext, String bundleDirSuffix,
      Iterable<BundleableFile> extraBundleFiles, ObjcProvider objcProvider,
      OptionsProvider optionsProvider) {
    IntermediateArtifacts intermediateArtifacts = ObjcLibrary.intermediateArtifacts(ruleContext);
    InfoplistMerging infoplistMerging = new InfoplistMerging.Builder(ruleContext)
        .setIntermediateArtifacts(intermediateArtifacts)
        .setInputPlists(optionsProvider.getInfoplists())
        .setPlmerge(ruleContext.getExecutablePrerequisite("$plmerge", Mode.HOST))
        .build();
    return new Bundling.Builder()
        .setName(ruleContext.getLabel().getName())
        .setBundleDirSuffix(bundleDirSuffix)
        .setExtraBundleFiles(ImmutableList.copyOf(extraBundleFiles))
        .setObjcProvider(objcProvider)
        .setInfoplistMerging(infoplistMerging)
        .setIntermediateArtifacts(intermediateArtifacts)
        .build();
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    ObjcCommon common = ObjcLibrary.common(ruleContext, ImmutableList.<SdkFramework>of());
    OptionsProvider optionsProvider = ObjcLibrary.optionsProvider(ruleContext,
        new InfoplistsFromRule(ruleContext.getPrerequisiteArtifacts("infoplist", Mode.TARGET)));
    Bundling bundling = bundling(ruleContext, ".bundle", ImmutableList.<BundleableFile>of(),
        common.getObjcProvider(), optionsProvider);

    Iterable<XcodeProvider> depXcodeProviders =
        ruleContext.getPrerequisites("deps", Mode.TARGET, XcodeProvider.class);
    XcodeProvider xcodeProvider = common.xcodeProvider(
        bundling.getInfoplistMerging().getPlistWithEverything(),
        targetDependenciesTransitive(depXcodeProviders),
        ImmutableList.<XcodeprojBuildSetting>of(),
        optionsProvider.getCopts(),
        BUNDLE,
        depXcodeProviders);

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
