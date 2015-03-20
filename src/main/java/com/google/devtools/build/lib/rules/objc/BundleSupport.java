// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.base.Optional;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.objc.ObjcActionsBuilder.ExtraActoolArgs;
import com.google.devtools.build.lib.rules.objc.XcodeProvider.Builder;

import java.util.Set;

/**
 * Support for generating iOS bundles which contain metadata (a plist file), assets, resources and
 * optionally a binary: registers actions that assemble resources and merge plists, provides data
 * to providers and validates bundle-related attributes.
 *
 * <p>Methods on this class can be called in any order without impacting the result.
 */
final class BundleSupport {
  private final RuleContext ruleContext;
  private final Set<TargetDeviceFamily> targetDeviceFamilies;
  private final ExtraActoolArgs extraActoolArgs;
  private final Bundling bundling;

  /**
   * Returns merging instructions for a bundle's {@code Info.plist}.
   *
   * @param ruleContext context this bundle is constructed in
   * @param objcProvider provider containing all dependencies' information as well as some of this
   *    rule's
   * @param optionsProvider provider containing options and plist settings for this rule and its
   *    dependencies
   */
  static InfoplistMerging infoPlistMerging(RuleContext ruleContext,
      ObjcProvider objcProvider, OptionsProvider optionsProvider) {
    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);
    
    return new InfoplistMerging.Builder(ruleContext)
        .setIntermediateArtifacts(intermediateArtifacts)
        .setInputPlists(NestedSetBuilder.<Artifact>stableOrder()
            .addTransitive(optionsProvider.getInfoplists())
            .addAll(actoolPartialInfoplist(ruleContext, objcProvider).asSet())
            .build())
        .setPlmerge(ruleContext.getExecutablePrerequisite("$plmerge", Mode.HOST))
        .build();
  }

  /**
   * Creates a new bundle support with no special {@code actool} arguments.
   *
   * @param ruleContext context this bundle is constructed in
   * @param targetDeviceFamilies device families used in asset catalogue construction
   * @param bundling bundle information as configured for this rule
   */
  public BundleSupport(
      RuleContext ruleContext, Set<TargetDeviceFamily> targetDeviceFamilies, Bundling bundling) {
    this(ruleContext, targetDeviceFamilies, bundling, new ExtraActoolArgs());
  }

  /**
   * Creates a new bundle support.
   *
   * @param ruleContext context this bundle is constructed in
   * @param targetDeviceFamilies device families used in asset catalogue construction
   * @param bundling bundle information as configured for this rule
   * @param extraActoolArgs any additional parameters to be used for invoking {@code actool}
   */
  public BundleSupport(RuleContext ruleContext, Set<TargetDeviceFamily> targetDeviceFamilies,
      Bundling bundling, ExtraActoolArgs extraActoolArgs) {
    this.ruleContext = ruleContext;
    this.targetDeviceFamilies = targetDeviceFamilies;
    this.extraActoolArgs = extraActoolArgs;
    this.bundling = bundling;
  }

  /**
   * Registers actions required for constructing this bundle, namely merging all involved {@code
   * Info.plist} files and generating asset catalogues.
   *
   * @param objcProvider source of information from this rule's attributes and its dependencies
   *
   * @return this bundle support
   */
  BundleSupport registerActions(ObjcProvider objcProvider) {
    registerMergeInfoplistAction();
    registerActoolActionIfNecessary(objcProvider);

    return this;
  }

  /**
   * Adds any Xcode settings related to this bundle to the given provider builder.
   *
   * @return this bundle support
   */
  BundleSupport addXcodeSettings(Builder xcodeProviderBuilder) {
    xcodeProviderBuilder.setInfoplistMerging(bundling.getInfoplistMerging());
    return this;
  }

  private void registerMergeInfoplistAction() {
    // TODO(bazel-team): Move action implementation from InfoplistMerging to this class.
    ruleContext.registerAction(bundling.getInfoplistMerging().getMergeAction());
  }

  private void registerActoolActionIfNecessary(ObjcProvider objcProvider) {
    Optional<Artifact> actoolzipOutput = bundling.getActoolzipOutput();
    if (!actoolzipOutput.isPresent()) {
      return;
    }

    ObjcActionsBuilder actionsBuilder = ObjcRuleClasses.actionsBuilder(ruleContext);

    Artifact actoolPartialInfoplist = actoolPartialInfoplist(ruleContext, objcProvider).get();
    actionsBuilder.registerActoolzipAction(
        new ObjcRuleClasses.Tools(ruleContext),
        objcProvider,
        actoolzipOutput.get(),
        actoolPartialInfoplist,
        extraActoolArgs,
        targetDeviceFamilies);
  }

  /**
   * Returns the artifact that is a plist file generated by an invocation of {@code actool} or
   * {@link Optional#absent()} if no asset catalogues are present in this target and its
   * dependencies.
   *
   * <p>All invocations of {@code actool} generate this kind of plist file, which contains metadata
   * about the {@code app_icon} and {@code launch_image} if supplied. If neither an app icon or a
   * launch image was supplied, the plist file generated is empty.
   */
  private static Optional<Artifact> actoolPartialInfoplist(
      RuleContext ruleContext, ObjcProvider objcProvider) {
    if (objcProvider.hasAssetCatalogs()) {
      IntermediateArtifacts intermediateArtifacts =
          ObjcRuleClasses.intermediateArtifacts(ruleContext);
      return Optional.of(intermediateArtifacts.actoolPartialInfoplist());
    } else {
      return Optional.absent();
    }
  }

}
