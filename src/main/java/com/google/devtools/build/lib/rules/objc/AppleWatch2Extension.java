// Copyright 2016 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromTemplates;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FLAG;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.Flag.HAS_WATCH2_EXTENSION;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.MERGE_ZIP;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchApplicationBundleRule.WATCH_APP_NAME_ATTR;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multiset;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.objc.WatchUtils.WatchOSVersion;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.syntax.Type;

/** Implementation for {@code apple_watch2_extension}. */
public class AppleWatch2Extension implements RuleConfiguredTargetFactory {

  /** Template for the containing application folder. */
  public static final SafeImplicitOutputsFunction APP_NAME_IPA = fromTemplates("%{app_name}.ipa");

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    validateAttributes(ruleContext);

    ObjcProvider.Builder exposedObjcProviderBuilder = new ObjcProvider.Builder();
    NestedSetBuilder<Artifact> applicationFilesToBuild = NestedSetBuilder.stableOrder();

    // 1. Build watch extension bundle.
    createWatchExtensionBundle(ruleContext);

    // 2. Build watch application bundle, which will contain the extension bundle.
    createWatchApplicationBundle(
        ruleContext,
        watchExtensionIpaArtifact(ruleContext),
        applicationFilesToBuild,
        exposedObjcProviderBuilder);

    RuleConfiguredTargetBuilder targetBuilder =
        ObjcRuleClasses.ruleConfiguredTarget(ruleContext, applicationFilesToBuild.build())
            .addProvider(
                InstrumentedFilesProvider.class,
                InstrumentedFilesCollector.forward(ruleContext, "binary"));

    // 3. Add final watch application artifacts to the ObjcProvider, for bundling the watch
    // application bundle into the final iOS application IPA depending on this rule.
    exposedObjcProviderBuilder.add(MERGE_ZIP, ruleContext.getImplicitOutputArtifact(APP_NAME_IPA));
    WatchUtils.registerActionsToAddWatchSupport(
        ruleContext, exposedObjcProviderBuilder, WatchOSVersion.OS2);
    exposedObjcProviderBuilder.add(FLAG, HAS_WATCH2_EXTENSION);
    targetBuilder.addProvider(ObjcProvider.class, exposedObjcProviderBuilder.build());

    return targetBuilder.build();
  }

  /**
   * Registers actions to create the watch extension bundle.
   *
   * @param ruleContext rule context in which to create the bundle
   */
  private void createWatchExtensionBundle(RuleContext ruleContext) throws InterruptedException {
    new Watch2ExtensionSupport(
            ruleContext,
            ObjcRuleClasses.intermediateArtifacts(ruleContext),
            watchExtensionBundleName(ruleContext))
        .createBundle(watchExtensionIpaArtifact(ruleContext));
  }

  /**
   * Registers actions to create the watch application bundle. This will contain the watch extension
   * bundle. The output artifacts are {@link #APP_NAME_IPA} (which is an implicit output of this
   * rule), and artifacts which are added to {@code exposedObjcProviderBuilder} for consumption by
   * depending targets.
   *
   * @param ruleContext rule context in which to create the bundle
   * @param filesToBuild the list to contain the files to be built for this bundle
   * @param exposedObjcProviderBuilder builder of {@link ObjcProvider} exposed to the parent target;
   *     bundling information will be added to this builder
   */
  private void createWatchApplicationBundle(
      RuleContext ruleContext,
      Artifact extensionIpa,
      NestedSetBuilder<Artifact> filesToBuild,
      ObjcProvider.Builder exposedObjcProviderBuilder)
      throws InterruptedException {
    new WatchApplicationSupport(
            ruleContext,
            WatchOSVersion.OS2,
            // TODO(cparsons): Remove dependency attributes from WatchApplicationSupport,
            // as this is redundant with other attributes.
            ImmutableSet.<Attribute>of(),
            new IntermediateArtifacts(ruleContext, "", watchApplicationBundleName(ruleContext)),
            watchApplicationBundleName(ruleContext),
            watchApplicationIpaArtifact(ruleContext),
            watchApplicationBundleName(ruleContext))
        .createBundle(ImmutableList.of(extensionIpa), filesToBuild, exposedObjcProviderBuilder);
  }

  /** Returns the {@Artifact} containing final watch application bundle. */
  private Artifact watchApplicationIpaArtifact(RuleContext ruleContext)
      throws InterruptedException {
    return ruleContext.getImplicitOutputArtifact(APP_NAME_IPA);
  }

  /** Returns the {@Artifact} containing final watch extension bundle. */
  private Artifact watchExtensionIpaArtifact(RuleContext ruleContext) throws InterruptedException {
    return ruleContext.getImplicitOutputArtifact(ReleaseBundlingSupport.IPA);
  }

  private String watchApplicationBundleName(RuleContext ruleContext) {
    return ruleContext.attributes().get(WATCH_APP_NAME_ATTR, Type.STRING);
  }

  private String watchExtensionBundleName(RuleContext ruleContext) {
    return ruleContext.getLabel().getName();
  }

  private void validateAttributes(RuleContext ruleContext) throws RuleErrorException {
    boolean hasError = false;

    Multiset<Artifact> appResources = HashMultiset.create();
    appResources.addAll(ruleContext.getPrerequisiteArtifacts("app_resources", Mode.TARGET).list());
    appResources.addAll(ruleContext.getPrerequisiteArtifacts("app_strings", Mode.TARGET).list());

    for (Multiset.Entry<Artifact> entry : appResources.entrySet()) {
      if (entry.getCount() > 1) {
        ruleContext.ruleError(
            "The same file was included multiple times in this rule: "
                + entry.getElement().getRootRelativePathString());
        hasError = true;
      }
    }

    Multiset<Artifact> extResources = HashMultiset.create();
    extResources.addAll(ruleContext.getPrerequisiteArtifacts("ext_resources", Mode.TARGET).list());
    extResources.addAll(ruleContext.getPrerequisiteArtifacts("ext_strings", Mode.TARGET).list());

    for (Multiset.Entry<Artifact> entry : extResources.entrySet()) {
      if (entry.getCount() > 1) {
        ruleContext.ruleError(
            "The same file was included multiple times in this rule: "
                + entry.getElement().getRootRelativePathString());
        hasError = true;
      }
    }

    if (hasError) {
      throw new RuleErrorException();
    }
  }
}
