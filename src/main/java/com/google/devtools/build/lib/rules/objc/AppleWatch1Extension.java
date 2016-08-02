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

import static com.google.devtools.build.lib.rules.objc.AppleWatch1ExtensionRule.WATCH_APP_DEPS_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FLAG;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.Flag.HAS_WATCH1_EXTENSION;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.MERGE_ZIP;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchApplicationBundleRule.WATCH_APP_NAME_ATTR;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Attribute.SplitTransition;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.objc.IosExtension.ExtensionSplitArchTransition;
import com.google.devtools.build.lib.rules.objc.WatchUtils.WatchOSVersion;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.syntax.Type;

/**
 * Implementation for {@code apple_watch1_extension}.
 */
public class AppleWatch1Extension implements RuleConfiguredTargetFactory {

  static final SplitTransition<BuildOptions> MINIMUM_OS_AND_SPLIT_ARCH_TRANSITION =
      new ExtensionSplitArchTransition(WatchUtils.MINIMUM_OS_VERSION,
          ConfigurationDistinguisher.WATCH_OS1_EXTENSION);
  private static final ImmutableSet<Attribute> extensionDependencyAttributes =
      ImmutableSet.of(new Attribute("binary", Mode.SPLIT));
  private static final ImmutableSet<Attribute> applicationDependencyAttributes =
      ImmutableSet.of(new Attribute(WATCH_APP_DEPS_ATTR, Mode.SPLIT));

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    ObjcProvider.Builder extensionObjcProviderBuilder = new ObjcProvider.Builder();
    ObjcProvider.Builder exposedObjcProviderBuilder = new ObjcProvider.Builder();
    XcodeProvider.Builder applicationXcodeProviderBuilder = new XcodeProvider.Builder();
    XcodeProvider.Builder extensionXcodeProviderBuilder = new XcodeProvider.Builder();
    NestedSetBuilder<Artifact> applicationFilesToBuild = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Artifact> extensionfilesToBuild = NestedSetBuilder.stableOrder();

    // 1. Build watch application bundle.
    createWatchApplicationBundle(
        ruleContext,
        applicationXcodeProviderBuilder,
        applicationFilesToBuild,
        exposedObjcProviderBuilder);

    // 2. Build watch extension bundle.
    createWatchExtensionBundle(ruleContext, extensionXcodeProviderBuilder,
        applicationXcodeProviderBuilder, extensionObjcProviderBuilder, extensionfilesToBuild);

    // 3. Extract the watch application bundle into the extension bundle.
    registerWatchApplicationUnBundlingAction(ruleContext);

    RuleConfiguredTargetBuilder targetBuilder =
        ObjcRuleClasses.ruleConfiguredTarget(ruleContext, extensionfilesToBuild.build())
            .addProvider(XcodeProvider.class, extensionXcodeProviderBuilder.build())
            .addProvider(
                InstrumentedFilesProvider.class,
                InstrumentedFilesCollector.forward(ruleContext, "binary"));

    // 4. Exposed {@ObjcProvider} for bundling into final IPA.
    exposeObjcProvider(ruleContext, targetBuilder, exposedObjcProviderBuilder);

    return targetBuilder.build();
  }

  /**
   * Exposes an {@link ObjcProvider} with the following to create the final IPA:
   *    1. Watch extension bundle.
   *    2. WatchKitSupport.
   *    3. A flag to indicate that watch os 1 extension is included.
   */
  private void exposeObjcProvider(
      RuleContext ruleContext,
      RuleConfiguredTargetBuilder targetBuilder,
      ObjcProvider.Builder exposedObjcProviderBuilder)
      throws InterruptedException {
    exposedObjcProviderBuilder.add(MERGE_ZIP,
        ruleContext.getImplicitOutputArtifact(ReleaseBundlingSupport.IPA));
    WatchUtils.registerActionsToAddWatchSupport(ruleContext, exposedObjcProviderBuilder,
        WatchOSVersion.OS1);
    exposedObjcProviderBuilder.add(FLAG, HAS_WATCH1_EXTENSION);

    targetBuilder.addProvider(ObjcProvider.class, exposedObjcProviderBuilder.build());
  }

  /**
   * Creates a watch extension bundle.
   *
   * @param ruleContext rule context in which to create the bundle
   * @param extensionXcodeProviderBuilder {@link XcodeProvider.Builder} for the extension
   * @param applicationXcodeProviderBuilder {@link XcodeProvider.Builder} for the watch application
   *    which is added as a dependency to the extension
   * @param objcProviderBuilder {@link ObjcProvider.Builder} for the extension
   * @param filesToBuild the list to contain the files to be built for this extension bundle
   */
  private void createWatchExtensionBundle(RuleContext ruleContext,
      XcodeProvider.Builder extensionXcodeProviderBuilder,
      XcodeProvider.Builder applicationXcodeProviderBuilder,
      ObjcProvider.Builder objcProviderBuilder,
      NestedSetBuilder<Artifact> filesToBuild) throws InterruptedException {
    new WatchExtensionSupport(ruleContext,
        WatchOSVersion.OS1,
        extensionDependencyAttributes,
        ObjcRuleClasses.intermediateArtifacts(ruleContext),
        watchExtensionBundleName(ruleContext),
        watchExtensionIpaArtifact(ruleContext),
        watchApplicationBundle(ruleContext),
        applicationXcodeProviderBuilder.build(),
        ConfigurationDistinguisher.WATCH_OS1_EXTENSION)
    .createBundle(filesToBuild, objcProviderBuilder, extensionXcodeProviderBuilder);
  }

  /**
   * Creates a watch application bundle.
   * @param ruleContext rule context in which to create the bundle
   * @param xcodeProviderBuilder {@link XcodeProvider.Builder} for the application
   * @param filesToBuild the list to contain the files to be built for this bundle
   * @param exposedObjcProviderBuilder {@link ObjcProvider.Builder} exposed to the parent target
   */
  private void createWatchApplicationBundle(
      RuleContext ruleContext,
      XcodeProvider.Builder xcodeProviderBuilder,
      NestedSetBuilder<Artifact> filesToBuild,
      ObjcProvider.Builder exposedObjcProviderBuilder)
      throws InterruptedException {
    new WatchApplicationSupport(
            ruleContext,
            WatchOSVersion.OS1,
            applicationDependencyAttributes,
            new IntermediateArtifacts(ruleContext, "", watchApplicationBundleName(ruleContext)),
            watchApplicationBundleName(ruleContext),
            watchApplicationIpaArtifact(ruleContext),
            watchApplicationBundleName(ruleContext))
        .createBundleAndXcodeproj(
            xcodeProviderBuilder,
            ImmutableList.<Artifact>of(),
            filesToBuild,
            exposedObjcProviderBuilder);
  }

  /**
   * Registers action to extract the watch application ipa (after signing if required) to the
   * extension bundle.
   *
   * For example, TestWatchApp.ipa will be unbundled into,
   *   PlugIns/TestWatchExtension.appex
   *   PlugIns/TestWatchExtension.appex/TestWatchApp.app
   */
  private void registerWatchApplicationUnBundlingAction(RuleContext ruleContext) {
    Artifact watchApplicationIpa = watchApplicationIpaArtifact(ruleContext);
    Artifact watchApplicationBundle = watchApplicationBundle(ruleContext);

    String workingDirectory = watchApplicationBundle.getExecPathString().substring(0,
        watchApplicationBundle.getExecPathString().lastIndexOf('/'));

    ImmutableList<String> command = ImmutableList.of(
        "mkdir -p " + workingDirectory,
        "&&",
        String.format("/usr/bin/unzip -q -o %s -d %s",
            watchApplicationIpa.getExecPathString(),
            workingDirectory),
        "&&",
        String.format("cd %s/Payload", workingDirectory),
        "&&",
        String.format("/usr/bin/zip -q -r -0 ../%s *", watchApplicationBundle.getFilename()));
    ruleContext.registerAction(
        ObjcRuleClasses.spawnOnDarwinActionBuilder()
            .setProgressMessage("Extracting watch app: " + ruleContext.getLabel())
            .setShellCommand(ImmutableList.of("/bin/bash", "-c", Joiner.on(" ").join(command)))
            .addInput(watchApplicationIpa)
            .addOutput(watchApplicationBundle)
            .build(ruleContext));
  }

  /**
   * Returns a zip {@Artifact} containing extracted watch application - "TestWatchApp.app"
   * which is to be merged into the extension bundle.
   */
  private Artifact watchApplicationBundle(RuleContext ruleContext) {
    return ruleContext.getRelatedArtifact(ruleContext.getUniqueDirectory(
        "_watch"), String.format("/%s", watchApplicationIpaArtifact(ruleContext)
            .getFilename().replace(".ipa", ".zip")));
  }

  /**
   * Returns the {@Artifact} containing final watch application bundle.
   */
  private Artifact watchApplicationIpaArtifact(RuleContext ruleContext) {
    return ruleContext.getRelatedArtifact(ruleContext.getUniqueDirectory("_watch"),
          String.format("/%s.ipa", watchApplicationBundleName(ruleContext)));
  }

  /**
   * Returns the {@Artifact} containing final watch extension bundle.
   */
  private Artifact watchExtensionIpaArtifact(RuleContext ruleContext) throws InterruptedException {
    return ruleContext.getImplicitOutputArtifact(ReleaseBundlingSupport.IPA);
  }

  private String watchApplicationBundleName(RuleContext ruleContext) {
    return ruleContext.attributes().get(WATCH_APP_NAME_ATTR, Type.STRING);
  }

  private String watchExtensionBundleName(RuleContext ruleContext) {
    return ruleContext.getLabel().getName();
  }
}
