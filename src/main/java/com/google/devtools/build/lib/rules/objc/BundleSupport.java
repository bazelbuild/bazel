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

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.ASSET_CATALOG;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.STRINGS;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCASSETS_DIR;

import com.google.common.base.Optional;
import com.google.common.base.Verify;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.objc.ObjcActionsBuilder.ExtraActoolArgs;
import com.google.devtools.build.lib.rules.objc.XcodeProvider.Builder;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Support for generating iOS bundles which contain metadata (a plist file), assets, resources and
 * optionally a binary: registers actions that assemble resources and merge plists, provides data
 * to providers and validates bundle-related attributes.
 *
 * <p>Methods on this class can be called in any order without impacting the result.
 */
final class BundleSupport {

  static class ExtraMergePlists extends IterableWrapper<Artifact> {
    ExtraMergePlists(Artifact... inputs) {
      super(inputs);
    }
  }

  private final RuleContext ruleContext;
  private final Set<TargetDeviceFamily> targetDeviceFamilies;
  private final ExtraActoolArgs extraActoolArgs;
  private final Bundling bundling;
  private final Attributes attributes;

  /**
   * Returns merging instructions for a bundle's {@code Info.plist}.
   *
   * @param ruleContext context this bundle is constructed in
   * @param objcProvider provider containing all dependencies' information as well as some of this
   *    rule's
   * @param optionsProvider provider containing options and plist settings for this rule and its
   *    dependencies
   * @param primaryBundleId used to set the bundle identifier or override the existing one from
   *     plist file, can be null
   * @param fallbackBundleId used to set the bundle identifier if it is not set by plist file or
   *     primary identifier, can be null
   * @param extraMergePlists additional plist files to merge
   */
  static InfoplistMerging infoPlistMerging(
      RuleContext ruleContext,
      ObjcProvider objcProvider,
      OptionsProvider optionsProvider,
      String primaryBundleId,
      String fallbackBundleId,
      ExtraMergePlists extraMergePlists) {
    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);

    return new InfoplistMerging.Builder(ruleContext)
        .setIntermediateArtifacts(intermediateArtifacts)
        .setInputPlists(NestedSetBuilder.<Artifact>stableOrder()
            .addTransitive(optionsProvider.getInfoplists())
            .addAll(actoolPartialInfoplist(ruleContext, objcProvider).asSet())
            .addAll(extraMergePlists)
            .build())
        .setPlmerge(ruleContext.getExecutablePrerequisite("$plmerge", Mode.HOST))
        .setBundleIdentifiers(primaryBundleId, fallbackBundleId)
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
    this.attributes = new Attributes(ruleContext);
  }

  /**
   * Registers actions required for constructing this bundle, namely merging all involved {@code
   * Info.plist} files and generating asset catalogues.
   *
   * @param objcProvider source of information from this rule's attributes and its dependencies
   * @return this bundle support
   */
  BundleSupport registerActions(ObjcProvider objcProvider) {
    registerConvertStringsActions(objcProvider);
    registerConvertXibsActions(objcProvider);
    registerMomczipActions(objcProvider);
    registerInterfaceBuilderActions(objcProvider);
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

  /**
   * Validates that resources defined in this rule and its dependencies and written to this bundle
   * are legal (for example that they are not mapped to the same bundle location).
   *
   * @return this bundle support
   */
  BundleSupport validateResources(ObjcProvider objcProvider) {
    Map<String, Artifact> bundlePathToFile = new HashMap<>();
    for (Artifact stringsFile : objcProvider.get(STRINGS)) {
      String bundlePath = BundleableFile.flatBundlePath(stringsFile.getExecPath());

      // Normally any two resources mapped to the same path in the bundle are illegal. However, we
      // currently don't have a good solution for resources generated by a genrule in
      // multi-architecture builds: They map to the same bundle path but have different owners (the
      // genrules targets in the various configurations) and roots (one for each architecture).
      // Since we know that architecture shouldn't matter for strings file generation we silently
      // ignore cases like this and pick one of the outputs at random to put in the bundle (see also
      // related filtering code in Bundling.Builder.build()).
      if (bundlePathToFile.containsKey(bundlePath)) {
        Artifact original = bundlePathToFile.get(bundlePath);
        if (!Objects.equals(original.getOwner(), stringsFile.getOwner())) {
          ruleContext.ruleError(String.format(
              "Two string files map to the same path [%s] in this bundle but come from different "
                  + "locations: %s and %s",
              bundlePath, original.getOwner(), stringsFile.getOwner()));
        } else {
          Verify.verify(!original.getRoot().equals(stringsFile.getRoot()),
              "%s and %s should have different roots but have %s and %s",
              original, stringsFile, original.getRoot(), stringsFile.getRoot());
        }

      } else {
        bundlePathToFile.put(bundlePath, stringsFile);
      }
    }

    // TODO(bazel-team): Do the same validation for storyboards and datamodels which could also be
    // generated by genrules or doubly defined.

    return this;
  }

  private void registerInterfaceBuilderActions(ObjcProvider objcProvider) {
    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);
    for (Artifact storyboardInput : objcProvider.get(ObjcProvider.STORYBOARD)) {
      String archiveRoot = BundleableFile.flatBundlePath(storyboardInput.getExecPath()) + "c";
      Artifact zipOutput = intermediateArtifacts.compiledStoryboardZip(storyboardInput);

      ruleContext.registerAction(
          ObjcActionsBuilder.spawnJavaOnDarwinActionBuilder(attributes.ibtoolzipDeployJar())
              .setMnemonic("StoryboardCompile")
              .setCommandLine(CustomCommandLine.builder()
                  // The next three arguments are positional,
                  // i.e. they don't have flags before them.
                  .addPath(zipOutput.getExecPath())
                  .add(archiveRoot)
                  .addPath(ObjcActionsBuilder.IBTOOL)

                  .add("--minimum-deployment-target").add(bundling.getMinimumOsVersion())
                  .addPath(storyboardInput.getExecPath())
                  .build())
              .addOutput(zipOutput)
              .addInput(storyboardInput)
              .build(ruleContext));
    }
  }

  private void registerMomczipActions(ObjcProvider objcProvider) {
    ObjcConfiguration objcConfiguration = ObjcRuleClasses.objcConfiguration(ruleContext);
    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);
    Iterable<Xcdatamodel> xcdatamodels = Xcdatamodels.xcdatamodels(
        intermediateArtifacts, objcProvider.get(ObjcProvider.XCDATAMODEL));
    for (Xcdatamodel datamodel : xcdatamodels) {
      Artifact outputZip = datamodel.getOutputZip();
      ruleContext.registerAction(
          ObjcActionsBuilder.spawnJavaOnDarwinActionBuilder(attributes.momczipDeployJar())
              .setMnemonic("MomCompile")
              .addOutput(outputZip)
              .addInputs(datamodel.getInputs())
              .setCommandLine(CustomCommandLine.builder()
                  .addPath(outputZip.getExecPath())
                  .add(datamodel.archiveRootForMomczip())
                  .add(IosSdkCommands.MOMC_PATH)

                  .add("-XD_MOMC_SDKROOT=" + IosSdkCommands.sdkDir(objcConfiguration))
                  .add("-XD_MOMC_IOS_TARGET_VERSION=" + bundling.getMinimumOsVersion())
                  .add("-MOMC_PLATFORMS")
                  .add(objcConfiguration.getBundlingPlatform().getLowerCaseNameInPlist())
                  .add("-XD_MOMC_TARGET_VERSION=10.6")
                  .add(datamodel.getContainer().getSafePathString())
                  .build())
              .build(ruleContext));
    }
  }

  private void registerConvertXibsActions(ObjcProvider objcProvider) {
    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);
    for (Artifact original : objcProvider.get(ObjcProvider.XIB)) {
      Artifact zipOutput = intermediateArtifacts.compiledXibFileZip(original);
      String archiveRoot = BundleableFile.flatBundlePath(
          FileSystemUtils.replaceExtension(original.getExecPath(), ".nib"));
      ruleContext.registerAction(
          ObjcActionsBuilder.spawnJavaOnDarwinActionBuilder(attributes.ibtoolzipDeployJar())
              .setMnemonic("XibCompile")
              .setCommandLine(CustomCommandLine.builder()
                  // The next three arguments are positional,
                  // i.e. they don't have flags before them.
                  .addPath(zipOutput.getExecPath())
                  .add(archiveRoot)
                  .addPath(ObjcActionsBuilder.IBTOOL)

                  .add("--minimum-deployment-target").add(bundling.getMinimumOsVersion())
                  .addPath(original.getExecPath())
                  .build())
              .addOutput(zipOutput)
              .addInput(original)
              .build(ruleContext));
    }
  }

  private void registerConvertStringsActions(ObjcProvider objcProvider) {
    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);
    for (Artifact strings : objcProvider.get(ObjcProvider.STRINGS)) {
      Artifact bundled = intermediateArtifacts.convertedStringsFile(strings);
      ruleContext.registerAction(new SpawnAction.Builder()
          .setMnemonic("ConvertStringsPlist")
          .setExecutable(attributes.plmerge())
          .setCommandLine(CustomCommandLine.builder()
              .addExecPath("--source_file", strings)
              .addExecPath("--out_file", bundled)
              .build())
          .addInput(strings)
          .addOutput(bundled)
          .build(ruleContext));
    }
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

    Artifact actoolPartialInfoplist = actoolPartialInfoplist(ruleContext, objcProvider).get();
    Artifact zipOutput = actoolzipOutput.get();

    // TODO(bazel-team): Do not use the deploy jar explicitly here. There is currently a bug where
    // we cannot .setExecutable({java_binary target}) and set REQUIRES_DARWIN in the execution info.
    // Note that below we set the archive root to the empty string. This means that the generated
    // zip file will be rooted at the bundle root, and we have to prepend the bundle root to each
    // entry when merging it with the final .ipa file.
    ruleContext.registerAction(
        ObjcActionsBuilder.spawnJavaOnDarwinActionBuilder(attributes.actoolzipDeployJar())
            .setMnemonic("AssetCatalogCompile")
            .addTransitiveInputs(objcProvider.get(ASSET_CATALOG))
            .addOutput(zipOutput)
            .addOutput(actoolPartialInfoplist)
            .setCommandLine(actoolzipCommandLine(
                objcProvider,
                zipOutput,
                actoolPartialInfoplist))
            .build(ruleContext));
  }

  private CommandLine actoolzipCommandLine(
      final ObjcProvider provider,
      final Artifact zipOutput,
      final Artifact partialInfoPlist) {
    ObjcConfiguration objcConfiguration = ObjcRuleClasses.objcConfiguration(ruleContext);
    CustomCommandLine.Builder commandLine = CustomCommandLine.builder()
        // The next three arguments are positional, i.e. they don't have flags before them.
        .addPath(zipOutput.getExecPath())
        .add("") // archive root
        .add(IosSdkCommands.ACTOOL_PATH)

        .add("--platform").add(objcConfiguration.getBundlingPlatform().getLowerCaseNameInPlist())
        .addExecPath("--output-partial-info-plist", partialInfoPlist)
        .add("--minimum-deployment-target").add(bundling.getMinimumOsVersion());

    for (TargetDeviceFamily targetDeviceFamily : targetDeviceFamilies) {
      commandLine.add("--target-device").add(targetDeviceFamily.name().toLowerCase(Locale.US));
    }

    return commandLine
        .add(PathFragment.safePathStrings(provider.get(XCASSETS_DIR)))
        .add(extraActoolArgs)
        .build();
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

  /**
   * Common rule attributes used by a bundle support.
   */
  private static class Attributes {
    private final RuleContext ruleContext;

    private Attributes(RuleContext ruleContext) {
      this.ruleContext = ruleContext;
    }

    /**
     * Returns a reference to the plmerge executable.
     */
    FilesToRunProvider plmerge() {
      return ruleContext.getExecutablePrerequisite("$plmerge", Mode.HOST);
    }

    /**
     * Returns the location of the ibtoolzip deploy jar.
     */
    Artifact ibtoolzipDeployJar() {
      return ruleContext.getPrerequisiteArtifact("$ibtoolzip_deploy", Mode.HOST);
    }

    /**
     * Returns the location of the momczip deploy jar.
     */
    Artifact momczipDeployJar() {
      return ruleContext.getPrerequisiteArtifact("$momczip_deploy", Mode.HOST);
    }

    /**
     * Returns the location of the actoolzip deploy jar.
     */
    Artifact actoolzipDeployJar() {
      return ruleContext.getPrerequisiteArtifact("$actoolzip_deploy", Mode.HOST);
    }
  }
}
