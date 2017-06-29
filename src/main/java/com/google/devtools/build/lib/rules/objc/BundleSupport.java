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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.ASSET_CATALOG;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.BUNDLE_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.STRINGS;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCASSETS_DIR;

import com.google.common.base.Optional;
import com.google.common.base.Verify;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.BinaryFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.apple.Platform;
import com.google.devtools.build.lib.rules.apple.Platform.PlatformType;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Support for generating iOS bundles which contain metadata (a plist file), assets, resources and
 * optionally a binary: registers actions that assemble resources and merge plists, provides data
 * to providers and validates bundle-related attributes.
 *
 * <p>Methods on this class can be called in any order without impacting the result.
 */
final class BundleSupport {

  /**
   * Iterable wrapper used to strongly type arguments eventually passed to {@code actool}.
   */
  static final class ExtraActoolArgs extends IterableWrapper<String> {
    ExtraActoolArgs(Iterable<String> args) {
      super(args);
    }

    ExtraActoolArgs(String... args) {
      super(args);
    }
  }

  // TODO(cparsons): Take restricted interfaces of RuleContext instead of RuleContext (such as
  // RuleErrorConsumer).
  private final RuleContext ruleContext;
  private final AppleConfiguration appleConfiguration;
  private final Platform platform;
  private final ExtraActoolArgs extraActoolArgs;
  private final Bundling bundling;
  private final Attributes attributes;

  /**
   * Creates a new bundle support.
   *
   * @param ruleContext context this bundle is constructed in
   * @param appleConfiguration the configuration this bundle is constructed in
   * @param platform the platform this bundle is built for
   * @param bundling bundle information as configured for this rule
   * @param extraActoolArgs any additional parameters to be used for invoking {@code actool}
   */
  public BundleSupport(RuleContext ruleContext, AppleConfiguration appleConfiguration,
      Platform platform, Bundling bundling, ExtraActoolArgs extraActoolArgs) {
    this.ruleContext = ruleContext;
    this.appleConfiguration = appleConfiguration;
    this.platform = platform;
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
    registerActoolActionIfNecessary(objcProvider);

    if (bundling.needsToMergeInfoplist()) {
      NestedSet<Artifact> mergingContentArtifacts = bundling.getMergingContentArtifacts();
      Artifact mergedPlist = bundling.getBundleInfoplist().get();
      registerMergeInfoplistAction(
          mergingContentArtifacts, PlMergeControlBytes.fromBundling(bundling, mergedPlist));
    }
    return this;
  }

  /**
   * Validates the platform for this build is either simulator or device, and does not
   * contain architectures for both platforms.
   *
   * @return this bundle support
   */
  public BundleSupport validatePlatform() {
    Platform platform = null;
    for (String architecture : appleConfiguration.getIosMultiCpus()) {
      if (platform == null) {
        platform = Platform.forTarget(PlatformType.IOS, architecture);
      } else if (platform != Platform.forTarget(PlatformType.IOS, architecture)) {
        ruleContext.ruleError(
            String.format("In builds which require bundling, --ios_multi_cpus does not currently "
                + "allow values for both simulator and device builds. Flag was %s",
                appleConfiguration.getIosMultiCpus()));
      }
    }
    return this;
  }

  /**
   * Validates that resources defined in this rule and its dependencies and written to this
   * bundle are legal (for example that they are not mapped to the same bundle location).
   *
   * @return this bundle support
   */
  public BundleSupport validateResources(ObjcProvider objcProvider) {
    Map<String, Artifact> bundlePathToFile = new HashMap<>();
    NestedSet<Artifact> artifacts = objcProvider.get(STRINGS);

    Iterable<BundleableFile> bundleFiles =
        Iterables.concat(
            objcProvider.get(BUNDLE_FILE), BundleableFile.flattenedRawResourceFiles(artifacts));
    for (BundleableFile bundleFile : bundleFiles) {
      String bundlePath = bundleFile.getBundlePath();
      Artifact bundled = bundleFile.getBundled();

      // Normally any two resources mapped to the same path in the bundle are illegal. However, we
      // currently don't have a good solution for resources generated by a genrule in
      // multi-architecture builds: They map to the same bundle path but have different owners (the
      // genrules targets in the various configurations) and roots (one for each architecture).
      // Since we know that architecture shouldn't matter for strings file generation we silently
      // ignore cases like this and pick one of the outputs at random to put in the bundle (see also
      // related filtering code in Bundling.Builder.build()).
      if (bundlePathToFile.containsKey(bundlePath)) {
        Artifact original = bundlePathToFile.get(bundlePath);
        if (!Objects.equals(original.getOwner(), bundled.getOwner())) {
          ruleContext.ruleError(
              String.format(
                  "Two files map to the same path [%s] in this bundle but come from different "
                      + "locations: %s and %s",
                  bundlePath,
                  original.getOwner(),
                  bundled.getOwner()));
        } else {
          Verify.verify(
              !original.getRoot().equals(bundled.getRoot()),
              "%s and %s should have different roots but have %s and %s",
              original,
              bundleFile,
              original.getRoot(),
              bundled.getRoot());
        }

      } else {
        bundlePathToFile.put(bundlePath, bundled);
      }
    }
    // TODO(bazel-team): Do the same validation for storyboards and datamodels which could also be
    // generated by genrules or doubly defined.

    return this;
  }

  /**
   * Returns a set containing the {@link TargetDeviceFamily} values which this bundle is targeting.
   * Returns an empty set for any invalid value of the target device families attribute.
   */
  ImmutableSet<TargetDeviceFamily> targetDeviceFamilies() {
    return bundling.getTargetDeviceFamilies();
  }
 
  /**
   * Returns true if this bundle is targeted to {@link TargetDeviceFamily#WATCH}, false otherwise.
   */
  boolean isBuildingForWatch() {
    return targetDeviceFamilies()
        .stream()
        .anyMatch(
            targetDeviceFamily ->
                targetDeviceFamily
                    .name()
                    .equalsIgnoreCase(TargetDeviceFamily.WATCH.getNameInRule()));
  }

  /**
   * Returns a set containing the {@link TargetDeviceFamily} values the resources in this bundle
   * are targeting. When watch is included as one of the families, (for example [iphone, watch] for
   * simulator builds, assets should always be compiled for {@link TargetDeviceFamily#WATCH}.
   */
  private ImmutableSet<TargetDeviceFamily> targetDeviceFamiliesForResources() {
    if (isBuildingForWatch()) {
      return ImmutableSet.of(TargetDeviceFamily.WATCH);
    } else {
      return targetDeviceFamilies();
    }
  }

  private void registerInterfaceBuilderActions(ObjcProvider objcProvider) {
    for (Artifact storyboardInput : objcProvider.get(ObjcProvider.STORYBOARD)) {
      String archiveRoot = storyboardArchiveRoot(storyboardInput);
      Artifact zipOutput = bundling.getIntermediateArtifacts()
          .compiledStoryboardZip(storyboardInput);

      ruleContext.registerAction(
          ObjcRuleClasses.spawnAppleEnvActionBuilder(appleConfiguration, platform)
              .setMnemonic("StoryboardCompile")
              .setExecutable(attributes.ibtoolWrapper())
              .setCommandLine(ibActionsCommandLine(archiveRoot, zipOutput, storyboardInput))
              .addOutput(zipOutput)
              .addInput(storyboardInput)
              .build(ruleContext));
    }
  }

  /**
   * Returns the root file path to which storyboard interfaces are compiled.
   */
  protected String storyboardArchiveRoot(Artifact storyboardInput) {
    // When storyboards are compiled for {@link TargetDeviceFamily#WATCH}, return the containing
    // directory if it ends with .lproj to account for localization or "." representing the bundle
    // root otherwise. Examples: Payload/Foo.app/Base.lproj/<compiled_file>,
    // Payload/Foo.app/<compile_file_1>
    if (isBuildingForWatch()) {
      String containingDir = storyboardInput.getExecPath().getParentDirectory().getBaseName();
      return containingDir.endsWith(".lproj") ? (containingDir + "/") : ".";
    } else {
      return BundleableFile.flatBundlePath(storyboardInput.getExecPath()) + "c";
    }
  }

  private CommandLine ibActionsCommandLine(String archiveRoot, Artifact zipOutput,
      Artifact storyboardInput) {
    CustomCommandLine.Builder commandLine =
        CustomCommandLine.builder()
            // The next three arguments are positional, i.e. they don't have flags before them.
            .addPath(zipOutput.getExecPath())
            .add(archiveRoot)
            .add("--minimum-deployment-target")
            .add(bundling.getMinimumOsVersion().toString())
            .add("--module")
            .add(ruleContext.getLabel().getName());

    for (TargetDeviceFamily targetDeviceFamily : targetDeviceFamiliesForResources()) {
      commandLine.add("--target-device").add(targetDeviceFamily.name().toLowerCase(Locale.US));
    }

    return commandLine
        .addPath(storyboardInput.getExecPath())
        .build();
  }

  private void registerMomczipActions(ObjcProvider objcProvider) {
    Iterable<Xcdatamodel> xcdatamodels = Xcdatamodels.xcdatamodels(
        bundling.getIntermediateArtifacts(), objcProvider.get(ObjcProvider.XCDATAMODEL));
    for (Xcdatamodel datamodel : xcdatamodels) {
      Artifact outputZip = datamodel.getOutputZip();
      ruleContext.registerAction(
          ObjcRuleClasses.spawnAppleEnvActionBuilder(appleConfiguration, platform)
              .setMnemonic("MomCompile")
              .setExecutable(attributes.momcWrapper())
              .addOutput(outputZip)
              .addInputs(datamodel.getInputs())
              .setCommandLine(CustomCommandLine.builder()
                  .addPath(outputZip.getExecPath())
                  .add(datamodel.archiveRootForMomczip())
                  .add("-XD_MOMC_SDKROOT=" + AppleToolchain.sdkDir())
                  .add("-XD_MOMC_IOS_TARGET_VERSION=" + bundling.getMinimumOsVersion())
                  .add("-MOMC_PLATFORMS")
                  .add(appleConfiguration.getMultiArchPlatform(PlatformType.IOS)
                      .getLowerCaseNameInPlist())
                  .add("-XD_MOMC_TARGET_VERSION=10.6")
                  .add(datamodel.getContainer().getSafePathString())
                  .build())
              .build(ruleContext));
    }
  }

  private void registerConvertXibsActions(ObjcProvider objcProvider) {
    for (Artifact original : objcProvider.get(ObjcProvider.XIB)) {
      Artifact zipOutput = bundling.getIntermediateArtifacts().compiledXibFileZip(original);
      String archiveRoot = BundleableFile.flatBundlePath(
          FileSystemUtils.replaceExtension(original.getExecPath(), ".nib"));

      ruleContext.registerAction(
          ObjcRuleClasses.spawnAppleEnvActionBuilder(appleConfiguration, platform)
              .setMnemonic("XibCompile")
              .setExecutable(attributes.ibtoolWrapper())
              .setCommandLine(ibActionsCommandLine(archiveRoot, zipOutput, original))
              .addOutput(zipOutput)
              .addInput(original)
              // Disable sandboxing due to Bazel issue #2189.
              .disableSandboxing()
              .build(ruleContext));
    }
  }

  private void registerConvertStringsActions(ObjcProvider objcProvider) {
    for (Artifact strings : objcProvider.get(ObjcProvider.STRINGS)) {
      Artifact bundled = bundling.getIntermediateArtifacts().convertedStringsFile(strings);
      ruleContext.registerAction(
          ObjcRuleClasses.spawnAppleEnvActionBuilder(appleConfiguration, platform)
              .setMnemonic("ConvertStringsPlist")
              .setExecutable(PathFragment.create("/usr/bin/plutil"))
              .setCommandLine(CustomCommandLine.builder()
                  .add("-convert").add("binary1")
                  .addExecPath("-o", bundled)
                  .add("--")
                  .addPath(strings.getExecPath())
                  .build())
              .addInput(strings)
              .addInput(CompilationSupport.xcrunwrapper(ruleContext).getExecutable())
              .addOutput(bundled)
              .build(ruleContext));
    }
  }

  /**
   * Creates action to merge multiple Info.plist files of a bundle into a single Info.plist. The
   * merge action is necessary if there are more than one input plist files or we have a bundle ID
   * to stamp on the merged plist.
   */
  private void registerMergeInfoplistAction(
      NestedSet<Artifact> mergingContentArtifacts, PlMergeControlBytes controlBytes) {
    if (!bundling.needsToMergeInfoplist()) {
      return; // Nothing to do here.
    }

    Artifact plMergeControlArtifact = baseNameArtifact(ruleContext, ".plmerge-control");

    ruleContext.registerAction(
        new BinaryFileWriteAction(
            ruleContext.getActionOwner(),
            plMergeControlArtifact,
            controlBytes,
            /*makeExecutable=*/ false));

    ruleContext.registerAction(
        new SpawnAction.Builder()
            .setMnemonic("MergeInfoPlistFiles")
            .setExecutable(attributes.plmerge())
            .addArgument("--control")
            .addInputArgument(plMergeControlArtifact)
            .addTransitiveInputs(mergingContentArtifacts)
            .addOutput(bundling.getIntermediateArtifacts().mergedInfoplist())
            .build(ruleContext));
  }

  /**
   * Returns an {@link Artifact} with name prefixed with prefix given in {@link Bundling} if
   * available. This helps in creating unique artifact name when multiple bundles are created
   * with a different name than the target name.
   */
  private Artifact baseNameArtifact(RuleContext ruleContext, String artifactName) {
    String prefixedArtifactName;
    if (bundling.getArtifactPrefix() != null) {
      prefixedArtifactName = String.format("-%s%s", bundling.getArtifactPrefix(), artifactName);
    } else {
      prefixedArtifactName = artifactName;
    }
    return ObjcRuleClasses.artifactByAppendingToBaseName(ruleContext, prefixedArtifactName);
  }

  private void registerActoolActionIfNecessary(ObjcProvider objcProvider) {
    Optional<Artifact> actoolzipOutput = bundling.getActoolzipOutput();
    if (!actoolzipOutput.isPresent()) {
      return;
    }

    Artifact actoolPartialInfoplist = actoolPartialInfoplist(objcProvider).get();
    Artifact zipOutput = actoolzipOutput.get();

    // TODO(bazel-team): Do not use the deploy jar explicitly here. There is currently a bug where
    // we cannot .setExecutable({java_binary target}) and set REQUIRES_DARWIN in the execution info.
    // Note that below we set the archive root to the empty string. This means that the generated
    // zip file will be rooted at the bundle root, and we have to prepend the bundle root to each
    // entry when merging it with the final .ipa file.
    ruleContext.registerAction(
        ObjcRuleClasses.spawnAppleEnvActionBuilder(appleConfiguration, platform)
            .setMnemonic("AssetCatalogCompile")
            .setExecutable(attributes.actoolWrapper())
            .addTransitiveInputs(objcProvider.get(ASSET_CATALOG))
            .addOutput(zipOutput)
            .addOutput(actoolPartialInfoplist)
            .setCommandLine(actoolzipCommandLine(
                objcProvider,
                zipOutput,
                actoolPartialInfoplist))
            .disableSandboxing()
            .build(ruleContext));
  }

  private CommandLine actoolzipCommandLine(ObjcProvider provider, Artifact zipOutput,
      Artifact partialInfoPlist) {
    PlatformType platformType = PlatformType.IOS;
    // watchOS 1 and 2 use different platform arguments. It is likely that versions 2 and later will
    // use the watchos platform whereas watchOS 1 uses the iphone platform.
    if (isBuildingForWatch() && bundling.getBundleDir().startsWith("Watch")) {
      platformType = PlatformType.WATCHOS;
    }
    CustomCommandLine.Builder commandLine =
        CustomCommandLine.builder()
            // The next three arguments are positional, i.e. they don't have flags before them.
            .addPath(zipOutput.getExecPath())
            .add("--platform")
            .add(appleConfiguration.getMultiArchPlatform(platformType)
                .getLowerCaseNameInPlist())
            .addExecPath("--output-partial-info-plist", partialInfoPlist)
            .add("--minimum-deployment-target")
            .add(bundling.getMinimumOsVersion().toString());

    for (TargetDeviceFamily targetDeviceFamily : targetDeviceFamiliesForResources()) {
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
  private Optional<Artifact> actoolPartialInfoplist(ObjcProvider objcProvider) {
    if (objcProvider.hasAssetCatalogs()) {
      return Optional.of(bundling.getIntermediateArtifacts().actoolPartialInfoplist());
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
     * Returns the location of the ibtoolwrapper tool.
     */
    FilesToRunProvider ibtoolWrapper() {
      return ruleContext.getExecutablePrerequisite("$ibtoolwrapper", Mode.HOST);
    }

    /**
     * Returns the location of the momcwrapper.
     */
    FilesToRunProvider momcWrapper() {
      return ruleContext.getExecutablePrerequisite("$momcwrapper", Mode.HOST);
    }

    /**
     * Returns the location of the actoolwrapper.
     */
    FilesToRunProvider actoolWrapper() {
      return ruleContext.getExecutablePrerequisite("$actoolwrapper", Mode.HOST);
    }
  }
}
