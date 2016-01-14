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
import com.google.devtools.build.lib.rules.objc.TargetDeviceFamily.InvalidFamilyNameException;
import com.google.devtools.build.lib.rules.objc.TargetDeviceFamily.RepeatedFamilyNameException;
import com.google.devtools.build.lib.rules.objc.XcodeProvider.Builder;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.HashMap;
import java.util.List;
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

  private final RuleContext ruleContext;
  private final ExtraActoolArgs extraActoolArgs;
  private final Bundling bundling;
  private final Attributes attributes;

  /**
   * Creates a new bundle support with no special {@code actool} arguments.
   *
   * @param ruleContext context this bundle is constructed in
   * @param bundling bundle information as configured for this rule
   */
  public BundleSupport(RuleContext ruleContext, Bundling bundling) {
    this(ruleContext, bundling, new ExtraActoolArgs());
  }

  /**
   * Creates a new bundle support.
   *
   * @param ruleContext context this bundle is constructed in
   * @param bundling bundle information as configured for this rule
   * @param extraActoolArgs any additional parameters to be used for invoking {@code actool}
   */
  public BundleSupport(RuleContext ruleContext,
      Bundling bundling, ExtraActoolArgs extraActoolArgs) {
    this.ruleContext = ruleContext;
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
      PlMergeControlBytes plMergeControlBytes = new PlMergeControlBytes(bundling, mergedPlist);
      registerMergeInfoplistAction(mergingContentArtifacts, plMergeControlBytes);
    }
    return this;
  }

  /**
   * Adds any Xcode settings related to this bundle to the given provider builder.
   *
   * @return this bundle support
   */
  BundleSupport addXcodeSettings(Builder xcodeProviderBuilder) {
    if (bundling.getBundleInfoplist().isPresent()) {
      xcodeProviderBuilder.setBundleInfoplist(bundling.getBundleInfoplist().get());
    }
    return this;
  }

  private void validatePlatform() {
    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);
    Platform platform = null;
    for (String architecture : appleConfiguration.getIosMultiCpus()) {
      if (platform == null) {
        platform = Platform.forIosArch(architecture);
      } else if (platform != Platform.forIosArch(architecture)) {
        ruleContext.ruleError(
            String.format("In builds which require bundling, --ios_multi_cpus does not currently "
                + "allow values for both simulator and device builds. Flag was %s",
                appleConfiguration.getIosMultiCpus()));
      }
    }
  }

  private void validateResources(ObjcProvider objcProvider) {
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
  }

  /**
   * Validates bundle support.
   * <ul>
   * <li>Validates that resources defined in this rule and its dependencies and written to this
   *     bundle are legal (for example that they are not mapped to the same bundle location)
   * <li>Validates the platform for this build is either simulator or device, and does not
   *     contain architectures for both platforms
   * </ul>
   *
   * @return this bundle support
   */
  BundleSupport validate(ObjcProvider objcProvider) {
    validatePlatform();
    validateResources(objcProvider);

    return this;
  }

  /**
   * Returns a set containing the {@link TargetDeviceFamily} values
   * which this bundle is targeting. Returns an empty set for any
   * invalid value of the target device families attribute.
   */
  ImmutableSet<TargetDeviceFamily> targetDeviceFamilies() {
    return attributes.families();
  }

  private void registerInterfaceBuilderActions(ObjcProvider objcProvider) {
    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);
    for (Artifact storyboardInput : objcProvider.get(ObjcProvider.STORYBOARD)) {
      String archiveRoot = BundleableFile.flatBundlePath(storyboardInput.getExecPath()) + "c";
      Artifact zipOutput = intermediateArtifacts.compiledStoryboardZip(storyboardInput);

      ruleContext.registerAction(
          ObjcRuleClasses.spawnXcrunActionBuilder(ruleContext)
              .setMnemonic("StoryboardCompile")
              .setExecutable(attributes.ibtoolWrapper())
              .setCommandLine(ibActionsCommandLine(archiveRoot, zipOutput, storyboardInput))
              .addOutput(zipOutput)
              .addInput(storyboardInput)
              // TODO(dmaclach): Adding realpath and xcrunwrapper should not be required once
              // https://github.com/bazelbuild/bazel/issues/285 is fixed.
              .addInput(attributes.realpath())
              .addInput(CompilationSupport.xcrunwrapper(ruleContext).getExecutable())
              .setVerboseFailuresAndSubcommandsInEnv()
              .build(ruleContext));
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

    for (TargetDeviceFamily targetDeviceFamily : attributes.families()) {
      commandLine.add("--target-device").add(targetDeviceFamily.name().toLowerCase(Locale.US));
    }

    return commandLine
        .addPath(storyboardInput.getExecPath())
        .build();
  }

  private void registerMomczipActions(ObjcProvider objcProvider) {
    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);
    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);
    Iterable<Xcdatamodel> xcdatamodels = Xcdatamodels.xcdatamodels(
        intermediateArtifacts, objcProvider.get(ObjcProvider.XCDATAMODEL));
    for (Xcdatamodel datamodel : xcdatamodels) {
      Artifact outputZip = datamodel.getOutputZip();
      ruleContext.registerAction(
          ObjcRuleClasses.spawnXcrunActionBuilder(ruleContext)
              .setMnemonic("MomCompile")
              .setExecutable(attributes.momcWrapper())
              .addOutput(outputZip)
              .addInputs(datamodel.getInputs())
              // TODO(dmaclach): Adding realpath and xcrunwrapper should not be required once
              // https://github.com/google/bazel/issues/285 is fixed.
              .addInput(attributes.realpath())
              .addInput(CompilationSupport.xcrunwrapper(ruleContext).getExecutable())
              .setVerboseFailuresAndSubcommandsInEnv()
             .setCommandLine(CustomCommandLine.builder()
                  .addPath(outputZip.getExecPath())
                  .add(datamodel.archiveRootForMomczip())
                  .add("-XD_MOMC_SDKROOT=" + AppleToolchain.sdkDir())
                  .add("-XD_MOMC_IOS_TARGET_VERSION=" + bundling.getMinimumOsVersion())
                  .add("-MOMC_PLATFORMS")
                  .add(appleConfiguration.getBundlingPlatform().getLowerCaseNameInPlist())
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
          ObjcRuleClasses.spawnXcrunActionBuilder(ruleContext)
              .setMnemonic("XibCompile")
              .setExecutable(attributes.ibtoolWrapper())
              .setCommandLine(ibActionsCommandLine(archiveRoot, zipOutput, original))
              .addOutput(zipOutput)
              .addInput(original)
              // TODO(dmaclach): Adding realpath and xcrunwrapper should not be required once
              // https://github.com/bazelbuild/bazel/issues/285 is fixed.
              .addInput(attributes.realpath())
              .addInput(CompilationSupport.xcrunwrapper(ruleContext).getExecutable())
              .setVerboseFailuresAndSubcommandsInEnv()
              .build(ruleContext));
    }
  }

  private void registerConvertStringsActions(ObjcProvider objcProvider) {
    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);
    for (Artifact strings : objcProvider.get(ObjcProvider.STRINGS)) {
      Artifact bundled = intermediateArtifacts.convertedStringsFile(strings);
      ruleContext.registerAction(ObjcRuleClasses.spawnXcrunActionBuilder(ruleContext)
          .setMnemonic("ConvertStringsPlist")
          .setExecutable(new PathFragment("/usr/bin/plutil"))
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
    
    Artifact plMergeControlArtifact =
        ObjcRuleClasses.artifactByAppendingToBaseName(ruleContext, ".plmerge-control");

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
            .addOutput(ObjcRuleClasses.intermediateArtifacts(ruleContext).mergedInfoplist())
            .setVerboseFailuresAndSubcommandsInEnv()
            .build(ruleContext));
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
        ObjcRuleClasses.spawnXcrunActionBuilder(ruleContext)
            .setMnemonic("AssetCatalogCompile")
            .setExecutable(attributes.actoolWrapper())
            .addTransitiveInputs(objcProvider.get(ASSET_CATALOG))
            .addOutput(zipOutput)
            .addOutput(actoolPartialInfoplist)
            // TODO(dmaclach): Adding realpath and xcrunwrapper should not be required once
            // https://github.com/google/bazel/issues/285 is fixed.
            .addInput(attributes.realpath())
            .addInput(CompilationSupport.xcrunwrapper(ruleContext).getExecutable())
            .setVerboseFailuresAndSubcommandsInEnv()
            .setCommandLine(actoolzipCommandLine(
                objcProvider,
                zipOutput,
                actoolPartialInfoplist))
            .build(ruleContext));
  }

  private CommandLine actoolzipCommandLine(ObjcProvider provider, Artifact zipOutput,
      Artifact partialInfoPlist) {
    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);
    CustomCommandLine.Builder commandLine =
        CustomCommandLine.builder()
            // The next three arguments are positional, i.e. they don't have flags before them.
            .addPath(zipOutput.getExecPath())
            .add("--platform")
            .add(appleConfiguration.getBundlingPlatform().getLowerCaseNameInPlist())
            .addExecPath("--output-partial-info-plist", partialInfoPlist)
            .add("--minimum-deployment-target")
            .add(bundling.getMinimumOsVersion().toString());

    for (TargetDeviceFamily targetDeviceFamily : attributes.families()) {
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
     * Returns the value of the {@code families} attribute in a form
     * that is more useful than a list of strings. Returns an empty
     * set for any invalid {@code families} attribute value, including
     * an empty list.
     */
    ImmutableSet<TargetDeviceFamily> families() {
      List<String> rawFamilies = ruleContext.attributes().get("families", Type.STRING_LIST);
      try {
        return ImmutableSet.copyOf(TargetDeviceFamily.fromNamesInRule(rawFamilies));
      } catch (InvalidFamilyNameException | RepeatedFamilyNameException e) {
        return ImmutableSet.of();
      }
    }

    /**
     * Returns the location of the ibtoolwrapper tool.
     */
    FilesToRunProvider ibtoolWrapper() {
      return ruleContext.getExecutablePrerequisite("$ibtoolwrapper", Mode.HOST);
    }

    /**
     * Returns the location of the realpath tool.
     * TODO(dmaclach): Should not be required once https://github.com/bazelbuild/bazel/issues/285
     * is fixed.
     */
    Artifact realpath() {
      return ruleContext.getPrerequisiteArtifact("$realpath", Mode.HOST);
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
