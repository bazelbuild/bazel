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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromTemplates;
import static com.google.devtools.build.xcode.common.TargetDeviceFamily.UI_DEVICE_FAMILY_VALUES;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.actions.BinaryFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Substitution;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.objc.ObjcActionsBuilder.ExtraActoolArgs;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.xcode.common.InvalidFamilyNameException;
import com.google.devtools.build.xcode.common.Platform;
import com.google.devtools.build.xcode.common.RepeatedFamilyNameException;
import com.google.devtools.build.xcode.common.TargetDeviceFamily;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.XcodeprojBuildSetting;

import java.util.List;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * Support for released bundles, such as an application or extension. Such a bundle is generally
 * composed of a top-level {@link BundleSupport bundle}, potentially signed, as well as some debug
 * information, if {@link ObjcConfiguration#generateDebugSymbols() requested}.
 *
 * <p>Contains actions, validation logic and provider value generation.
 *
 * <p>Methods on this class can be called in any order without impacting the result.
 */
public final class ReleaseBundlingSupport {

  /**
   * Template for the containing application folder.
   */
  public static final SafeImplicitOutputsFunction IPA = fromTemplates("%{name}.ipa");

  @VisibleForTesting
  static final String NO_ASSET_CATALOG_ERROR_FORMAT =
      "a value was specified (%s), but this app does not have any asset catalogs";
  @VisibleForTesting
  static final String INVALID_FAMILIES_ERROR =
      "Expected one or two strings from the list 'iphone', 'ipad'";
  @VisibleForTesting
  static final String DEVICE_NO_PROVISIONING_PROFILE =
      "Provisioning profile must be set for device build";

  @VisibleForTesting
  static final String PROVISIONING_PROFILE_BUNDLE_FILE = "embedded.mobileprovision";
  @VisibleForTesting
  static final String APP_BUNDLE_DIR_FORMAT = "Payload/%s.app";
  @VisibleForTesting
  static final String EXTENSION_BUNDLE_DIR_FORMAT = "PlugIns/%s.appex";

  private final Attributes attributes;
  private final BundleSupport bundleSupport;
  private final RuleContext ruleContext;
  private final Bundling bundling;
  private final ObjcProvider objcProvider;
  private final LinkedBinary linkedBinary;
  private final ImmutableSet<TargetDeviceFamily> families;
  private final IntermediateArtifacts intermediateArtifacts;

  /**
   * Indicator as to whether this rule generates a binary directly or whether only dependencies
   * should be considered.
   */
  enum LinkedBinary {
    /**
     * This rule generates its own binary which should be included as well as dependency-generated
     * binaries.
     */
    LOCAL_AND_DEPENDENCIES,

    /**
     * This rule does not generate its own binary, only consider binaries from dependencies.
     */
    DEPENDENCIES_ONLY
  }

  /**
   * Creates a new application support within the given rule context.
   *
   * @param ruleContext context for the application-generating rule
   * @param objcProvider provider containing all dependencies' information as well as some of this
   *    rule's
   * @param optionsProvider provider containing options and plist settings for this rule and its
   *    dependencies
   * @param linkedBinary whether to look for a linked binary from this rule and dependencies or just
   *    the latter
   * @param bundleDirFormat format string representing the bundle's directory with a single
   *     placeholder for the target name (e.g. {@code "Payload/%s.app"})
   */
  ReleaseBundlingSupport(
      RuleContext ruleContext, ObjcProvider objcProvider, OptionsProvider optionsProvider,
      LinkedBinary linkedBinary, String bundleDirFormat) {
    this.linkedBinary = linkedBinary;
    this.attributes = new Attributes(ruleContext);
    this.ruleContext = ruleContext;
    this.objcProvider = objcProvider;
    this.families = ImmutableSet.copyOf(attributes.families());
    this.intermediateArtifacts = ObjcRuleClasses.intermediateArtifacts(ruleContext);
    bundling = bundling(ruleContext, objcProvider, optionsProvider, bundleDirFormat);
    bundleSupport = new BundleSupport(ruleContext, families, bundling, extraActoolArgs());
  }

  /**
   * Validates application-related attributes set on this rule and registers any errors with the
   * rule context.
   *
   * @return this application support
   */
  ReleaseBundlingSupport validateAttributes() {
    // No asset catalogs. That means you cannot specify app_icon or
    // launch_image attributes, since they must not exist. However, we don't
    // run actool in this case, which means it does not do validity checks,
    // and we MUST raise our own error somehow...
    if (!objcProvider.hasAssetCatalogs()) {
      if (attributes.appIcon() != null) {
        ruleContext.attributeError("app_icon",
            String.format(NO_ASSET_CATALOG_ERROR_FORMAT, attributes.appIcon()));
      }
      if (attributes.launchImage() != null) {
        ruleContext.attributeError("launch_image",
            String.format(NO_ASSET_CATALOG_ERROR_FORMAT, attributes.launchImage()));
      }
    }

    if (families.isEmpty()) {
      ruleContext.attributeError("families", INVALID_FAMILIES_ERROR);
    }

    return this;
  }

  /**
   * Registers actions required to build an application. This includes any
   * {@link BundleSupport#registerActions(ObjcProvider) bundle} and bundle merge actions, signing
   * this application if appropriate and combining several single-architecture binaries into one
   * multi-architecture binary.
   *
   * @return this application support
   */
  ReleaseBundlingSupport registerActions() {
    bundleSupport.registerActions(objcProvider);

    registerCombineArchitecturesAction();

    ObjcConfiguration objcConfiguration = ObjcRuleClasses.objcConfiguration(ruleContext);
    Artifact ipaOutput = ruleContext.getImplicitOutputArtifact(IPA);

    Artifact maybeSignedIpa;
    if (objcConfiguration.getPlatform() == Platform.SIMULATOR) {
      maybeSignedIpa = ipaOutput;
    } else if (attributes.provisioningProfile() == null) {
      throw new IllegalStateException(DEVICE_NO_PROVISIONING_PROFILE);
    } else {
      maybeSignedIpa = registerBundleSigningActions(ipaOutput);
    }
    
    BundleMergeControlBytes bundleMergeControlBytes = new BundleMergeControlBytes(
        bundling, maybeSignedIpa, objcConfiguration, families);
    registerBundleMergeActions(
        maybeSignedIpa, bundling.getBundleContentArtifacts(), bundleMergeControlBytes);

    return this;
  }

  private Artifact registerBundleSigningActions(Artifact ipaOutput) {
    PathFragment entitlementsDirectory = ruleContext.getUniqueDirectory("entitlements");
    Artifact teamPrefixFile = ruleContext.getRelatedArtifact(
        entitlementsDirectory, ".team_prefix_file");
    registerExtractTeamPrefixAction(teamPrefixFile);

    Artifact entitlementsNeedingSubstitution = attributes.entitlements();
    if (entitlementsNeedingSubstitution == null) {
      entitlementsNeedingSubstitution = ruleContext.getRelatedArtifact(
          entitlementsDirectory, ".entitlements_with_variables");
      registerExtractEntitlementsAction(entitlementsNeedingSubstitution);
    }
    Artifact entitlements = ruleContext.getRelatedArtifact(
        entitlementsDirectory, ".entitlements");
    registerEntitlementsVariableSubstitutionAction(
        entitlementsNeedingSubstitution, entitlements, teamPrefixFile);
    Artifact ipaUnsigned = ObjcRuleClasses.artifactByAppendingToRootRelativePath(
        ruleContext, ipaOutput.getExecPath(), ".unsigned");
    registerSignBundleAction(entitlements, ipaOutput, ipaUnsigned);
    return ipaUnsigned;
  }

  /**
   * Adds bundle- and application-related settings to the given Xcode provider builder.
   *
   * @return this application support
   */
  ReleaseBundlingSupport addXcodeSettings(XcodeProvider.Builder xcodeProviderBuilder) {
    bundleSupport.addXcodeSettings(xcodeProviderBuilder);
    xcodeProviderBuilder.addXcodeprojBuildSettings(buildSettings());

    return this;
  }

  /**
   * Adds any files to the given nested set builder that should be built if this application is the
   * top level target in a blaze invocation.
   *
   * @return this application support
   */
  ReleaseBundlingSupport addFilesToBuild(NestedSetBuilder<Artifact> filesToBuild) {
    NestedSetBuilder<Artifact> debugSymbolBuilder = NestedSetBuilder.<Artifact>stableOrder()
        .addTransitive(objcProvider.get(ObjcProvider.DEBUG_SYMBOLS));

    if (linkedBinary == LinkedBinary.LOCAL_AND_DEPENDENCIES
        && ObjcRuleClasses.objcConfiguration(ruleContext).generateDebugSymbols()) {
      IntermediateArtifacts intermediateArtifacts =
          ObjcRuleClasses.intermediateArtifacts(ruleContext);
      debugSymbolBuilder.add(intermediateArtifacts.dsymPlist())
          .add(intermediateArtifacts.dsymSymbol())
          .add(intermediateArtifacts.breakpadSym());
    }

    filesToBuild.add(ruleContext.getImplicitOutputArtifact(ReleaseBundlingSupport.IPA))
        // TODO(bazel-team): Fat binaries may require some merging of these file rather than just
        // making them available.
        .addTransitive(debugSymbolBuilder.build());
    return this;
  }

  /**
   * Creates the {@link XcTestAppProvider} that can be used if this application is used as an
   * {@code xctest_app}.
   */
  XcTestAppProvider xcTestAppProvider() {
    // We want access to #import-able things from our test rig's dependency graph, but we don't
    // want to link anything since that stuff is shared automatically by way of the
    // -bundle_loader linker flag.
    ObjcProvider partialObjcProvider = new ObjcProvider.Builder()
        .addTransitiveAndPropagate(ObjcProvider.HEADER, objcProvider)
        .addTransitiveAndPropagate(ObjcProvider.INCLUDE, objcProvider)
        .addTransitiveAndPropagate(ObjcProvider.SDK_DYLIB, objcProvider)
        .addTransitiveAndPropagate(ObjcProvider.SDK_FRAMEWORK, objcProvider)
        .addTransitiveAndPropagate(ObjcProvider.WEAK_SDK_FRAMEWORK, objcProvider)
        .addTransitiveAndPropagate(ObjcProvider.FRAMEWORK_DIR, objcProvider)
        .addTransitiveAndPropagate(ObjcProvider.FRAMEWORK_FILE, objcProvider)
        .build();
    // TODO(bazel-team): Handle the FRAMEWORK_DIR key properly. We probably want to add it to
    // framework search paths, but not actually link it with the -framework flag.
    return new XcTestAppProvider(intermediateArtifacts.singleArchitectureBinary(),
        ruleContext.getImplicitOutputArtifact(IPA), partialObjcProvider);
  }

  /**
   * Registers an action to generate a runner script based on a template.
   */
  ReleaseBundlingSupport registerGenerateRunnerScriptAction(Artifact runnerScript,
      Artifact ipaInput) {
    ObjcConfiguration objcConfiguration = ObjcRuleClasses.objcConfiguration(ruleContext);
    ImmutableList<Substitution> substitutions = ImmutableList.of(
        Substitution.of("%app_name%", ruleContext.getLabel().getName()),
        Substitution.of("%ipa_file%", ipaInput.getRootRelativePath().getPathString()),
        Substitution.of("%sim_device%", objcConfiguration.getIosSimulatorDevice()),
        Substitution.of("%sdk_version%", objcConfiguration.getIosSimulatorVersion()),
        Substitution.of("%iossim%", attributes.iossim().getRootRelativePath().getPathString()));

    ruleContext.registerAction(
        new TemplateExpansionAction(ruleContext.getActionOwner(), attributes.runnerScriptTemplate(),
            runnerScript, substitutions, true));
    return this;
  }

  /**
   * Returns a {@link RunfilesSupport} that uses the provided runner script as the executable.
   */
  RunfilesSupport runfilesSupport(Artifact runnerScript) {
    Artifact ipaFile = ruleContext.getImplicitOutputArtifact(ReleaseBundlingSupport.IPA);
    Runfiles runfiles = new Runfiles.Builder()
        .addArtifact(ipaFile)
        .addArtifact(runnerScript)
        .addArtifact(attributes.iossim())
        .build();
    return RunfilesSupport.withExecutable(ruleContext, runfiles, runnerScript);
  }

  private ExtraActoolArgs extraActoolArgs() {
    ImmutableList.Builder<String> extraArgs = ImmutableList.builder();
    if (attributes.appIcon() != null) {
      extraArgs.add("--app-icon", attributes.appIcon());
    }
    if (attributes.launchImage() != null) {
      extraArgs.add("--launch-image", attributes.launchImage());
    }
    return new ExtraActoolArgs(extraArgs.build());
  }

  private static Bundling bundling(
      RuleContext ruleContext, ObjcProvider objcProvider, OptionsProvider optionsProvider,
      String bundleDirFormat) {
    ImmutableList<BundleableFile> extraBundleFiles;
    ObjcConfiguration objcConfiguration = ObjcRuleClasses.objcConfiguration(ruleContext);
    if (objcConfiguration.getPlatform() == Platform.DEVICE) {
      extraBundleFiles = ImmutableList.of(new BundleableFile(
          new Attributes(ruleContext).provisioningProfile(),
          PROVISIONING_PROFILE_BUNDLE_FILE));
    } else {
      extraBundleFiles = ImmutableList.of();
    }
    
    String primaryBundleId = null; 
    String fallbackBundleId = null;

    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("bundle_id")) {
      primaryBundleId = ruleContext.attributes().get("bundle_id", Type.STRING);
    } else {
      fallbackBundleId = ruleContext.attributes().get("bundle_id", Type.STRING);
    }

    return new Bundling.Builder()
        .setName(ruleContext.getLabel().getName())
        .setBundleDirFormat(bundleDirFormat)
        .setExtraBundleFiles(extraBundleFiles)
        .setObjcProvider(objcProvider)
        .setInfoplistMerging(
            BundleSupport.infoPlistMerging(ruleContext, objcProvider, optionsProvider))
        .setIntermediateArtifacts(ObjcRuleClasses.intermediateArtifacts(ruleContext))
        .setPrimaryBundleId(primaryBundleId)
        .setFallbackBundleId(fallbackBundleId)
        .build();
  }

  private void registerCombineArchitecturesAction() {
    Artifact resultingLinkedBinary = intermediateArtifacts.combinedArchitectureBinary();
    NestedSet<Artifact> linkedBinaries = linkedBinaries();

    ruleContext.registerAction(ObjcActionsBuilder.spawnOnDarwinActionBuilder()
        .setMnemonic("ObjcCombiningArchitectures")
        .addTransitiveInputs(linkedBinaries)
        .addOutput(resultingLinkedBinary)
        .setExecutable(ObjcActionsBuilder.LIPO)
        .setCommandLine(CustomCommandLine.builder()
            .addExecPaths("-create", linkedBinaries)
            .addExecPath("-o", resultingLinkedBinary)
            .build())
        .build(ruleContext));
  }

  private NestedSet<Artifact> linkedBinaries() {
    NestedSetBuilder<Artifact> linkedBinariesBuilder = NestedSetBuilder.<Artifact>stableOrder()
        .addTransitive(attributes.dependentLinkedBinaries());
    if (linkedBinary == LinkedBinary.LOCAL_AND_DEPENDENCIES) {
      linkedBinariesBuilder.add(intermediateArtifacts.singleArchitectureBinary());
    }
    return linkedBinariesBuilder.build();
  }

  /** Returns this target's Xcode build settings. */
  private Iterable<XcodeprojBuildSetting> buildSettings() {
    ImmutableList.Builder<XcodeprojBuildSetting> buildSettings = new ImmutableList.Builder<>();
    if (attributes.appIcon() != null) {
      buildSettings.add(XcodeprojBuildSetting.newBuilder()
          .setName("ASSETCATALOG_COMPILER_APPICON_NAME")
          .setValue(attributes.appIcon())
          .build());
    }
    if (attributes.launchImage() != null) {
      buildSettings.add(XcodeprojBuildSetting.newBuilder()
          .setName("ASSETCATALOG_COMPILER_LAUNCHIMAGE_NAME")
          .setValue(attributes.launchImage())
          .build());
    }

    // Convert names to a sequence containing "1" and/or "2" for iPhone and iPad, respectively.
    Iterable<Integer> familyIndexes =
        families.isEmpty() ? ImmutableList.<Integer>of() : UI_DEVICE_FAMILY_VALUES.get(families);
    buildSettings.add(XcodeprojBuildSetting.newBuilder()
        .setName("TARGETED_DEVICE_FAMILY")
        .setValue(Joiner.on(',').join(familyIndexes))
        .build());

    Artifact entitlements = attributes.entitlements();
    if (entitlements != null) {
      buildSettings.add(XcodeprojBuildSetting.newBuilder()
          .setName("CODE_SIGN_ENTITLEMENTS")
          .setValue("$(WORKSPACE_ROOT)/" + entitlements.getExecPathString())
          .build());
    }

    return buildSettings.build();
  }

  private ReleaseBundlingSupport registerSignBundleAction(
      Artifact entitlements, Artifact ipaOutput, Artifact ipaUnsigned) {
    // TODO(bazel-team): Support variable substitution
    ruleContext.registerAction(ObjcActionsBuilder.spawnOnDarwinActionBuilder()
        .setMnemonic("IosSignBundle")
        .setProgressMessage("Signing iOS bundle: " + ruleContext.getLabel())
        .setExecutable(new PathFragment("/bin/bash"))
        .addArgument("-c")
        // TODO(bazel-team): Support --resource-rules for resources
        .addArgument("set -e && "
            + "t=$(mktemp -d -t signing_intermediate) && "
            // Get an absolute path since we need to cd into the temp directory for zip.
            + "signed_ipa=${PWD}/" + ipaOutput.getExecPathString() + " && "
            + "unzip -qq " + ipaUnsigned.getExecPathString() + " -d ${t} && "
            + codesignCommand(
                attributes.provisioningProfile(),
                entitlements,
                "${t}/" + bundling.getBundleDir())
            // Using zip since we need to preserve permissions
            + " && cd \"${t}\" && /usr/bin/zip -q -r \"${signed_ipa}\" .")
        .addInput(ipaUnsigned)
        .addInput(attributes.provisioningProfile())
        .addInput(entitlements)
        .addOutput(ipaOutput)
        .build(ruleContext));

    return this;
  }

  private void registerBundleMergeActions(Artifact ipaUnsigned,
      NestedSet<Artifact> bundleContentArtifacts, BundleMergeControlBytes controlBytes) {
    Artifact bundleMergeControlArtifact =
        ObjcRuleClasses.artifactByAppendingToBaseName(ruleContext, ".ipa-control");

    ruleContext.registerAction(
        new BinaryFileWriteAction(
            ruleContext.getActionOwner(), bundleMergeControlArtifact, controlBytes,
            /*makeExecutable=*/false));

    ruleContext.registerAction(new SpawnAction.Builder()
        .setMnemonic("IosBundle")
        .setProgressMessage("Bundling iOS application: " + ruleContext.getLabel())
        .setExecutable(attributes.bundleMergeExecutable())
        .addInputArgument(bundleMergeControlArtifact)
        .addTransitiveInputs(bundleContentArtifacts)
        .addOutput(ipaUnsigned)
        .build(ruleContext));
  }

  private void registerExtractTeamPrefixAction(Artifact teamPrefixFile) {
    ruleContext.registerAction(ObjcActionsBuilder.spawnOnDarwinActionBuilder()
        .setMnemonic("ExtractIosTeamPrefix")
        .setExecutable(new PathFragment("/bin/bash"))
        .addArgument("-c")
        .addArgument("set -e &&"
            + " PLIST=$(" + extractPlistCommand(attributes.provisioningProfile()) + ") && "

            // We think PlistBuddy uses PRead internally to seek through the file. Or possibly
            // mmaps the file. Or something similar.
            //
            // Pipe FDs do not support PRead or mmap, though.
            //
            // <<< however does something magical like write to a temporary file or something
            // like that internally, which means that this Just Works.
            + " PREFIX=$(/usr/libexec/PlistBuddy -c 'Print ApplicationIdentifierPrefix:0'"
            + " /dev/stdin <<< \"${PLIST}\") && "
            + " echo ${PREFIX} > " + teamPrefixFile.getExecPathString())
        .addInput(attributes.provisioningProfile())
        .addOutput(teamPrefixFile)
        .build(ruleContext));
  }

  private ReleaseBundlingSupport registerExtractEntitlementsAction(Artifact entitlements) {
    // See Apple Glossary (http://goo.gl/EkhXOb)
    // An Application Identifier is constructed as: TeamID.BundleID
    // TeamID is extracted from the provisioning profile.
    // BundleID consists of a reverse-DNS string to identify the app, where the last component
    // is the application name, and is specified as an attribute.

    ruleContext.registerAction(ObjcActionsBuilder.spawnOnDarwinActionBuilder()
        .setMnemonic("ExtractIosEntitlements")
        .setProgressMessage("Extracting entitlements: " + ruleContext.getLabel())
        .setExecutable(new PathFragment("/bin/bash"))
        .addArgument("-c")
        .addArgument("set -e && "
            + "PLIST=$("
            + extractPlistCommand(attributes.provisioningProfile()) + ") && "

            // We think PlistBuddy uses PRead internally to seek through the file. Or possibly
            // mmaps the file. Or something similar.
            //
            // Pipe FDs do not support PRead or mmap, though.
            //
            // <<< however does something magical like write to a temporary file or something
            // like that internally, which means that this Just Works.

            + "/usr/libexec/PlistBuddy -x -c 'Print Entitlements' /dev/stdin <<< \"${PLIST}\" "
            + "> " + entitlements.getExecPathString())
        .addInput(attributes.provisioningProfile())
        .addOutput(entitlements)
        .build(ruleContext));

    return this;
  }

  private void registerEntitlementsVariableSubstitutionAction(Artifact in, Artifact out,
      Artifact prefix) {
    String escapedBundleId = ShellUtils.shellEscape(attributes.bundleId());
    ruleContext.registerAction(new SpawnAction.Builder()
        .setMnemonic("SubstituteIosEntitlements")
        .setExecutable(new PathFragment("/bin/bash"))
        .addArgument("-c")
        .addArgument("set -e && "
            + "PREFIX=\"$(cat " + prefix.getExecPathString() + ")\" && "
            + "sed "
            // Replace .* from default entitlements file with bundle ID where suitable.
            + "-e \"s#${PREFIX}\\.\\*#${PREFIX}." + escapedBundleId + "#g\" "

            // Replace some variables that people put in their own entitlements files
            + "-e \"s#\\$(AppIdentifierPrefix)#${PREFIX}.#g\" "
            + "-e \"s#\\$(CFBundleIdentifier)#" + escapedBundleId + "#g\" "

            + in.getExecPathString() + " "
            + "> " + out.getExecPathString())
        .addInput(in)
        .addInput(prefix)
        .addOutput(out)
        .build(ruleContext));
  }


  private String extractPlistCommand(Artifact provisioningProfile) {
    return "security cms -D -i " + ShellUtils.shellEscape(provisioningProfile.getExecPathString());
  }

  private String codesignCommand(
      Artifact provisioningProfile, Artifact entitlements, String appDir) {
    String fingerprintCommand =
        "/usr/libexec/PlistBuddy -c 'Print DeveloperCertificates:0' /dev/stdin <<< "
            + "$(" + extractPlistCommand(provisioningProfile) + ") | "
            + "openssl x509 -inform DER -noout -fingerprint | "
            + "cut -d= -f2 | sed -e 's#:##g'";
    return String.format(
        "/usr/bin/codesign --force --sign $(%s) --entitlements %s %s",
        fingerprintCommand,
        entitlements.getExecPathString(),
        appDir);
  }

  /**
   * Logic to access attributes required by application support. Attributes are required and
   * guaranteed to return a value or throw unless they are annotated with {@link Nullable} in which
   * case they can return {@code null} if no value is defined.
   */
  private static class Attributes {
    private final RuleContext ruleContext;

    private Attributes(RuleContext ruleContext) {
      this.ruleContext = ruleContext;
    }

    @Nullable
    String appIcon() {
      return stringAttribute("app_icon");
    }

    @Nullable
    String launchImage() {
      return stringAttribute("launch_image");
    }

    @Nullable
    Artifact provisioningProfile() {
      return ruleContext.getPrerequisiteArtifact("provisioning_profile", Mode.TARGET);
    }

    /**
     * Returns the value of the {@code families} attribute in a form that is more useful than a list
     * of strings. Returns an empty set for any invalid {@code families} attribute value, including
     * an empty list.
     */
    Set<TargetDeviceFamily> families() {
      List<String> rawFamilies = ruleContext.attributes().get("families", Type.STRING_LIST);
      try {
        return TargetDeviceFamily.fromNamesInRule(rawFamilies);
      } catch (InvalidFamilyNameException | RepeatedFamilyNameException e) {
        return ImmutableSet.of();
      }
    }

    @Nullable
    Artifact entitlements() {
      return ruleContext.getPrerequisiteArtifact("entitlements", Mode.TARGET);
    }

    NestedSet<? extends Artifact> dependentLinkedBinaries() {
      if (ruleContext.attributes().getAttributeDefinition("binary") == null) {
        return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
      }

      return ruleContext.getPrerequisite("binary", Mode.TARGET, ObjcProvider.class)
          .get(ObjcProvider.LINKED_BINARY);
    }

    FilesToRunProvider bundleMergeExecutable() {
      return checkNotNull(ruleContext.getExecutablePrerequisite("$bundlemerge", Mode.HOST));
    }

    Artifact iossim() {
      return checkNotNull(ruleContext.getPrerequisiteArtifact("$iossim", Mode.HOST));
    }

    Artifact runnerScriptTemplate() {
      return checkNotNull(
          ruleContext.getPrerequisiteArtifact("$runner_script_template", Mode.HOST));
    }

    String bundleId() {
      return checkNotNull(stringAttribute("bundle_id"));
    }

    @Nullable
    private String stringAttribute(String attribute) {
      String value = ruleContext.attributes().get(attribute, Type.STRING);
      return value.isEmpty() ? null : value;
    }
  }
}
