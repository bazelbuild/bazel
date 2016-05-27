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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromTemplates;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.Flag.USES_SWIFT;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ReleaseBundlingRule.APP_ICON_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ReleaseBundlingRule.DEBUG_ENTITLEMENTS_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ReleaseBundlingRule.EXTRA_ENTITLEMENTS_ATTR;
import static com.google.devtools.build.lib.rules.objc.TargetDeviceFamily.UI_DEVICE_FAMILY_VALUES;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.BuildInfo;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.BinaryFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Substitution;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.Attribute.SplitTransition;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.rules.apple.Platform;
import com.google.devtools.build.lib.rules.apple.Platform.PlatformType;
import com.google.devtools.build.lib.rules.objc.BundleSupport.ExtraActoolArgs;
import com.google.devtools.build.lib.rules.objc.Bundling.Builder;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.XcodeprojBuildSetting;

import com.dd.plist.NSArray;
import com.dd.plist.NSDictionary;
import com.dd.plist.NSObject;

import java.util.List;
import java.util.Map.Entry;

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
  static final String DEVICE_NO_PROVISIONING_PROFILE =
      "Provisioning profile must be set for device build";

  @VisibleForTesting
  static final String PROVISIONING_PROFILE_BUNDLE_FILE = "embedded.mobileprovision";
  @VisibleForTesting
  static final String APP_BUNDLE_DIR_FORMAT = "Payload/%s.app";
  @VisibleForTesting
  static final String XCTEST_BUNDLE_DIR_FORMAT = "Payload/%s.xctest";
  @VisibleForTesting
  static final String EXTENSION_BUNDLE_DIR_FORMAT = "PlugIns/%s.appex";
  @VisibleForTesting
  static final String FRAMEWORK_BUNDLE_DIR_FORMAT = "Frameworks/%s.framework";

  /**
   * Command string for "sed" that tries to extract the application version number from a larger
   * string. For example, from "foo_1.2.3_RC00" this would extract "1.2.3". This regex looks for
   * versions of the format "x.y" or "x.y.z", which may be preceded and/or followed by other text,
   * such as a project name or release candidate number.
   *
   * <p>This command also preserves double quotes around the string, if any.
   */
  private static final String EXTRACT_VERSION_NUMBER_SED_COMMAND =
      "s#\\(\"\\)\\{0,1\\}\\(.*_\\)\\{0,1\\}\\([0-9][0-9]*\\(\\.[0-9][0-9]*\\)\\{1,2\\}\\)"
      + "\\(_[^\"]*\\)\\{0,1\\}\\(\"\\)\\{0,1\\}#\\1\\3\\6#";

  private final Attributes attributes;
  private final BundleSupport bundleSupport;
  private final RuleContext ruleContext;
  private final Bundling bundling;
  private final ObjcProvider objcProvider;
  private final LinkedBinary linkedBinary;
  private final IntermediateArtifacts intermediateArtifacts;
  private final ReleaseBundling releaseBundling;

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
   * @param linkedBinary whether to look for a linked binary from this rule and dependencies or just
   *    the latter
   * @param bundleDirFormat format string representing the bundle's directory with a single
   *    placeholder for the target name (e.g. {@code "Payload/%s.app"})
   * @param bundleName name of the bundle, used with bundleDirFormat
   * @param bundleMinimumOsVersion the minimum OS version this bundle's plist should be generated
   *    for (<b>not</b> the minimum OS version its binary is compiled with, that needs to be set
   *    through the configuration)
   * @param releaseBundling the {@link ReleaseBundling} containing information for creating a
   *    releaseable bundle.
   */
  ReleaseBundlingSupport(
      RuleContext ruleContext,
      ObjcProvider objcProvider,
      LinkedBinary linkedBinary,
      String bundleDirFormat,
      String bundleName,
      DottedVersion bundleMinimumOsVersion,
      ReleaseBundling releaseBundling) {
    this.linkedBinary = linkedBinary;
    this.attributes = new Attributes(ruleContext);
    this.ruleContext = ruleContext;
    this.objcProvider = objcProvider;
    this.releaseBundling = releaseBundling;
    this.intermediateArtifacts = releaseBundling.getIntermediateArtifacts();
    this.bundling = bundling(ruleContext, objcProvider, bundleDirFormat, bundleName,
        bundleMinimumOsVersion);
    bundleSupport = new BundleSupport(ruleContext, bundling, extraActoolArgs());
  }

  /**
   * Creates a new application support within the given rule context.
   *
   * @param ruleContext context for the application-generating rule
   * @param objcProvider provider containing all dependencies' information as well as some of this
   *    rule's
   * @param linkedBinary whether to look for a linked binary from this rule and dependencies or just
   *    the latter
   * @param bundleDirFormat format string representing the bundle's directory with a single
   *    placeholder for the target name (e.g. {@code "Payload/%s.app"})
   * @param bundleName name of the bundle, used with bundleDirFormat
   */
  ReleaseBundlingSupport(
      RuleContext ruleContext,
      ObjcProvider objcProvider,
      LinkedBinary linkedBinary,
      String bundleDirFormat,
      String bundleName,
      DottedVersion bundleMinimumOsVersion) throws InterruptedException {
    this(
        ruleContext,
        objcProvider,
        linkedBinary,
        bundleDirFormat,
        bundleName,
        bundleMinimumOsVersion,
        ReleaseBundling.releaseBundling(ruleContext));
  }

  /**
   * Creates a new application support within the given rule context.
   *
   * {@code bundleName} defaults to label name
   *
   * @param ruleContext context for the application-generating rule
   * @param objcProvider provider containing all dependencies' information as well as some of this
   *    rule's
   * @param linkedBinary whether to look for a linked binary from this rule and dependencies or just
   *    the latter
   * @param bundleDirFormat format string representing the bundle's directory with a single
   *    placeholder for the target name (e.g. {@code "Payload/%s.app"})
   * @param bundleMinimumOsVersion the minimum OS version this bundle's plist should be generated
   *    for (<b>not</b> the minimum OS version its binary is compiled with, that needs to be set
   *    through the configuration)
   * @throws InterruptedException
   */
  ReleaseBundlingSupport(
      RuleContext ruleContext,
      ObjcProvider objcProvider,
      LinkedBinary linkedBinary,
      String bundleDirFormat,
      DottedVersion bundleMinimumOsVersion) throws InterruptedException {
    this(ruleContext, objcProvider, linkedBinary, bundleDirFormat, ruleContext.getLabel().getName(),
        bundleMinimumOsVersion);
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
      if (releaseBundling.getAppIcon() != null) {
        ruleContext.attributeError(APP_ICON_ATTR,
            String.format(NO_ASSET_CATALOG_ERROR_FORMAT, releaseBundling.getAppIcon()));
      }
      if (releaseBundling.getLaunchImage() != null) {
        ruleContext.attributeError("launch_image",
            String.format(NO_ASSET_CATALOG_ERROR_FORMAT, releaseBundling.getLaunchImage()));
      }
    }

    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);
    if (releaseBundling.getProvisioningProfile() == null
        && appleConfiguration.getBundlingPlatform() != Platform.IOS_SIMULATOR) {
      ruleContext.attributeError(releaseBundling.getProvisioningProfileAttrName(),
          DEVICE_NO_PROVISIONING_PROFILE);
    }

    return this;
  }

  /**
   * Validates that resources defined in this rule and its dependencies and written to this bundle
   * are legal.
   *
   * @return this release bundling support
   */
  ReleaseBundlingSupport validateResources() {
    bundleSupport.validate(objcProvider);
    return this;
  }

  /**
   * Registers actions required to build an application. This includes any
   * {@link BundleSupport#registerActions(ObjcProvider) bundle} and bundle merge actions, signing
   * this application if appropriate and combining several single-architecture binaries into one
   * multi-architecture binary.
   *
   * @param dsymOutputType the file type of the dSYM bundle to be generated
   *
   * @return this application support
   */
  ReleaseBundlingSupport registerActions(DsymOutputType dsymOutputType)
      throws InterruptedException {
    bundleSupport.registerActions(objcProvider);

    registerCombineArchitecturesAction();
    registerTransformAndCopyBreakpadFilesAction();
    registerCopyDsymFilesAction(dsymOutputType);
    registerCopyDsymPlistAction(dsymOutputType);
    registerCopyLinkmapFilesAction();
    registerSwiftStdlibActionsIfNecessary();

    registerEmbedLabelPlistAction();
    registerEnvironmentPlistAction();
    registerAutomaticPlistAction();

    if (releaseBundling.getLaunchStoryboard() != null) {
      registerLaunchStoryboardPlistAction();
    }

    registerBundleMergeActions();
    registerPostProcessAndSigningActions();

    return this;
  }

  private void registerEmbedLabelPlistAction() {
    Artifact buildInfo = Iterables.getOnlyElement(
        ruleContext.getBuildInfo(ObjcBuildInfoFactory.KEY));
    String generatedVersionPlistPath = getGeneratedVersionPlist().getShellEscapedExecPathString();
    String shellCommand = "VERSION=\"$("
        + "grep \"^" + BuildInfo.BUILD_EMBED_LABEL + "\" "
        + buildInfo.getShellEscapedExecPathString()
        + " | cut -d' ' -f2- | sed -e '" + EXTRACT_VERSION_NUMBER_SED_COMMAND + "' | "
        + "sed -e 's#\"#\\\"#g')\" && "
        + "cat >" + generatedVersionPlistPath + " <<EOF\n"
        + "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        + "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" "
        + "\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
        + "<plist version=\"1.0\">\n"
        + "<dict>\n"
        + "EOF\n"

            + "if [[ -n \"${VERSION}\" ]]; then\n"
            + "  for KEY in CFBundleVersion CFBundleShortVersionString; do\n"
            + "    echo \"  <key>${KEY}</key>\n\" >> "
            + generatedVersionPlistPath + "\n"
            + "    echo \"  <string>${VERSION}</string>\n\" >> "
            + generatedVersionPlistPath + "\n"
            + "  done\n"
            + "fi\n"

            + "cat >>" + generatedVersionPlistPath + " <<EOF\n"
            + "</dict>\n"
            + "</plist>\n"
            + "EOF\n";
    ruleContext.registerAction(new SpawnAction.Builder()
        .setMnemonic("ObjcVersionPlist")
        .setShellCommand(shellCommand)
        .addInput(buildInfo)
        .addOutput(getGeneratedVersionPlist())
        .build(ruleContext));
  }

  private void registerLaunchStoryboardPlistAction() {
    String launchStoryboard = releaseBundling.getLaunchStoryboard().getFilename();
    String launchStoryboardName = launchStoryboard.substring(0, launchStoryboard.lastIndexOf('.'));
    NSDictionary result = new NSDictionary();
    result.put("UILaunchStoryboardName", launchStoryboardName);
    String contents = result.toGnuStepASCIIPropertyList();
    ruleContext.registerAction(
        new FileWriteAction(
            ruleContext.getActionOwner(), getLaunchStoryboardPlist(), contents, false));
  }

  private void registerEnvironmentPlistAction() {
    AppleConfiguration configuration = ruleContext.getFragment(AppleConfiguration.class);
    // Generates a .plist that contains environment values (such as the SDK used to build, the Xcode
    // version, etc), which are parsed from various .plist files of the OS, namely Xcodes' and
    // Platforms' plists.
    // The resulting file is meant to be merged with the final bundle.
    String platformWithVersion =
        String.format(
            "%s%s",
            configuration.getBundlingPlatform().getLowerCaseNameInPlist(),
            configuration.getIosSdkVersion());
    ruleContext.registerAction(
        ObjcRuleClasses.spawnAppleEnvActionBuilder(ruleContext)
            .setMnemonic("EnvironmentPlist")
            .setExecutable(attributes.environmentPlist())
            .addArguments("--platform", platformWithVersion)
            .addArguments("--output", getGeneratedEnvironmentPlist().getExecPathString())
            .addOutput(getGeneratedEnvironmentPlist())
            .build(ruleContext));
  }

  private void registerAutomaticPlistAction() {
    ruleContext.registerAction(
        new FileWriteAction(
            ruleContext.getActionOwner(),
            getGeneratedAutomaticPlist(),
            automaticEntries().toGnuStepASCIIPropertyList(),
            /*makeExecutable=*/ false));
  }

  /**
   * Returns a map containing entries that should be added to the merged plist. These are usually
   * generated by Xcode automatically during the build process.
   */
  private NSDictionary automaticEntries() {
    List<Integer> uiDeviceFamily =
        TargetDeviceFamily.UI_DEVICE_FAMILY_VALUES.get(bundleSupport.targetDeviceFamilies());
    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);
    Platform platform = appleConfiguration.getBundlingPlatform();

    NSDictionary result = new NSDictionary();

    if (uiDeviceFamily != null) {
      result.put("UIDeviceFamily", uiDeviceFamily.toArray());
    }
    result.put("DTPlatformName", platform.getLowerCaseNameInPlist());
    result.put(
        "DTSDKName",
        platform.getLowerCaseNameInPlist() + appleConfiguration.getIosSdkVersion());
    result.put("CFBundleSupportedPlatforms", new NSArray(NSObject.wrap(platform.getNameInPlist())));
    result.put("MinimumOSVersion", bundling.getMinimumOsVersion().toString());

    return result;
  }

  /**
   * Registers all actions necessary to create a processed and signed IPA from the initial merged
   * IPA.
   *
   * <p>Includes user-provided actions to process IPA contents (via {@code ipa_post_processor}),
   * and signing actions if the IPA is being built for device architectures. If signing is necessary
   * also includes entitlements generation and processing actions.
   *
   * <p>Note that multiple "actions" on the IPA contents may be run in a single blaze action to
   * avoid excessive zipping/unzipping of IPA contents.
   */
  private void registerPostProcessAndSigningActions() throws InterruptedException {
    Artifact processedIpa = releaseBundling.getIpaArtifact();
    Artifact unprocessedIpa = intermediateArtifacts.unprocessedIpa();

    boolean processingNeeded = false;
    NestedSetBuilder<Artifact> inputs =
        NestedSetBuilder.<Artifact>stableOrder().add(unprocessedIpa);

    String actionCommandLine =
        "set -e && "
            + "t=$(mktemp -d \"${TMPDIR:-/tmp}/signing_intermediate.XXXXXX\") && "
            + "trap \"rm -rf ${t}\" EXIT && "
            // Get an absolute path since we need to cd into the temp directory for zip.
            + "signed_ipa=${PWD}/"
            + processedIpa.getShellEscapedExecPathString()
            + " && "
            + "/usr/bin/unzip -qq "
            + unprocessedIpa.getShellEscapedExecPathString()
            + " -d ${t} && ";

    FilesToRunProvider processor = attributes.ipaPostProcessor();
    if (processor != null) {
      processingNeeded = true;
      actionCommandLine += processor.getExecutable().getShellEscapedExecPathString() + " ${t} && ";
    }

    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);
    if (appleConfiguration.getBundlingPlatform() == Platform.IOS_DEVICE) {
      processingNeeded = true;
      registerEntitlementsActions();
      actionCommandLine += signingCommandLine();
      inputs.add(releaseBundling.getProvisioningProfile()).add(
          intermediateArtifacts.entitlements());
    }

    actionCommandLine += "cd ${t} && /usr/bin/zip -q -r \"${signed_ipa}\" .";

    if (processingNeeded) {
      SpawnAction.Builder processAction =
          ObjcRuleClasses.spawnBashOnDarwinActionBuilder(actionCommandLine)
              .setMnemonic("ObjcProcessIpa")
              .setProgressMessage("Processing iOS IPA: " + ruleContext.getLabel())
              .addTransitiveInputs(inputs.build())
              .addOutput(processedIpa);

      if (processor != null) {
        processAction.addTool(processor);
      }

      ruleContext.registerAction(processAction.build(ruleContext));
    } else {
      ruleContext.registerAction(
          new SymlinkAction(
              ruleContext.getActionOwner(), unprocessedIpa, processedIpa, "Processing IPA"));
    }
  }

  private String signingCommandLine() {
    // The order here is important. The innermost code must singed first.
    ImmutableList.Builder<String> dirsToSign = new ImmutableList.Builder<>();
    String bundleDir = ShellUtils.shellEscape(bundling.getBundleDir());

    // Explicitly sign the frameworks (raw .dylib files and .framework directories in Frameworks/).
    // Unfortunately the --deep option on codesign doesn't do this automatically.
    if (objcProvider.is(USES_SWIFT)
        || !objcProvider.get(ObjcProvider.DYNAMIC_FRAMEWORK_FILE).isEmpty()) {
      dirsToSign.add(bundleDir + "/Frameworks/*");
    }
    dirsToSign.add(bundleDir);

    StringBuilder codesignCommandLineBuilder = new StringBuilder();
    for (String dir : dirsToSign.build()) {
      codesignCommandLineBuilder.append(codesignCommand("${t}/" + dir)).append(" && ");
    }
    return codesignCommandLineBuilder.toString();
  }

  /**
   * Creates entitlement actions such that an entitlements file is generated in
   * {@link IntermediateArtifacts#entitlements()} which can be used for signing in this bundle.
   *
   * <p>Entitlements are generated based on a plist-format entitlements file passed to this bundle's
   * {@code entitlements} attribute or, if that is not set, entitlements extracted from the provided
   * mobile provisioning profile. The team prefix is extracted from the provisioning profile and
   * the following substitutions performed (assuming the prefix extracted was {@code PREFIX}):
   * <ol>
   *   <li>"PREFIX.*" -> "PREFIX.BUNDLE_ID" (where BUNDLE_ID is this bundle's id)
   *   <li>"$(AppIdentifierPrefix)" -> "PREFIX."
   *   <li>"$(CFBundleIdentifier)" -> "BUNDLE_ID" (where BUNDLE_ID is this bundle's id)
   * </ol>
   *
   * <p>Finally, if an entitlements file was provided via {@code --extra_entitlements} it is merged
   * into the substituted entitlements.
   */
  private void registerEntitlementsActions() {
    Artifact teamPrefixFile =
        intermediateArtifacts.appendExtensionForEntitlementArtifact(".team_prefix_file");
    registerExtractTeamPrefixAction(teamPrefixFile);

    Artifact entitlementsNeedingSubstitution = releaseBundling.getEntitlements();
    if (entitlementsNeedingSubstitution == null) {
      entitlementsNeedingSubstitution =
          intermediateArtifacts.appendExtensionForEntitlementArtifact(
              ".entitlements_with_variables");
      registerExtractEntitlementsAction(entitlementsNeedingSubstitution);
    }

    Artifact substitutedEntitlements = intermediateArtifacts.entitlements();
    if (attributes.extraEntitlements() != null || includeDebugEntitlements()) {
      substitutedEntitlements =
          intermediateArtifacts.appendExtensionForEntitlementArtifact(".substituted");

      NestedSetBuilder<Artifact> entitlements =
          NestedSetBuilder.<Artifact>stableOrder().add(substitutedEntitlements);
      if (attributes.extraEntitlements() != null) {
        entitlements.add(attributes.extraEntitlements());
      }
      if (includeDebugEntitlements()) {
        entitlements.add(attributes.deviceDebugEntitlements());
      }

      registerMergeEntitlementsAction(entitlements.build());
    }

    registerEntitlementsVariableSubstitutionAction(
        entitlementsNeedingSubstitution, teamPrefixFile, substitutedEntitlements);
  }

  private boolean includeDebugEntitlements() {
    return attributes.deviceDebugEntitlements() != null
        && ruleContext.getConfiguration().getCompilationMode() != CompilationMode.OPT
        && ObjcRuleClasses.objcConfiguration(ruleContext).useDeviceDebugEntitlements();
  }

  private void registerMergeEntitlementsAction(NestedSet<Artifact> entitlements) {
    PlMergeControlBytes controlBytes =
        PlMergeControlBytes.fromPlists(
            entitlements,
            intermediateArtifacts.entitlements(),
            PlMergeControlBytes.OutputFormat.XML);

    Artifact plMergeControlArtifact = ObjcRuleClasses.artifactByAppendingToBaseName(ruleContext,
        artifactName(".merge-entitlements-control"));

    ruleContext.registerAction(
        new BinaryFileWriteAction(
            ruleContext.getActionOwner(),
            plMergeControlArtifact,
            controlBytes,
            /*makeExecutable=*/ false));

    ruleContext.registerAction(
        new SpawnAction.Builder()
            .setMnemonic("MergeEntitlementsFiles")
            .setExecutable(attributes.plmerge())
            .addArgument("--control")
            .addInputArgument(plMergeControlArtifact)
            .addTransitiveInputs(entitlements)
            .addOutput(intermediateArtifacts.entitlements())
            .build(ruleContext));
  }

  /**
   * Adds bundle- and application-related settings to the given Xcode provider builder.
   *
   * @return this application support
   */
  ReleaseBundlingSupport addXcodeSettings(XcodeProvider.Builder xcodeProviderBuilder) {
    bundleSupport.addXcodeSettings(xcodeProviderBuilder);
    // Add application-related Xcode build settings to the main target only. The companion library
    // target does not need them.
    xcodeProviderBuilder.addMainTargetXcodeprojBuildSettings(buildSettings());

    return this;
  }

  /**
   * Adds any files to the given nested set builder that should be built if this application is the
   * top level target in a blaze invocation.
   *
   * @param filesToBuild a collection of files to be built, where new artifacts to be built are
   *     going to be placed
   * @param dsymOutputType the file type of the dSYM bundle to be built
   *
   * @return this application support
   */
  ReleaseBundlingSupport addFilesToBuild(
      NestedSetBuilder<Artifact> filesToBuild, DsymOutputType dsymOutputType)
      throws InterruptedException {
    NestedSetBuilder<Artifact> debugSymbolBuilder = NestedSetBuilder.<Artifact>stableOrder();

    for (Artifact linkmapFile : getLinkmapFiles().values()) {
      filesToBuild.add(linkmapFile);
    }

    if (ObjcRuleClasses.objcConfiguration(ruleContext).generateDebugSymbols()
        || ObjcRuleClasses.objcConfiguration(ruleContext).generateDsym()) {
      filesToBuild.addAll(getDsymFiles(dsymOutputType).values());

      // TODO(bazel-team): Remove the 'if' when the objc_binary rule does not generate a bundle any
      // more. The reason this 'if' is here is because the plist is obtained from the ObjcProvider.
      // Since objc_binary is the rule that adds this file to the provider, and not before, when
      // running this the provider does not have the plist yet. This gets called again when running
      // the *_application targets, and since they depend on objc_binaries, the provider has the
      // files configured. When objc_binary stops bundling ipas as output, the bundling methods will
      // only get called by *_application rules, with the plist configured in the provider.
      Artifact cpuPlist = getAnyCpuSpecificDsymPlist();
      if (cpuPlist != null) {
        filesToBuild.add(intermediateArtifacts.dsymPlist(dsymOutputType));
      }

      if (linkedBinary == LinkedBinary.LOCAL_AND_DEPENDENCIES) {
        debugSymbolBuilder
            .add(intermediateArtifacts.dsymPlist(dsymOutputType))
            .add(intermediateArtifacts.dsymSymbol(dsymOutputType));
      }
    }

    if (ObjcRuleClasses.objcConfiguration(ruleContext).generateDebugSymbols()) {
      filesToBuild.addAll(getBreakpadFiles().values());

      if (linkedBinary == LinkedBinary.LOCAL_AND_DEPENDENCIES) {
        debugSymbolBuilder.add(intermediateArtifacts.breakpadSym());
      }
    }

    filesToBuild
        .add(releaseBundling.getIpaArtifact())
        .addTransitive(debugSymbolBuilder.build())
        .addTransitive(objcProvider.get(ObjcProvider.EXPORTED_DEBUG_ARTIFACTS));

    return this;
  }

  /**
   * Adds dSYM artifacts (plist, arch-speficic binaries) to the {@link ObjcProvider} for export.
   */
  public void addExportedDebugArtifacts(
      ObjcProvider.Builder objcBuilder, DsymOutputType dsymOutputType) {
    if (ObjcRuleClasses.objcConfiguration(ruleContext).generateDebugSymbols()
        || ObjcRuleClasses.objcConfiguration(ruleContext).generateDsym()) {
      objcBuilder
          .addAll(ObjcProvider.EXPORTED_DEBUG_ARTIFACTS, getDsymFiles(dsymOutputType).values())
          .add(
              ObjcProvider.EXPORTED_DEBUG_ARTIFACTS,
              intermediateArtifacts.dsymPlist(dsymOutputType));
    }
  }

  /**
   * Creates the {@link XcTestAppProvider} that can be used if this application is used as an
   * {@code xctest_app}.
   */
  XcTestAppProvider xcTestAppProvider() throws InterruptedException {
    // We want access to #import-able things from our test rig's dependency graph, but we don't
    // want to link anything since that stuff is shared automatically by way of the
    // -bundle_loader linker flag.
    // TODO(bazel-team): Handle the FRAMEWORK_DIR key properly. We probably want to add it to
    // framework search paths, but not actually link it with the -framework flag.
    ObjcProvider partialObjcProvider =
        new ObjcProvider.Builder()
            .addTransitiveAndPropagate(ObjcProvider.HEADER, objcProvider)
            .addTransitiveAndPropagate(ObjcProvider.INCLUDE, objcProvider)
            .addTransitiveAndPropagate(ObjcProvider.DEFINE, objcProvider)
            .addTransitiveAndPropagate(ObjcProvider.SDK_DYLIB, objcProvider)
            .addTransitiveAndPropagate(ObjcProvider.SDK_FRAMEWORK, objcProvider)
            .addTransitiveAndPropagate(ObjcProvider.SOURCE, objcProvider)
            .addTransitiveAndPropagate(ObjcProvider.WEAK_SDK_FRAMEWORK, objcProvider)
            .addTransitiveAndPropagate(ObjcProvider.STATIC_FRAMEWORK_FILE, objcProvider)
            .addTransitiveAndPropagate(ObjcProvider.DYNAMIC_FRAMEWORK_FILE, objcProvider)
            .addTransitiveAndPropagate(
                ObjcProvider.FRAMEWORK_SEARCH_PATH_ONLY,
                objcProvider.get(ObjcProvider.FRAMEWORK_DIR))
            .build();
    return new XcTestAppProvider(
        intermediateArtifacts.combinedArchitectureBinary(),
        releaseBundling.getIpaArtifact(),
        partialObjcProvider);
  }

  /**
   * Registers an action to generate a runner script based on a template.
   */
  ReleaseBundlingSupport registerGenerateRunnerScriptAction(Artifact runnerScript,
      Artifact ipaInput) {
    ObjcConfiguration objcConfiguration = ObjcRuleClasses.objcConfiguration(ruleContext);
    String escapedSimDevice = ShellUtils.shellEscape(objcConfiguration.getIosSimulatorDevice());
    String escapedSdkVersion =
        ShellUtils.shellEscape(objcConfiguration.getIosSimulatorVersion().toString());
    ImmutableList<Substitution> substitutions = ImmutableList.of(
        Substitution.of("%app_name%", ruleContext.getLabel().getName()),
        Substitution.of("%ipa_file%", ipaInput.getRunfilesPathString()),
        Substitution.of("%sim_device%", escapedSimDevice),
        Substitution.of("%sdk_version%", escapedSdkVersion),
        Substitution.of("%iossim%", attributes.iossim().getRunfilesPathString()),
        Substitution.of("%std_redirect_dylib_path%",
            attributes.stdRedirectDylib().getRunfilesPathString()));

    ruleContext.registerAction(
        new TemplateExpansionAction(ruleContext.getActionOwner(), attributes.runnerScriptTemplate(),
            runnerScript, substitutions, true));
    return this;
  }

  /**
   * Returns a {@link RunfilesSupport} that uses the provided runner script as the executable.
   */
  RunfilesSupport runfilesSupport(Artifact runnerScript) throws InterruptedException {
    Runfiles runfiles = new Runfiles.Builder(
        ruleContext.getWorkspaceName(), ruleContext.getConfiguration().legacyExternalRunfiles())
        .addArtifact(releaseBundling.getIpaArtifact())
        .addArtifact(runnerScript)
        .addArtifact(attributes.iossim())
        .build();
    return RunfilesSupport.withExecutable(ruleContext, runfiles, runnerScript);
  }

  private ExtraActoolArgs extraActoolArgs() {
    ImmutableList.Builder<String> extraArgs = ImmutableList.builder();
    if (releaseBundling.getAppIcon() != null) {
      extraArgs.add("--app-icon", releaseBundling.getAppIcon());
    }
    if (releaseBundling.getLaunchImage() != null) {
      extraArgs.add("--launch-image", releaseBundling.getLaunchImage());
    }
    return new ExtraActoolArgs(extraArgs.build());
  }

  private Bundling bundling(
      RuleContext ruleContext,
      ObjcProvider objcProvider,
      String bundleDirFormat,
      String bundleName,
      DottedVersion minimumOsVersion) {
    ImmutableList<BundleableFile> extraBundleFiles;
    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);
    if (appleConfiguration.getBundlingPlatform() == Platform.IOS_DEVICE) {
      extraBundleFiles = ImmutableList.of(new BundleableFile(
          releaseBundling.getProvisioningProfile(), PROVISIONING_PROFILE_BUNDLE_FILE));
    } else {
      extraBundleFiles = ImmutableList.of();
    }

    Bundling.Builder bundling =
        new Builder()
            .setName(bundleName)
            // Architecture that determines which nested bundles are kept.
            .setArchitecture(appleConfiguration.getDependencySingleArchitecture())
            .setBundleDirFormat(bundleDirFormat)
            .addExtraBundleFiles(extraBundleFiles)
            .setObjcProvider(objcProvider)
            .setIntermediateArtifacts(intermediateArtifacts)
            .setPrimaryBundleId(releaseBundling.getPrimaryBundleId())
            .setFallbackBundleId(releaseBundling.getFallbackBundleId())
            .setMinimumOsVersion(minimumOsVersion)
            .setArtifactPrefix(releaseBundling.getArtifactPrefix())
            .setTargetDeviceFamilies(releaseBundling.getTargetDeviceFamilies());

    // Add plists from rule first.
    if (releaseBundling.getInfoPlistsFromRule() != null) {
      bundling.addInfoplistInputs(releaseBundling.getInfoPlistsFromRule());
    } else {
      bundling.addInfoplistInputFromRule(ruleContext);
    }

    // Add generated plists next so that generated values can override the default values in the
    // plists from rule.
    bundling.setAutomaticEntriesInfoplistInput(getGeneratedAutomaticPlist())
        .addInfoplistInput(getGeneratedVersionPlist())
        .addInfoplistInput(getGeneratedEnvironmentPlist())
        .addInfoplistInputs(releaseBundling.getInfoplistInputs());

    if (releaseBundling.getLaunchStoryboard() != null) {
      bundling.addInfoplistInput(getLaunchStoryboardPlist());
    }

    return bundling.build();
  }

  private void registerCombineArchitecturesAction() {
    // Skip combining binaries when building for watch as there is only one stub binary and it
    // it should not be corrupted when combining.
    if (bundleSupport.isBuildingForWatch()) {
      return;
    }

    Artifact resultingLinkedBinary = intermediateArtifacts.combinedArchitectureBinary();
    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);

    new LipoSupport(ruleContext).registerCombineArchitecturesAction(linkedBinaries(),
        resultingLinkedBinary, appleConfiguration.getPlatform(PlatformType.IOS));
  }

  private NestedSet<Artifact> linkedBinaries() {
    NestedSetBuilder<Artifact> linkedBinariesBuilder = NestedSetBuilder.<Artifact>stableOrder()
        .addTransitive(attributes.dependentLinkedBinaries());
    if (linkedBinary == LinkedBinary.LOCAL_AND_DEPENDENCIES) {
      linkedBinariesBuilder.add(intermediateArtifacts.strippedSingleArchitectureBinary());
    }
    return linkedBinariesBuilder.build();
  }

  /** Returns this target's Xcode build settings. */
  private Iterable<XcodeprojBuildSetting> buildSettings() {
    ImmutableList.Builder<XcodeprojBuildSetting> buildSettings = new ImmutableList.Builder<>();
    if (releaseBundling.getAppIcon() != null) {
      buildSettings.add(XcodeprojBuildSetting.newBuilder()
          .setName("ASSETCATALOG_COMPILER_APPICON_NAME")
          .setValue(releaseBundling.getAppIcon())
          .build());
    }
    if (releaseBundling.getLaunchImage() != null) {
      buildSettings.add(XcodeprojBuildSetting.newBuilder()
          .setName("ASSETCATALOG_COMPILER_LAUNCHIMAGE_NAME")
          .setValue(releaseBundling.getLaunchImage())
          .build());
    }

    // Convert names to a sequence containing "1" and/or "2" for iPhone and iPad, respectively.
    ImmutableSet<TargetDeviceFamily> families;
    if (bundleSupport.isBuildingForWatch()) {
      families = ImmutableSet.of(TargetDeviceFamily.WATCH);
    } else {
      families = bundleSupport.targetDeviceFamilies();
    }
    Iterable<Integer> familyIndexes =
        families.isEmpty() ? ImmutableList.<Integer>of() : UI_DEVICE_FAMILY_VALUES.get(families);
    buildSettings.add(XcodeprojBuildSetting.newBuilder()
        .setName("TARGETED_DEVICE_FAMILY")
        .setValue(Joiner.on(',').join(familyIndexes))
        .build());

    Artifact entitlements = releaseBundling.getEntitlements();
    if (entitlements != null) {
      buildSettings.add(XcodeprojBuildSetting.newBuilder()
          .setName("CODE_SIGN_ENTITLEMENTS")
          .setValue("$(WORKSPACE_ROOT)/" + entitlements.getExecPathString())
          .build());
    }

    return buildSettings.build();
  }

  private void registerBundleMergeActions() {
    Artifact bundleMergeControlArtifact = ObjcRuleClasses.artifactByAppendingToBaseName(ruleContext,
        artifactName(".ipa-control"));

    BundleMergeControlBytes controlBytes =
        new BundleMergeControlBytes(
            bundling,
            intermediateArtifacts.unprocessedIpa(),
            ruleContext.getFragment(AppleConfiguration.class));

    ruleContext.registerAction(
        new BinaryFileWriteAction(
            ruleContext.getActionOwner(), bundleMergeControlArtifact, controlBytes,
            /*makeExecutable=*/false));

    ruleContext.registerAction(
        new SpawnAction.Builder()
            .setMnemonic("IosBundle")
            .setProgressMessage("Bundling iOS application: " + ruleContext.getLabel())
            .setExecutable(attributes.bundleMergeExecutable())
            .addInputArgument(bundleMergeControlArtifact)
            .addTransitiveInputs(bundling.getBundleContentArtifacts())
            .addOutput(intermediateArtifacts.unprocessedIpa())
            .build(ruleContext));
  }

  /**
   * Registers the actions that transform and copy the breakpad files from the CPU-specific binaries
   * that are part of this application. There are two steps involved: 1) The breakpad files have to
   * be renamed to include their corresponding CPU architecture as a suffix. 2) The first line of
   * the breakpad file has to be rewritten, as it has to include the name of the application instead
   * of the name of the binary artifact.
   *
   * <p>Example:<br>
   * The ios_application "PrenotCalculator" is specified to use "PrenotCalculatorBinary" as its
   * binary. Assuming that the application is built for armv7 and arm64 CPUs, in the build process
   * two binaries with a corresponding breakpad file each will be built:
   *
   * <pre>blaze-out/xyz-crosstool-ios-arm64/.../PrenotCalculatorBinary_bin
   * blaze-out/xyz-crosstool-ios-arm64/.../PrenotCalculatorBinary.breakpad
   * blaze-out/xyz-crosstool-ios-armv7/.../PrenotCalculatorBinary_bin
   * blaze-out/xyz-crosstool-ios-armv7/.../PrenotCalculatorBinary.breakpad</pre>
   *
   * <p>The first line of the breakpad files will look like this:
   * <pre>MODULE mac arm64 8A7A2DDD28E83E27B339E63631ADBEF30 PrenotCalculatorBinary_bin</pre>
   *
   * <p>For our application, we have to transform & copy these breakpad files like this:
   * <pre>$ head -n1 blaze-bin/.../PrenotCalculator_arm64.breakpad
   * MODULE mac arm64 8A7A2DDD28E83E27B339E63631ADBEF30 PrenotCalculator</pre>
   */
  private void registerTransformAndCopyBreakpadFilesAction() {
    for (Entry<Artifact, Artifact> breakpadFiles : getBreakpadFiles().entrySet()) {
      ruleContext.registerAction(
          new SpawnAction.Builder().setMnemonic("CopyBreakpadFile")
              .setShellCommand(String.format(
                  // This sed command replaces the last word of the first line with the application
                  // name.
                  "sed \"1 s/^\\(MODULE \\w* \\w* \\w*\\).*$/\\1 %s/\" < %s > %s",
                  ruleContext.getLabel().getName(), breakpadFiles.getKey().getExecPathString(),
                  breakpadFiles.getValue().getExecPathString()))
              .addInput(breakpadFiles.getKey())
              .addOutput(breakpadFiles.getValue())
              .build(ruleContext));
    }
  }

  private void registerCopyLinkmapFilesAction() {
   for (Entry<Artifact, Artifact> linkmapFile : getLinkmapFiles().entrySet()) {
      ruleContext.registerAction(
          new SymlinkAction(ruleContext.getActionOwner(), linkmapFile.getKey(),
              linkmapFile.getValue(), String.format("Copying Linkmap %s",
              linkmapFile.getValue().prettyPrint())));
   }
  }

  /**
   * Registers the actions that copy the debug symbol files from the CPU-specific binaries that are
   * part of this application. The only one step executed is that he dsym files have to be renamed
   * to include their corresponding CPU architecture as a suffix.
   *
   * @param dsymOutputType the file type of the dSYM bundle to be copied
   */
  private void registerCopyDsymFilesAction(DsymOutputType dsymOutputType) {
    for (Entry<Artifact, Artifact> dsymFiles : getDsymFiles(dsymOutputType).entrySet()) {
      ruleContext.registerAction(
          new SymlinkAction(
              ruleContext.getActionOwner(),
              dsymFiles.getKey(),
              dsymFiles.getValue(),
              "Symlinking dSYM files"));
    }
  }

  /**
   * Registers the action that copies the debug symbol plist from the binary.
   *
   * @param dsymOutputType the file type of the dSYM bundle to be copied
   */
  private void registerCopyDsymPlistAction(DsymOutputType dsymOutputType) {
    Artifact dsymPlist = getAnyCpuSpecificDsymPlist();
    if (dsymPlist != null) {
      ruleContext.registerAction(
          new SymlinkAction(
              ruleContext.getActionOwner(),
              dsymPlist,
              intermediateArtifacts.dsymPlist(dsymOutputType),
              "Symlinking dSYM plist"));
    }
  }

  /**
   * Returns a map of input breakpad artifacts from the CPU-specific binaries built for this
   * ios_application to the new output breakpad artifacts.
   */
  private ImmutableMap<Artifact, Artifact> getBreakpadFiles() {
    ImmutableMap.Builder<Artifact, Artifact> results = ImmutableMap.builder();
    for (Entry<String, Artifact> breakpadFile : attributes.cpuSpecificBreakpadFiles().entrySet()) {
      Artifact destBreakpad = intermediateArtifacts.breakpadSym(breakpadFile.getKey());
      results.put(breakpadFile.getValue(), destBreakpad);
    }
    return results.build();
  }

  /**
   * Returns a map of input dsym artifacts from the CPU-specific binaries built for this
   * ios_application to the new output dsym artifacts.
   *
   * @param dsymOutputType the file type of the dSYM bundle to be generated
   */
  private ImmutableMap<Artifact, Artifact> getDsymFiles(DsymOutputType dsymOutputType) {
    ImmutableMap.Builder<Artifact, Artifact> results = ImmutableMap.builder();
    for (Entry<String, Artifact> dsymFile : attributes.cpuSpecificDsymFiles().entrySet()) {
      Artifact destDsym = intermediateArtifacts.dsymSymbol(dsymOutputType, dsymFile.getKey());
      results.put(dsymFile.getValue(), destDsym);
    }
    return results.build();
  }

  /**
   * Returns any available CPU specific dSYM plist file.
   */
  @Nullable
  private Artifact getAnyCpuSpecificDsymPlist() {
    for (Artifact dsymPlist : attributes.cpuSpecificDsymPlists().values()) {
      // The plist files generated by the dsym tool are all equal, and don't really have any
      // useful information. For now, just retrieving any one is OK, but ideally all of them should
      // be merged.
      return dsymPlist;
    }
    return null;
  }

  /**
   * Returns a map of input linkmap artifacts from the CPU-specific binaries built for this
   * ios_application to the new output linkmap artifacts.
   */
  private ImmutableMap<Artifact, Artifact> getLinkmapFiles() {
    ImmutableMap.Builder<Artifact, Artifact> results = ImmutableMap.builder();
    for (Entry<String, Artifact> linkmapFile : attributes.cpuSpecificLinkmapFiles().entrySet()) {
      Artifact destLinkMap = intermediateArtifacts.linkmap(linkmapFile.getKey());
      results.put(linkmapFile.getValue(), destLinkMap);
    }
    return results.build();
  }

  private void registerExtractTeamPrefixAction(Artifact teamPrefixFile) {
    String shellCommand = "set -e && "
        + "PLIST=$(mktemp -t teamprefix.plist) && trap \"rm ${PLIST}\" EXIT && "
        + extractPlistCommand(releaseBundling.getProvisioningProfile()) + " > ${PLIST} && "
        + "/usr/libexec/PlistBuddy -c 'Print ApplicationIdentifierPrefix:0' ${PLIST} > "
        + teamPrefixFile.getShellEscapedExecPathString();
    ruleContext.registerAction(
        ObjcRuleClasses.spawnBashOnDarwinActionBuilder(shellCommand)
            .setMnemonic("ExtractIosTeamPrefix")
            .addInput(releaseBundling.getProvisioningProfile())
            .addOutput(teamPrefixFile)
            .build(ruleContext));
  }

  private ReleaseBundlingSupport registerExtractEntitlementsAction(Artifact entitlements) {
    // See Apple Glossary (http://goo.gl/EkhXOb)
    // An Application Identifier is constructed as: TeamID.BundleID
    // TeamID is extracted from the provisioning profile.
    // BundleID consists of a reverse-DNS string to identify the app, where the last component
    // is the application name, and is specified as an attribute.
    String shellCommand = "set -e && "
        + "PLIST=$(mktemp -t entitlements.plist) && trap \"rm ${PLIST}\" EXIT && "
        + extractPlistCommand(releaseBundling.getProvisioningProfile()) + " > ${PLIST} && "
        + "/usr/libexec/PlistBuddy -x -c 'Print Entitlements' ${PLIST} > "
        + entitlements.getShellEscapedExecPathString();
    ruleContext.registerAction(
        ObjcRuleClasses.spawnBashOnDarwinActionBuilder(shellCommand)
            .setMnemonic("ExtractIosEntitlements")
            .setProgressMessage("Extracting entitlements: " + ruleContext.getLabel())
            .addInput(releaseBundling.getProvisioningProfile())
            .addOutput(entitlements)
            .build(ruleContext));

    return this;
  }

  private void registerEntitlementsVariableSubstitutionAction(
      Artifact inputEntitlements, Artifact prefix, Artifact substitutedEntitlements) {
    String escapedBundleId = ShellUtils.shellEscape(releaseBundling.getBundleId());
    String shellCommand =
        "set -e && "
            + "PREFIX=\"$(cat "
            + prefix.getShellEscapedExecPathString()
            + ")\" && "
            + "sed "
            // Replace .* from default entitlements file with bundle ID where suitable.
            + "-e \"s#${PREFIX}\\.\\*#${PREFIX}."
            + escapedBundleId
            + "#g\" "

            // Replace some variables that people put in their own entitlements files
            + "-e \"s#\\$(AppIdentifierPrefix)#${PREFIX}.#g\" "
            + "-e \"s#\\$(CFBundleIdentifier)#"
            + escapedBundleId
            + "#g\" "
            + inputEntitlements.getShellEscapedExecPathString()
            + " > "
            + substitutedEntitlements.getShellEscapedExecPathString();
    ruleContext.registerAction(
        new SpawnAction.Builder()
            .setMnemonic("SubstituteIosEntitlements")
            .setShellCommand(shellCommand)
            .addInput(inputEntitlements)
            .addInput(prefix)
            .addOutput(substitutedEntitlements)
            .build(ruleContext));
  }

  /** Registers an action to copy Swift standard library dylibs into app bundle. */
  private void registerSwiftStdlibActionsIfNecessary() {
    if (!objcProvider.is(USES_SWIFT)) {
      return;
    }

    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);

    CustomCommandLine.Builder commandLine = CustomCommandLine.builder()
        .addPath(intermediateArtifacts.swiftFrameworksFileZip().getExecPath())
        .add("--platform").add(AppleToolchain.swiftPlatform(appleConfiguration))
        .addExecPath("--scan-executable", intermediateArtifacts.combinedArchitectureBinary());

    ruleContext.registerAction(
        ObjcRuleClasses.spawnAppleEnvActionBuilder(ruleContext)
            .setMnemonic("SwiftStdlibCopy")
            .setExecutable(attributes.swiftStdlibToolWrapper())
            .setCommandLine(commandLine.build())
            .addOutput(intermediateArtifacts.swiftFrameworksFileZip())
            .addInput(intermediateArtifacts.combinedArchitectureBinary())
            .build(ruleContext));
  }

  private String extractPlistCommand(Artifact provisioningProfile) {
    return "security cms -D -i " + ShellUtils.shellEscape(provisioningProfile.getExecPathString());
  }

  private String codesignCommand(String appDir) {
    String signingCertName = ObjcRuleClasses.objcConfiguration(ruleContext).getSigningCertName();
    Artifact entitlements = intermediateArtifacts.entitlements();

    final String identity;
    if (signingCertName != null) {
      identity = '"' + signingCertName + '"';
    } else {
      // Extracts an identity hash from the configured provisioning profile. Note that this will use
      // the first certificate identity in the profile, regardless of how many identities are
      // configured in it (DeveloperCertificates:0).
      identity =
          "$(PLIST=$(mktemp -t cert.plist) && trap \"rm ${PLIST}\" EXIT && "
              + extractPlistCommand(releaseBundling.getProvisioningProfile())
              + " > ${PLIST} && "
              + "/usr/libexec/PlistBuddy -c 'Print DeveloperCertificates:0' ${PLIST} | "
              + "openssl x509 -inform DER -noout -fingerprint | "
              + "cut -d= -f2 | sed -e 's#:##g')";
    }

    return String.format(
        "/usr/bin/codesign --force --sign %s --entitlements %s %s",
        identity,
        entitlements.getShellEscapedExecPathString(),
        appDir);
  }

  private Artifact getGeneratedVersionPlist() {
    return ruleContext.getRelatedArtifact(
        ruleContext.getUniqueDirectory("plists"), artifactName("-version.plist"));
  }

  private Artifact getGeneratedEnvironmentPlist() {
    return ruleContext.getRelatedArtifact(
        ruleContext.getUniqueDirectory("plists"), artifactName("-environment.plist"));
  }

  private Artifact getGeneratedAutomaticPlist() {
    return ruleContext.getRelatedArtifact(
        ruleContext.getUniqueDirectory("plists"), artifactName("-automatic.plist"));
  }

  private Artifact getLaunchStoryboardPlist() {
    return ruleContext.getRelatedArtifact(
        ruleContext.getUniqueDirectory("plists"), artifactName("-launchstoryboard.plist"));
  }

  /**
   * Returns artifact name prefixed with prefix given in {@link ReleaseBundling} if available.
   * This helps in creating unique artifact name when multiple bundles are created with a different
   * name than the target name.
   */
  private String artifactName(String artifactName) {
    if (releaseBundling.getArtifactPrefix() != null) {
      return String.format("-%s%s", releaseBundling.getArtifactPrefix(), artifactName);
    }
    return artifactName;
  }

  /**
   * Logic to access attributes to access tools required by application support.
   * Attributes are required and guaranteed to return a value or throw unless they are annotated
   * with {@link Nullable} in which case they can return {@code null} if no value is defined.
   */
  private static class Attributes {
    private final RuleContext ruleContext;

    private Attributes(RuleContext ruleContext) {
      this.ruleContext = ruleContext;
    }

    /**
     * Returns this target's user-specified {@code ipa_post_processor} or null if not present.
     */
    @Nullable
    FilesToRunProvider ipaPostProcessor() {
      if (!ruleContext.attributes().has("ipa_post_processor", BuildType.LABEL)) {
        return null;
      }
      return ruleContext.getExecutablePrerequisite("ipa_post_processor", Mode.TARGET);
    }

    NestedSet<? extends Artifact> dependentLinkedBinaries() {
      if (ruleContext.attributes().getAttributeDefinition("binary") == null) {
        return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
      }

      NestedSetBuilder<Artifact> linkedBinaries = NestedSetBuilder.stableOrder();
      for (ObjcProvider provider
          : ruleContext.getPrerequisites("binary", Mode.DONT_CHECK, ObjcProvider.class)) {
        linkedBinaries.addTransitive(provider.get(ObjcProvider.LINKED_BINARY));
      }

      return linkedBinaries.build();
    }

    FilesToRunProvider bundleMergeExecutable() {
      return checkNotNull(ruleContext.getExecutablePrerequisite("$bundlemerge", Mode.HOST));
    }

    /**
     * Returns a reference to the plmerge executable.
     */
    FilesToRunProvider plmerge() {
      return ruleContext.getExecutablePrerequisite("$plmerge", Mode.HOST);
    }

    Artifact iossim() {
      return checkNotNull(ruleContext.getPrerequisiteArtifact("$iossim", Mode.HOST));
    }

    Artifact stdRedirectDylib() {
      return checkNotNull(ruleContext.getPrerequisiteArtifact("$std_redirect_dylib", Mode.HOST));
    }

    Artifact runnerScriptTemplate() {
      return checkNotNull(
          ruleContext.getPrerequisiteArtifact("$runner_script_template", Mode.HOST));
    }

    /** Returns the location of the swiftstdlibtoolwrapper. */
    FilesToRunProvider swiftStdlibToolWrapper() {
      return ruleContext.getExecutablePrerequisite("$swiftstdlibtoolwrapper", Mode.HOST);
    }

    /**
     * Returns the location of the environment_plist.
     */
    FilesToRunProvider environmentPlist() {
      return ruleContext.getExecutablePrerequisite("$environment_plist", Mode.HOST);
    }

    /**
     * Returns a plist specified by the user via {@code --extra_entitlements} or {@code null}.
     */
    @Nullable
    Artifact extraEntitlements() {
      if (ruleContext.attributes().getAttributeDefinition(EXTRA_ENTITLEMENTS_ATTR) == null) {
        return null;
      }
      return ruleContext.getPrerequisiteArtifact(EXTRA_ENTITLEMENTS_ATTR, Mode.HOST);
    }

    /**
     * Returns a plist containing entitlements that allow the signed IPA to be debugged.
     */
    @Nullable
    Artifact deviceDebugEntitlements() {
      if (ruleContext.attributes().getAttributeDefinition(DEBUG_ENTITLEMENTS_ATTR) == null) {
        return null;
      }
      return ruleContext.getPrerequisiteArtifact(DEBUG_ENTITLEMENTS_ATTR, Mode.HOST);
    }

    ImmutableMap<String, Artifact> cpuSpecificBreakpadFiles() {
      return cpuSpecificArtifacts(ObjcProvider.BREAKPAD_FILE);
    }

    ImmutableMap<String, Artifact> cpuSpecificDsymFiles() {
      return cpuSpecificArtifacts(ObjcProvider.DEBUG_SYMBOLS);
    }

    ImmutableMap<String, Artifact> cpuSpecificDsymPlists() {
      return cpuSpecificArtifacts(ObjcProvider.DEBUG_SYMBOLS_PLIST);
    }

    ImmutableMap<String, Artifact> cpuSpecificLinkmapFiles() {
      return cpuSpecificArtifacts(ObjcProvider.LINKMAP_FILE);
    }

    ImmutableMap<String, Artifact> cpuSpecificArtifacts(ObjcProvider.Key<Artifact> key) {
      ImmutableMap.Builder<String, Artifact> results = ImmutableMap.builder();
      if (ruleContext.attributes().has("binary", BuildType.LABEL)) {
        for (TransitiveInfoCollection prerequisite
            : ruleContext.getPrerequisites("binary", Mode.DONT_CHECK)) {
          ObjcProvider prerequisiteProvider =  prerequisite.getProvider(ObjcProvider.class);
          if (prerequisiteProvider != null) {
            Artifact sourceArtifact = Iterables.getOnlyElement(prerequisiteProvider.get(key), null);
            if (sourceArtifact != null) {
              String cpu =
                  prerequisite.getConfiguration().getFragment(AppleConfiguration.class).getIosCpu();
              results.put(cpu, sourceArtifact);
            }
          }
        }
      }
      return results.build();
    }
  }

  /**
   * Transition that results in one configured target per architecture set in {@code
   * --ios_multi_cpus}.
   */
  protected static class SplitArchTransition implements SplitTransition<BuildOptions> {

    @Override
    public final List<BuildOptions> split(BuildOptions buildOptions) {
      List<String> iosMultiCpus = buildOptions.get(AppleCommandLineOptions.class).iosMultiCpus;
      if (iosMultiCpus.isEmpty()) {
        return defaultOptions(buildOptions);
      }

      ImmutableList.Builder<BuildOptions> splitBuildOptions = ImmutableList.builder();
      for (String iosCpu : iosMultiCpus) {
        BuildOptions splitOptions = buildOptions.clone();
        setArchitectureOptions(splitOptions, iosCpu);
        setAdditionalOptions(splitOptions, buildOptions);
        splitOptions.get(AppleCommandLineOptions.class).configurationDistinguisher =
            getConfigurationDistinguisher();
        splitBuildOptions.add(splitOptions);
      }
      return splitBuildOptions.build();
    }

    /**
     * Returns the default options to use if no split architectures are specified.
     *
     * @param originalOptions original options before this transition
     */
    protected ImmutableList<BuildOptions> defaultOptions(BuildOptions originalOptions) {
      return ImmutableList.of();
    }

    /**
     * Sets or overwrites flags on the given split options.
     *
     * <p>Invoked once for each configuration produced by this transition.
     *
     * @param splitOptions options to use after this transition
     * @param originalOptions original options before this transition
     */
    protected void setAdditionalOptions(BuildOptions splitOptions, BuildOptions originalOptions) {}

    private void setArchitectureOptions(BuildOptions splitOptions, String iosCpu) {
      splitOptions.get(AppleCommandLineOptions.class).applePlatformType = PlatformType.IOS;
      splitOptions.get(AppleCommandLineOptions.class).appleSplitCpu = iosCpu;
      splitOptions.get(AppleCommandLineOptions.class).iosCpu = iosCpu;
      if (splitOptions.get(ObjcCommandLineOptions.class).enableCcDeps) {
        // Only set the (CC-compilation) CPU for dependencies if explicitly required by the user.
        // This helps users of the iOS rules who do not depend on CC rules as these CPU values
        // require additional flags to work (e.g. a custom crosstool) which now only need to be set
        // if this feature is explicitly requested.
        splitOptions.get(BuildConfiguration.Options.class).cpu = "ios_" + iosCpu;
      }
    }

    @Override
    public boolean defaultsToSelf() {
      return true;
    }

    /** Returns the configuration distinguisher for this transition instance. */
    protected ConfigurationDistinguisher getConfigurationDistinguisher() {
      return ConfigurationDistinguisher.IOS_APPLICATION;
    }
  }
}
