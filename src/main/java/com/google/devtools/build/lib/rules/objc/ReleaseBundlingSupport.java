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
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.Attribute.SplitTransition;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.rules.apple.Platform;
import com.google.devtools.build.lib.rules.objc.BundleSupport.ExtraActoolArgs;
import com.google.devtools.build.lib.rules.objc.Bundling.Builder;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.syntax.Type;
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
   */
  ReleaseBundlingSupport(
      RuleContext ruleContext,
      ObjcProvider objcProvider,
      LinkedBinary linkedBinary,
      String bundleDirFormat,
      String bundleName,
      DottedVersion bundleMinimumOsVersion) {
    this.linkedBinary = linkedBinary;
    this.attributes = new Attributes(ruleContext);
    this.ruleContext = ruleContext;
    this.objcProvider = objcProvider;
    this.intermediateArtifacts = ObjcRuleClasses.intermediateArtifacts(ruleContext);
    bundling = bundling(ruleContext, objcProvider, bundleDirFormat, bundleName,
        bundleMinimumOsVersion);
    bundleSupport = new BundleSupport(ruleContext, bundling, extraActoolArgs());
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
   */
  ReleaseBundlingSupport(
      RuleContext ruleContext,
      ObjcProvider objcProvider,
      LinkedBinary linkedBinary,
      String bundleDirFormat,
      DottedVersion bundleMinimumOsVersion) {
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
      if (attributes.appIcon() != null) {
        ruleContext.attributeError("app_icon",
            String.format(NO_ASSET_CATALOG_ERROR_FORMAT, attributes.appIcon()));
      }
      if (attributes.launchImage() != null) {
        ruleContext.attributeError("launch_image",
            String.format(NO_ASSET_CATALOG_ERROR_FORMAT, attributes.launchImage()));
      }
    }

    if (bundleSupport.targetDeviceFamilies().isEmpty()) {
      ruleContext.attributeError("families", INVALID_FAMILIES_ERROR);
    }

    validateLaunchScreen();

    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);
    if (attributes.provisioningProfile() == null
        && appleConfiguration.getBundlingPlatform() != Platform.IOS_SIMULATOR) {
      ruleContext.attributeError("provisioning_profile", DEVICE_NO_PROVISIONING_PROFILE);
    }

    return this;
  }

  private void validateLaunchScreen() {
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("launch_storyboard")) {
      DottedVersion minimumOs = bundling.getMinimumOsVersion();
      if (ObjcRuleClasses.useLaunchStoryboard(ruleContext, bundling.getMinimumOsVersion())) {
        if (ruleContext.attributes().isAttributeValueExplicitlySpecified("launch_image")) {
          ruleContext.attributeWarning(
              "launch_image",
              String.format(
                  "launch_image was specified but since --ios_minimum_os=%s (>=8.0), the also "
                      + "specified launch_storyboard will be used instead",
                  minimumOs));
        }
      } else {
        if (!ruleContext.attributes().isAttributeValueExplicitlySpecified("launch_image")) {
          ruleContext.attributeWarning(
              "launch_storyboard",
              String.format(
                  "launch_storyboard was specified but since --ios_minimum_os=%s (<8.0) and no "
                      + "launch_image was specified instead it will be ignored",
                  minimumOs));
        }
      }
    }
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
   * @return this application support
   * @throws InterruptedException
   */
  ReleaseBundlingSupport registerActions() throws InterruptedException {
    bundleSupport.registerActions(objcProvider);

    registerCombineArchitecturesAction();
    registerTransformAndCopyBreakpadFilesAction();
    registerSwiftStdlibActionsIfNecessary();

    registerEmbedLabelPlistAction();
    registerEnvironmentPlistAction();
    registerAutomaticPlistAction();

    if (ObjcRuleClasses.useLaunchStoryboard(ruleContext, bundling.getMinimumOsVersion())) {
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
    String launchStoryboard = attributes.launchStoryboard().getFilename();
    String launchStoryboardName = launchStoryboard.substring(0, launchStoryboard.lastIndexOf('.'));
    String contents =
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            + "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" "
            + "\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
            + "<plist version=\"1.0\">\n"
            + "<dict>\n"
            + "  <key>UILaunchStoryboardName</key>\n"
            + "  <string>"
            + launchStoryboardName
            + "</string>\n"
            + "</dict>\n"
            + "</plist>\n";

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
        ObjcRuleClasses.spawnXcrunActionBuilder(ruleContext)
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
      result.put("UIDeviceFamily", NSObject.wrap(uiDeviceFamily.toArray()));
    }
    result.put("DTPlatformName", NSObject.wrap(platform.getLowerCaseNameInPlist()));
    result.put(
        "DTSDKName",
        NSObject.wrap(platform.getLowerCaseNameInPlist() + appleConfiguration.getIosSdkVersion()));
    result.put("CFBundleSupportedPlatforms", new NSArray(NSObject.wrap(platform.getNameInPlist())));
    result.put("MinimumOSVersion", NSObject.wrap(bundling.getMinimumOsVersion().toString()));

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
    Artifact processedIpa = ruleContext.getImplicitOutputArtifact(IPA);
    Artifact unprocessedIpa = intermediateArtifacts.unprocessedIpa();

    boolean processingNeeded = false;
    NestedSetBuilder<Artifact> inputs =
        NestedSetBuilder.<Artifact>stableOrder().add(unprocessedIpa);

    String actionCommandLine =
        "set -e && "
            + "t=$(mktemp -d -t signing_intermediate) && "
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
      inputs.add(attributes.provisioningProfile()).add(intermediateArtifacts.entitlements());
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
    ImmutableList.Builder<String> dirsToSign = new ImmutableList.Builder<>();

    // Explicitly sign Swift dylibs. Unfortunately --deep option on codesign doesn't do this
    // automatically.
    // The order here is important. The innermost code must singed first.
    String bundleDir = ShellUtils.shellEscape(bundling.getBundleDir());
    if (objcProvider.is(USES_SWIFT)) {
      dirsToSign.add(bundleDir + "/Frameworks/*");
    }
    dirsToSign.add(bundleDir);

    StringBuilder codesignCommandLineBuilder = new StringBuilder();
    for (String dir : dirsToSign.build()) {
      codesignCommandLineBuilder.append(codesignCommand("${t}/" + dir)).append(" && ");
    }
    return codesignCommandLineBuilder.toString();
  }

  private void registerEntitlementsActions() throws InterruptedException {
    Artifact teamPrefixFile =
        intermediateArtifacts.appendExtensionForEntitlementArtifact(".team_prefix_file");
    registerExtractTeamPrefixAction(teamPrefixFile);

    Artifact entitlementsNeedingSubstitution = attributes.entitlements();
    if (entitlementsNeedingSubstitution == null) {
      entitlementsNeedingSubstitution =
          intermediateArtifacts.appendExtensionForEntitlementArtifact(
              ".entitlements_with_variables");
      registerExtractEntitlementsAction(entitlementsNeedingSubstitution);
    }
    registerEntitlementsVariableSubstitutionAction(entitlementsNeedingSubstitution, teamPrefixFile);
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
   * @return this application support
   * @throws InterruptedException 
   */
  ReleaseBundlingSupport addFilesToBuild(NestedSetBuilder<Artifact> filesToBuild)
      throws InterruptedException {
    NestedSetBuilder<Artifact> debugSymbolBuilder =
        NestedSetBuilder.<Artifact>stableOrder().addTransitive(
            objcProvider.get(ObjcProvider.DEBUG_SYMBOLS));

    for (Artifact breakpadFile : getBreakpadFiles().values()) {
      filesToBuild.add(breakpadFile);
    }

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
   * @throws InterruptedException 
   */
  XcTestAppProvider xcTestAppProvider() throws InterruptedException {
    // We want access to #import-able things from our test rig's dependency graph, but we don't
    // want to link anything since that stuff is shared automatically by way of the
    // -bundle_loader linker flag.
    ObjcProvider partialObjcProvider = new ObjcProvider.Builder()
        .addTransitiveAndPropagate(ObjcProvider.HEADER, objcProvider)
        .addTransitiveAndPropagate(ObjcProvider.INCLUDE, objcProvider)
        .addTransitiveAndPropagate(ObjcProvider.DEFINE, objcProvider)
        .addTransitiveAndPropagate(ObjcProvider.SDK_DYLIB, objcProvider)
        .addTransitiveAndPropagate(ObjcProvider.SDK_FRAMEWORK, objcProvider)
        .addTransitiveAndPropagate(ObjcProvider.SOURCE, objcProvider)
        .addTransitiveAndPropagate(ObjcProvider.WEAK_SDK_FRAMEWORK, objcProvider)
        .addTransitiveAndPropagate(ObjcProvider.FRAMEWORK_DIR, objcProvider)
        .addTransitiveAndPropagate(ObjcProvider.FRAMEWORK_FILE, objcProvider)
        .build();
    // TODO(bazel-team): Handle the FRAMEWORK_DIR key properly. We probably want to add it to
    // framework search paths, but not actually link it with the -framework flag.
    return new XcTestAppProvider(intermediateArtifacts.combinedArchitectureBinary(),
        ruleContext.getImplicitOutputArtifact(IPA), partialObjcProvider);
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
        Substitution.of("%ipa_file%", ipaInput.getRootRelativePath().getPathString()),
        Substitution.of("%sim_device%", escapedSimDevice),
        Substitution.of("%sdk_version%", escapedSdkVersion),
        Substitution.of("%iossim%", attributes.iossim().getRootRelativePath().getPathString()),
        Substitution.of("%std_redirect_dylib_path%",
            attributes.stdRedirectDylib().getRootRelativePath().getPathString()));

    ruleContext.registerAction(
        new TemplateExpansionAction(ruleContext.getActionOwner(), attributes.runnerScriptTemplate(),
            runnerScript, substitutions, true));
    return this;
  }

  /**
   * Returns a {@link RunfilesSupport} that uses the provided runner script as the executable.
   * @throws InterruptedException 
   */
  RunfilesSupport runfilesSupport(Artifact runnerScript) throws InterruptedException {
    Artifact ipaFile = ruleContext.getImplicitOutputArtifact(ReleaseBundlingSupport.IPA);
    Runfiles runfiles = new Runfiles.Builder(ruleContext.getWorkspaceName())
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
    if (attributes.launchImage() != null
        && !ObjcRuleClasses.useLaunchStoryboard(ruleContext, bundling.getMinimumOsVersion())) {
      extraArgs.add("--launch-image", attributes.launchImage());
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

    Bundling.Builder bundling =
        new Builder()
            .setName(bundleName)
            // Architecture that determines which nested bundles are kept.
            .setArchitecture(appleConfiguration.getDependencySingleArchitecture())
            .setBundleDirFormat(bundleDirFormat)
            .addExtraBundleFiles(extraBundleFiles)
            .setObjcProvider(objcProvider)
            .addInfoplistInputFromRule(ruleContext)
            .addInfoplistInput(getGeneratedVersionPlist())
            .addInfoplistInput(getGeneratedEnvironmentPlist())
            .setAutomaticEntriesInfoplistInput(getGeneratedAutomaticPlist())
            .setIntermediateArtifacts(ObjcRuleClasses.intermediateArtifacts(ruleContext))
            .setPrimaryBundleId(primaryBundleId)
            .setFallbackBundleId(fallbackBundleId)
            .setMinimumOsVersion(minimumOsVersion);

    if (ObjcRuleClasses.useLaunchStoryboard(ruleContext, minimumOsVersion)) {
      bundling.addInfoplistInput(getLaunchStoryboardPlist());
    }

    return bundling.build();
  }

  private void registerCombineArchitecturesAction() {
    Artifact resultingLinkedBinary = intermediateArtifacts.combinedArchitectureBinary();
    NestedSet<Artifact> linkedBinaries = linkedBinaries();

    ruleContext.registerAction(ObjcRuleClasses.spawnXcrunActionBuilder(ruleContext)
        .setMnemonic("ObjcCombiningArchitectures")
        .addTransitiveInputs(linkedBinaries)
        .addOutput(resultingLinkedBinary)
        .setExecutable(CompilationSupport.xcrunwrapper(ruleContext))
        .setCommandLine(CustomCommandLine.builder()
            .add(ObjcRuleClasses.LIPO)
            .addExecPaths("-create", linkedBinaries)
            .addExecPath("-o", resultingLinkedBinary)
            .build())
        .build(ruleContext));
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
    if (attributes.appIcon() != null) {
      buildSettings.add(XcodeprojBuildSetting.newBuilder()
          .setName("ASSETCATALOG_COMPILER_APPICON_NAME")
          .setValue(attributes.appIcon())
          .build());
    }
    if (attributes.launchImage() != null
        && !ObjcRuleClasses.useLaunchStoryboard(ruleContext, bundling.getMinimumOsVersion())) {
      buildSettings.add(XcodeprojBuildSetting.newBuilder()
          .setName("ASSETCATALOG_COMPILER_LAUNCHIMAGE_NAME")
          .setValue(attributes.launchImage())
          .build());
    }

    // Convert names to a sequence containing "1" and/or "2" for iPhone and iPad, respectively.
    ImmutableSet<TargetDeviceFamily> families = bundleSupport.targetDeviceFamilies();
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

  private void registerBundleMergeActions() {
    Artifact bundleMergeControlArtifact =
        ObjcRuleClasses.artifactByAppendingToBaseName(ruleContext, ".ipa-control");

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

  private void registerExtractTeamPrefixAction(Artifact teamPrefixFile) {
    String shellCommand = "set -e && "
        + "PLIST=$(mktemp -t teamprefix.plist) && trap \"rm ${PLIST}\" EXIT && "
        + extractPlistCommand(attributes.provisioningProfile()) + " > ${PLIST} && "
        + "/usr/libexec/PlistBuddy -c 'Print ApplicationIdentifierPrefix:0' ${PLIST} > "
        + teamPrefixFile.getShellEscapedExecPathString();
    ruleContext.registerAction(
        ObjcRuleClasses.spawnBashOnDarwinActionBuilder(shellCommand)
            .setMnemonic("ExtractIosTeamPrefix")
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
    String shellCommand = "set -e && "
        + "PLIST=$(mktemp -t entitlements.plist) && trap \"rm ${PLIST}\" EXIT && "
        + extractPlistCommand(attributes.provisioningProfile()) + " > ${PLIST} && "
        + "/usr/libexec/PlistBuddy -x -c 'Print Entitlements' ${PLIST} > "
        + entitlements.getShellEscapedExecPathString();
    ruleContext.registerAction(
        ObjcRuleClasses.spawnBashOnDarwinActionBuilder(shellCommand)
            .setMnemonic("ExtractIosEntitlements")
            .setProgressMessage("Extracting entitlements: " + ruleContext.getLabel())
            .addInput(attributes.provisioningProfile())
            .addOutput(entitlements)
            .build(ruleContext));

    return this;
  }

  private void registerEntitlementsVariableSubstitutionAction(
      Artifact inputEntitlements, Artifact prefix) {
    Artifact substitutedEntitlements = intermediateArtifacts.entitlements();
    String escapedBundleId = ShellUtils.shellEscape(attributes.bundleId());
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
        ObjcRuleClasses.spawnXcrunActionBuilder(ruleContext)
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
              + extractPlistCommand(attributes.provisioningProfile())
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
        ruleContext.getUniqueDirectory("plists"), "-version.plist");
  }

  private Artifact getGeneratedEnvironmentPlist() {
    return ruleContext.getRelatedArtifact(
        ruleContext.getUniqueDirectory("plists"), "-environment.plist");
  }
  
  private Artifact getGeneratedAutomaticPlist() {
    return ruleContext.getRelatedArtifact(
        ruleContext.getUniqueDirectory("plists"), "-automatic.plist");
  }

  private Artifact getLaunchStoryboardPlist() {
    return ruleContext.getRelatedArtifact(
        ruleContext.getUniqueDirectory("plists"), "-launchstoryboard.plist");
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
    Artifact launchStoryboard() {
      return ruleContext.getPrerequisiteArtifact("launch_storyboard", Mode.TARGET);
    }

    @Nullable
    Artifact provisioningProfile() {
      Artifact explicitProvisioningProfile =
          ruleContext.getPrerequisiteArtifact("provisioning_profile", Mode.TARGET);
      if (explicitProvisioningProfile != null) {
        return explicitProvisioningProfile;
      }
      return ruleContext.getPrerequisiteArtifact(":default_provisioning_profile", Mode.TARGET);
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

    @Nullable
    Artifact entitlements() {
      return ruleContext.getPrerequisiteArtifact("entitlements", Mode.TARGET);
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
    public FilesToRunProvider environmentPlist() {
      return ruleContext.getExecutablePrerequisite("$environment_plist", Mode.HOST);
    }

    String bundleId() {
      return checkNotNull(stringAttribute("bundle_id"));
    }

    ImmutableMap<String, Artifact> cpuSpecificBreakpadFiles() {
      ImmutableMap.Builder<String, Artifact> results = ImmutableMap.builder();
      if (ruleContext.attributes().has("binary", BuildType.LABEL)) {
        for (TransitiveInfoCollection prerequisite
            : ruleContext.getPrerequisites("binary", Mode.DONT_CHECK)) {
          ObjcProvider prerequisiteProvider =  prerequisite.getProvider(ObjcProvider.class);
          if (prerequisiteProvider != null) {
            Artifact sourceBreakpad = Iterables.getOnlyElement(
                prerequisiteProvider.get(ObjcProvider.BREAKPAD_FILE), null);
            if (sourceBreakpad != null) {
              String cpu =
                  prerequisite.getConfiguration().getFragment(AppleConfiguration.class).getIosCpu();
              results.put(cpu, sourceBreakpad);
            }
          }
        }
      }
      return results.build();
    }

    @Nullable
    private String stringAttribute(String attribute) {
      String value = ruleContext.attributes().get(attribute, Type.STRING);
      return value.isEmpty() ? null : value;
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
        splitOptions.get(ObjcCommandLineOptions.class).configurationDistinguisher =
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
      splitOptions.get(ObjcCommandLineOptions.class).iosSplitCpu = iosCpu;
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
      return ConfigurationDistinguisher.APPLICATION;
    }

    /**
     * Value used to avoid multiple configurations from conflicting. No two instances of this
     * transition may exist with the same value in a single Bazel invocation.
     */
    enum ConfigurationDistinguisher {
      EXTENSION, APPLICATION, FRAMEWORK, UNKNOWN
    }
  }
}
