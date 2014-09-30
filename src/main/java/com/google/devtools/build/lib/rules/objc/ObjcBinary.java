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

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCASSETS_DIR;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.objc.ObjcActionsBuilder.ExtraActoolArgs;
import com.google.devtools.build.lib.rules.objc.ObjcActionsBuilder.ExtraLinkArgs;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.actions.SpawnAction;
import com.google.devtools.build.xcode.common.Platform;
import com.google.devtools.build.xcode.util.Interspersing;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.DependencyControl;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.TargetControl;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.XcodeprojBuildSetting;

import java.util.Map;

/**
 * Implementation for the "objc_binary" rule.
 */
public class ObjcBinary implements RuleConfiguredTargetFactory {
  @VisibleForTesting
  static final String REQUIRES_AT_LEAST_ONE_LIBRARY_OR_SOURCE_FILE = "At least one library "
      + "dependency or source file is required.";

  @VisibleForTesting
  static final String DEVICE_NO_PROVISIONING_PROFILE =
      "Provisioning profile must be set for device build";

  @VisibleForTesting
  static final String PROVISIONING_PROFILE_BUNDLE_FILE = "embedded.mobileprovision";

  @VisibleForTesting
  static final String SIMULATOR_PROVISIONING_PROFILE_ERROR =
      "must not specify provisioning profile for simulator build";

  @VisibleForTesting
  static final String NO_ASSET_CATALOG_ERROR_FORMAT =
      "a value was specified (%s), but this app does not have any asset catalogs";

  @VisibleForTesting
  static final String NO_INFOPLIST_ERROR = "An infoplist must be specified either in the "
      + "'infoplist' attribute or via the 'options' attribute, but none was found";

  private static Iterable<DependencyControl> targetDependenciesTransitive(
      Iterable<XcodeProvider> providers) {
    ImmutableSet.Builder<DependencyControl> result = new ImmutableSet.Builder<>();
    for (XcodeProvider provider : providers) {
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

  static void checkAttributes(RuleContext ruleContext, ObjcCommon common, Bundling bundling) {
    if (bundling.getInfoplistMerging().getInputPlists().isEmpty()) {
      ruleContext.ruleError(NO_INFOPLIST_ERROR);
    }

    common.reportErrors();
    if (!bundling.getLinkedBinary().isPresent()) {
      ruleContext.ruleError(REQUIRES_AT_LEAST_ONE_LIBRARY_OR_SOURCE_FILE);
    }

    // No asset catalogs. That means you cannot specify app_icon or
    // launch_image attributes, since they must not exist. However, we don't
    // run actool in this case, which means it does not do validity checks,
    // and we MUST raise our own error somehow...
    if (common.getObjcProvider().get(XCASSETS_DIR).isEmpty()) {
      for (String appIcon : ObjcBinaryRule.appIcon(ruleContext).asSet()) {
        ruleContext.attributeError("app_icon",
            String.format(NO_ASSET_CATALOG_ERROR_FORMAT, appIcon));
      }
      for (String launchImage : ObjcBinaryRule.launchImage(ruleContext).asSet()) {
        ruleContext.attributeError("launch_image",
            String.format(NO_ASSET_CATALOG_ERROR_FORMAT, launchImage));
      }
    }
  }

  static XcodeProvider xcodeProvider(RuleContext ruleContext, ObjcCommon common,
      InfoplistMerging infoplistMerging, OptionsProvider optionsProvider) {
    ImmutableList.Builder<XcodeprojBuildSetting> buildSettings = new ImmutableList.Builder<>();
    for (String appIcon : ObjcBinaryRule.appIcon(ruleContext).asSet()) {
      buildSettings.add(XcodeprojBuildSetting.newBuilder()
          .setName("ASSETCATALOG_COMPILER_APPICON_NAME")
          .setValue(appIcon)
          .build());
    }
    for (String launchImage : ObjcBinaryRule.launchImage(ruleContext).asSet()) {
      buildSettings.add(XcodeprojBuildSetting.newBuilder()
          .setName("ASSETCATALOG_COMPILER_LAUNCHIMAGE_NAME")
          .setValue(launchImage)
          .build());
    }

    return common.xcodeProvider(
        infoplistMerging.getPlistWithEverything(),
        targetDependenciesTransitive(ObjcRuleClasses.deps(ruleContext, XcodeProvider.class)),
        buildSettings.build(),
        optionsProvider.getCopts());
  }

  private static Optional<Artifact> provisioningProfile(RuleContext context) {
    return Optional.fromNullable(
        context.getPrerequisiteArtifact(ObjcBinaryRule.PROVISIONING_PROFILE_ATTR, Mode.TARGET));
  }

  static ImmutableList<BundleableFile> extraBundleFiles(RuleContext context) {
    ImmutableList.Builder<BundleableFile> files = new ImmutableList.Builder<>();

    ObjcConfiguration objcConfiguration = ObjcActionsBuilder.objcConfiguration(context);
    if (objcConfiguration.getPlatform() == Platform.DEVICE) {
      files.add(new BundleableFile(
          provisioningProfile(context).get(), PROVISIONING_PROFILE_BUNDLE_FILE));
    }

    return files.build();
  }

  static void registerActions(RuleContext ruleContext, ObjcCommon common,
      XcodeProvider xcodeProvider, ExtraLinkArgs extraLinkArgs,
      OptionsProvider optionsProvider, Bundling bundling) {
    ObjcConfiguration objcConfiguration = ObjcActionsBuilder.objcConfiguration(ruleContext);
    ObjcProvider objcProvider = common.getObjcProvider();

    ExtraActoolArgs extraActoolArgs = new ExtraActoolArgs(
        Iterables.concat(
            Interspersing.beforeEach(
                "--app-icon", ObjcBinaryRule.appIcon(ruleContext).asSet()),
            Interspersing.beforeEach(
                "--launch-image", ObjcBinaryRule.launchImage(ruleContext).asSet())));

    ObjcBundleLibrary.registerActions(ruleContext, bundling, common, xcodeProvider, optionsProvider,
        extraLinkArgs, extraActoolArgs);

    Artifact ipaOutput = ruleContext.getImplicitOutputArtifact(ObjcBinaryRule.IPA);

    Optional<Artifact> entitlements = Optional.fromNullable(
        ruleContext.getPrerequisiteArtifact("entitlements", Mode.TARGET));

    Artifact ipaUnsigned;

    if (objcConfiguration.getPlatform() == Platform.SIMULATOR) {
      ipaUnsigned = ipaOutput;

      if (ruleContext.attributes().isAttributeValueExplicitlySpecified(
          ObjcBinaryRule.EXPLICIT_PROVISIONING_PROFILE_ATTR)) {
        ruleContext.attributeError(ObjcBinaryRule.EXPLICIT_PROVISIONING_PROFILE_ATTR,
            SIMULATOR_PROVISIONING_PROFILE_ERROR);
      }
    } else if (!provisioningProfile(ruleContext).isPresent()) {
      throw new IllegalStateException(DEVICE_NO_PROVISIONING_PROFILE);
    } else {
      if (!entitlements.isPresent()) {
        entitlements = Optional.of(ruleContext.getRelatedArtifact(
            ruleContext.getUniqueDirectory("entitlements"), ".entitlements"));

        // See http://goo.gl/EkhXOb
        // An Application Identifier is constructed as: TeamID.BundleID
        // TeamID is extracted from the provisioning profile.
        // BundleID consists of a reverse-DNS string to identify the app, where the last component
        // is the application name, and is specified as an attribute.

        ruleContext.getAnalysisEnvironment().registerAction(new SpawnAction.Builder(ruleContext)
            .setMnemonic("Extract entitlements")
            .setExecutable(new PathFragment("/bin/bash"))
            .addArgument("-c")
            .addArgument("set -e && "
                + "PLIST=$(" + extractPlistCommand(provisioningProfile(ruleContext).get()) + ") && "

                // We think PlistBuddy uses PRead internally to seek through the file. Or possibly
                // mmaps the file. Or something similar.
                //
                // Pipe FDs do not support PRead or mmap, though.
                //
                // <<< however does something magical like write to a temporary file or something
                // like that internally, which means that this Just Works.
                + "PREFIX=$(/usr/libexec/PlistBuddy -c 'Print ApplicationIdentifierPrefix:0' "
                + "/dev/stdin <<< \"${PLIST}\") && "

                + "/usr/libexec/PlistBuddy -x -c 'Print Entitlements' /dev/stdin <<< \"${PLIST}\""
                // TODO(bazel-team): Do this substitution for all entitlements files, not just the
                // default.
                + "| sed -e \"s#${PREFIX}\\.\\*#${PREFIX}."
                + ShellUtils.shellEscape(ruleContext.attributes().get("bundle_id", Type.STRING))
                + "#g\" > " + entitlements.get().getExecPathString())
            .addInput(provisioningProfile(ruleContext).get())
            .addOutput(entitlements.get())
            .setExecutionInfo(ImmutableMap.of(ExecutionRequirements.REQUIRES_DARWIN, ""))
            .build());
      }
      ipaUnsigned = ObjcRuleClasses.artifactByAppendingToRootRelativePath(
          ruleContext, ipaOutput.getExecPath(), ".unsigned");

      // TODO(bazel-team): Support variable substitution
      ruleContext.getAnalysisEnvironment().registerAction(new SpawnAction.Builder(ruleContext)
          .setMnemonic("Sign app bundle")
          .setExecutable(new PathFragment("/bin/bash"))
          .addArgument("-c")
          // TODO(bazel-team): Support --resource-rules for resources
          .addArgument("set -e && "
              + "t=$(mktemp -d -t signing_intermediate) && "
              + "unzip -qq " + ipaUnsigned.getExecPathString() + " -d ${t} && "
              + codesignCommand(
                  provisioningProfile(ruleContext).get(),
                  entitlements.get(),
                  String.format("${t}/Payload/%s.app", ruleContext.getLabel().getName())) + " && "
              // Using jar not zip because it allows us to specify -C without actually changing
              // directory
              // TODO(bazel-team): Junk timestamps
              + "jar -cMf '" + ipaOutput.getExecPathString() + "' -C ${t} .")
          .addInput(ipaUnsigned)
          .addInput(provisioningProfile(ruleContext).get())
          .addInput(entitlements.get())
          .addOutput(ipaOutput)
          .setExecutionInfo(ImmutableMap.of(ExecutionRequirements.REQUIRES_DARWIN, ""))
          .build());
    }

    Artifact bundleMergeControlArtifact =
        ObjcRuleClasses.artifactByAppendingToBaseName(ruleContext, ".ipa-control");
    ruleContext.getAnalysisEnvironment().registerAction(
        new WriteMergeBundleControlFileAction(
            ruleContext.getActionOwner(), bundling, ipaUnsigned, bundleMergeControlArtifact,
            objcConfiguration, variableSubstitutionsInBundleMerge(ruleContext)));

    ruleContext.getAnalysisEnvironment().registerAction(new SpawnAction.Builder(ruleContext)
        .setMnemonic("Generate app bundle")
        .setExecutable(ruleContext.getExecutablePrerequisite("$bundlemerge", Mode.HOST))
        .addInputArgument(bundleMergeControlArtifact)
        .addTransitiveInputs(bundling.getBundleContentArtifacts())
        .addOutput(ipaUnsigned)
        .build());
  }

  private static String codesignCommand(
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

  private static String extractPlistCommand(Artifact provisioningProfile) {
    return "security cms -D -i " + ShellUtils.shellEscape(provisioningProfile.getExecPathString());
  }

  private static Map<String, String> variableSubstitutionsInBundleMerge(RuleContext ruleContext) {
    return ImmutableMap.of(
        "EXECUTABLE_NAME", ruleContext.getLabel().getName(),
        "BUNDLE_NAME", ruleContext.getLabel().getName() + ".app");
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    IntermediateArtifacts intermediateArtifacts = new IntermediateArtifacts(
        ruleContext.getAnalysisEnvironment(), ruleContext.getBinOrGenfilesDirectory(),
        ruleContext.getLabel());

    CompilationArtifacts compilationArtifacts = new CompilationArtifacts.Builder()
        .addSrcs(ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET))
        .addNonArcSrcs(ruleContext.getPrerequisiteArtifacts("non_arc_srcs", Mode.TARGET))
        .setIntermediateArtifacts(intermediateArtifacts)
        .setPchFile(Optional.fromNullable(ruleContext.getPrerequisiteArtifact("pch", Mode.TARGET)))
        .build();

    ObjcCommon common = new ObjcCommon.Builder(ruleContext)
        .addAssetCatalogs(ruleContext.getPrerequisiteArtifacts("asset_catalogs", Mode.TARGET))
        .addSdkDylibs(ruleContext.attributes().get("sdk_dylibs", Type.STRING_LIST))
        .setCompilationArtifacts(compilationArtifacts)
        .addHdrs(ruleContext.getPrerequisiteArtifacts("hdrs", Mode.TARGET))
        .build();

    OptionsProvider optionsProvider = new OptionsProvider.Builder()
        .addCopts(ruleContext.getTokenizedStringListAttr("copts"))
        .addInfoplists(ruleContext.getPrerequisiteArtifacts("infoplist", Mode.TARGET))
        .addTransitive(Optional.fromNullable(
            ruleContext.getPrerequisite("options", Mode.TARGET, OptionsProvider.class)))
        .build();

    InfoplistMerging infoplistMerging = new InfoplistMerging.Builder(ruleContext)
        .setIntermediateArtifacts(intermediateArtifacts)
        .setInputPlists(optionsProvider.getInfoplists())
        .setPlmerge(ruleContext.getExecutablePrerequisite("$plmerge", Mode.HOST))
        .build();

    Bundling bundling = new Bundling.Builder()
        .setName(ruleContext.getLabel().getName())
        .setBundleDirSuffix(".app")
        .setExtraBundleFiles(extraBundleFiles(ruleContext))
        .setObjcProvider(common.getObjcProvider())
        .setInfoplistMerging(infoplistMerging)
        .setIntermediateArtifacts(intermediateArtifacts)
        .build();

    checkAttributes(ruleContext, common, bundling);
    XcodeProvider xcodeProvider =
        xcodeProvider(ruleContext, common, infoplistMerging, optionsProvider);

    registerActions(
        ruleContext, common, xcodeProvider, new ExtraLinkArgs(), optionsProvider, bundling);

    return common.configuredTarget(
        NestedSetBuilder.<Artifact>stableOrder()
            .add(ruleContext.getImplicitOutputArtifact(ObjcBinaryRule.IPA))
            .add(ruleContext.getImplicitOutputArtifact(ObjcRuleClasses.PBXPROJ))
            .build(),
        Optional.of(xcodeProvider),
        Optional.<ObjcProvider>absent());
  }
}
