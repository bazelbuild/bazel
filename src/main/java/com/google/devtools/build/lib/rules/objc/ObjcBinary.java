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

import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.rules.objc.IosSdkCommands.MINIMUM_OS_VERSION;
import static com.google.devtools.build.lib.rules.objc.IosSdkCommands.TARGET_DEVICE_FAMILIES;
import static com.google.devtools.build.lib.rules.objc.ObjcActionsBuilder.CLANG;
import static com.google.devtools.build.lib.rules.objc.ObjcActionsBuilder.CLANG_PLUSPLUS;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.BUNDLE_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.Flag.USES_CPP;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCASSETS_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCDATAMODEL;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.actions.SpawnAction;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.MergeZip;
import com.google.devtools.build.xcode.common.Platform;
import com.google.devtools.build.xcode.common.TargetDeviceFamily;
import com.google.devtools.build.xcode.util.Interspersing;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.DependencyControl;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.TargetControl;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.XcodeprojBuildSetting;

import java.util.List;

/**
 * Implementation for the "objc_binary" rule.
 */
public class ObjcBinary implements RuleConfiguredTargetFactory {

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

  static void checkAttributes(RuleContext ruleContext, ObjcCommon info) {
    List<Artifact> infoplistFiles = ruleContext.getPrerequisiteArtifacts("infoplist", Mode.TARGET);

    if (infoplistFiles.size() != 1) {
      ruleContext.attributeError("infoplist", "expected exactly 1 infoplist, but found: "
          + infoplistFiles.size());
    }

    info.reportErrors();

    // No asset catalogs. That means you cannot specify app_icon or
    // launch_image attributes, since they must not exist. However, we don't
    // run actool in this case, which means it does not do validity checks,
    // and we MUST raise our own error somehow...
    if (info.getObjcProvider().get(XCASSETS_DIR).isEmpty()) {
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

  static XcodeProvider xcodeProvider(RuleContext ruleContext, ObjcCommon info) {
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

    return info.xcodeProvider(
        Optional.fromNullable(ruleContext.getPrerequisiteArtifact("infoplist", Mode.TARGET)),
        ObjcRuleClasses.pchFile(ruleContext),
        targetDependenciesTransitive(ObjcRuleClasses.deps(ruleContext, XcodeProvider.class)),
        buildSettings.build());
  }

  static void registerActions(RuleContext ruleContext, ObjcProvider objcProvider,
      XcodeProvider xcodeProvider, Iterable<String> extraLinkArgs) {
    ObjcConfiguration objcConfiguration = ObjcActionsBuilder.objcConfiguration(ruleContext);
    Artifact binaryOutput = ruleContext.getImplicitOutputArtifact(ObjcBinaryRule.BINARY);

    ruleContext.getAnalysisEnvironment().registerAction(new SpawnAction.Builder(ruleContext)
        .setMnemonic("Link")
        .setExecutable(objcProvider.is(USES_CPP) ? CLANG_PLUSPLUS : CLANG)
        .addArguments(objcProvider.is(USES_CPP)
            ? ImmutableList.of("-stdlib=libc++") : ImmutableList.<String>of())
        .addArguments(IosSdkCommands.commonLinkAndCompileArgsForClang(objcConfiguration))
        .addArgument("-Xlinker").addArgument("-objc_abi_version")
        .addArgument("-Xlinker").addArgument("2")
        .addArgument("-fobjc-link-runtime")
        .addArgument("-ObjC")
        .addArguments(Interspersing.beforeEach(
            "-framework", SdkFramework.names(objcProvider.get(SDK_FRAMEWORK))))
        .addArgument("-o").addOutputArgument(binaryOutput)
        .addInputArguments(objcProvider.get(LIBRARY))
        .addInputArguments(objcProvider.get(IMPORTED_LIBRARY))
        .addArguments(extraLinkArgs)
        .setExecutionInfo(ImmutableMap.of(ExecutionRequirements.REQUIRES_DARWIN, ""))
        .build());

    Iterable<Artifact> maybeActoolOutputZip = ImmutableList.of();
    if (!objcProvider.get(XCASSETS_DIR).isEmpty()) {
      Action actoolAction = ObjcActionsBuilder.actoolzipAction(ruleContext, objcProvider);
      ruleContext.getAnalysisEnvironment().registerAction(actoolAction);
      maybeActoolOutputZip = actoolAction.getOutputs();
    }

    Artifact ipaOutput = ruleContext.getImplicitOutputArtifact(ObjcBinaryRule.IPA);
    List<Artifact> infoplistFiles = ruleContext.getPrerequisiteArtifacts("infoplist", Mode.TARGET);
    Optional<Artifact> provisioningProfile = Optional.fromNullable(
        ruleContext.getPrerequisiteArtifact(ObjcBinaryRule.PROVISIONING_PROFILE_ATTR, Mode.TARGET));
    Optional<Artifact> entitlements = Optional.fromNullable(
        ruleContext.getPrerequisiteArtifact("entitlements", Mode.TARGET));

    BundleMergeProtos.Control.Builder mergeControl = BundleMergeProtos.Control.newBuilder()
        .addBundleFile(BundleMergeProtos.BundleFile.newBuilder()
            .setSourceFile(binaryOutput.getExecPathString())
            .setBundlePath(ruleContext.getTarget().getName())
            .build())
        .addAllBundleFile(BundleableFile.toBundleFiles(objcProvider.get(BUNDLE_FILE)))
        .addAllSourcePlistFile(Artifact.toExecPaths(infoplistFiles))
        // TODO(bazel-team): Add rule attributes for specifying targeted device family and minimum
        // OS version.
        .setMinimumOsVersion(MINIMUM_OS_VERSION)
        .setSdkVersion(objcConfiguration.getIosSdkVersion())
        .setPlatform(objcConfiguration.getPlatform().name())
        .setBundleRoot(ObjcBinaryRule.bundleRoot(ruleContext));

    Artifact ipaUnsigned;

    if (objcConfiguration.getPlatform() == Platform.SIMULATOR) {
      ipaUnsigned = ipaOutput;

      if (ruleContext.attributes().isAttributeValueExplicitlySpecified(
          ObjcBinaryRule.EXPLICIT_PROVISIONING_PROFILE_ATTR)) {
        ruleContext.attributeError(ObjcBinaryRule.EXPLICIT_PROVISIONING_PROFILE_ATTR,
            SIMULATOR_PROVISIONING_PROFILE_ERROR);
      }
    } else if (!provisioningProfile.isPresent()) {
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
                + "PLIST=$(" + extractPlistCommand(provisioningProfile.get()) + ") && "

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
                + ShellUtils.shellEscape(ruleContext.attributes().get("bundle_id", STRING))
                + "#g\" > " + entitlements.get().getExecPathString())
            .addInput(provisioningProfile.get())
            .addOutput(entitlements.get())
            .setExecutionInfo(ImmutableMap.of(ExecutionRequirements.REQUIRES_DARWIN, ""))
            .build());
      }
      ipaUnsigned = ObjcRuleClasses.artifactByAppendingToRootRelativePath(
          ruleContext, ipaOutput.getExecPath(), ".unsigned");
      mergeControl.addBundleFile(BundleMergeProtos.BundleFile.newBuilder()
          .setSourceFile(provisioningProfile.get().getExecPathString())
          .setBundlePath(PROVISIONING_PROFILE_BUNDLE_FILE)
          .build());

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
                  provisioningProfile.get(),
                  entitlements.get(),
                  String.format("${t}/Payload/%s.app", ruleContext.getLabel().getName())) + " && "
              // Using jar not zip because it allows us to specify -C without actually changing
              // directory
              // TODO(bazel-team): Junk timestamps
              + "jar -cMf '" + ipaOutput.getExecPathString() + "' -C ${t} .")
          .addInput(ipaUnsigned)
          .addInput(provisioningProfile.get())
          .addInput(entitlements.get())
          .addOutput(ipaOutput)
          .setExecutionInfo(ImmutableMap.of(ExecutionRequirements.REQUIRES_DARWIN, ""))
          .build());
    }

    for (Artifact actoolOutputZip : maybeActoolOutputZip) {
      mergeControl.addMergeZip(MergeZip.newBuilder()
          .setEntryNamePrefix(ObjcBinaryRule.bundleRoot(ruleContext) + "/")
          .setSourcePath(actoolOutputZip.getExecPathString())
          .build());
    }
    for (Xcdatamodel datamodel : objcProvider.get(XCDATAMODEL)) {
      mergeControl.addMergeZip(MergeZip.newBuilder()
          .setEntryNamePrefix(ObjcBinaryRule.bundleRoot(ruleContext) + "/")
          .setSourcePath(datamodel.getOutputZip().getExecPathString())
          .build());
    }
    for (TargetDeviceFamily targetDeviceFamily : TARGET_DEVICE_FAMILIES) {
      mergeControl.addTargetDeviceFamily(targetDeviceFamily.name());
    }

    mergeControl.setOutFile(ipaUnsigned.getExecPathString());

    Artifact bundleMergeControlArtifact = ObjcRuleClasses.bundleMergeControlArtifact(ruleContext);
    ruleContext.getAnalysisEnvironment().registerAction(new WriteMergeBundleControlFileAction(
        ruleContext.getActionOwner(), bundleMergeControlArtifact, mergeControl.build()));
    ruleContext.getAnalysisEnvironment().registerAction(new SpawnAction.Builder(ruleContext)
        .setMnemonic("Generate app bundle")
        .setExecutable(ruleContext.getExecutablePrerequisite("$bundlemerge", Mode.HOST))
        .addArgument(bundleMergeControlArtifact.getExecPathString())
        .addInput(bundleMergeControlArtifact)
        .addInput(binaryOutput)
        .addInputs(infoplistFiles)
        .addInputs(maybeActoolOutputZip)
        .addInputs(provisioningProfile.asSet())
        .addInputs(BundleableFile.toBundleMergeInputs(objcProvider.get(BUNDLE_FILE)))
        .addInputs(Xcdatamodel.outputZips(objcProvider.get(XCDATAMODEL)))
        .addOutput(ipaUnsigned)
        .build());

    ObjcActionsBuilder.registerAll(
        ruleContext, ObjcActionsBuilder.baseActions(ruleContext, objcProvider, xcodeProvider));
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

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    ObjcCommon info = ObjcCommon.fromContext(
        ruleContext, ImmutableList.<SdkFramework>of() /* extraSdkFrameworks */);
    checkAttributes(ruleContext, info);
    ObjcProvider objcProvider = info.getObjcProvider();
    XcodeProvider xcodeProvider = xcodeProvider(ruleContext, info);

    registerActions(ruleContext, objcProvider, xcodeProvider,
        ImmutableList.<String>of() /* extraLinkArgs */);

    return info.configuredTarget(
        NestedSetBuilder.<Artifact>stableOrder()
            .add(ruleContext.getImplicitOutputArtifact(ObjcBinaryRule.IPA))
            .add(ruleContext.getImplicitOutputArtifact(ObjcRuleClasses.PBXPROJ))
            .build(),
        xcodeProvider);
  }
}
