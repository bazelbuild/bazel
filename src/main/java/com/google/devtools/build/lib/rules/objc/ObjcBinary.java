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

import static com.google.devtools.build.lib.rules.objc.IosSdkCommands.BIN_DIR;
import static com.google.devtools.build.lib.rules.objc.IosSdkCommands.MINIMUM_OS_VERSION;
import static com.google.devtools.build.lib.rules.objc.IosSdkCommands.TARGET_DEVICE_FAMILIES;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.BUNDLE_FILE;
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
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.BuildSetting;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.DependencyControl;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.TargetControl;

import java.util.List;

/**
 * Implementation for the "objc_binary" rule.
 */
public class ObjcBinary implements RuleConfiguredTargetFactory {

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
    ImmutableList.Builder<BuildSetting> buildSettings = new ImmutableList.Builder<>();
    for (String appIcon : ObjcBinaryRule.appIcon(ruleContext).asSet()) {
      buildSettings.add(BuildSetting.newBuilder()
          .setName("ASSETCATALOG_COMPILER_APPICON_NAME")
          .setValue(appIcon)
          .build());
    }
    for (String launchImage : ObjcBinaryRule.launchImage(ruleContext).asSet()) {
      buildSettings.add(BuildSetting.newBuilder()
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

    // TODO(bazel-team): See if this can be converted to a direct call to ld. It is possible that
    // the code will be cleaner or clearer if we do so.
    ruleContext.getAnalysisEnvironment().registerAction(new SpawnAction.Builder(ruleContext)
        .setMnemonic("Link")
        .setExecutable(new PathFragment(BIN_DIR + "/clang"))
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

    BundleMergeProtos.Control.Builder mergeControl = BundleMergeProtos.Control.newBuilder()
        .addBundleFile(BundleMergeProtos.BundleFile.newBuilder()
            .setSourceFile(binaryOutput.getExecPathString())
            .setBundlePath(ruleContext.getTarget().getName())
            .build())
        .addAllBundleFile(BundleableFile.toBundleFiles(objcProvider.get(BUNDLE_FILE)))
        .addAllSourcePlistFile(Artifact.toExecPaths(infoplistFiles))
        .setOutFile(ipaOutput.getExecPathString())
        // TODO(bazel-team): Add rule attributes for specifying targeted device family and minimum
        // OS version.
        .setMinimumOsVersion(MINIMUM_OS_VERSION)
        .setSdkVersion(objcConfiguration.getIosSdkVersion())
        .setPlatform(objcConfiguration.getPlatform().name())
        .setBundleRoot(ObjcBinaryRule.bundleRoot(ruleContext));
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
    if (objcConfiguration.getPlatform() == Platform.DEVICE) {
      if (!provisioningProfile.isPresent()) {
        ruleContext.attributeError(ObjcBinaryRule.PROVISIONING_PROFILE_ATTR,
            "must specify provisioning profile for device build");
      } else {
        mergeControl.addBundleFile(BundleMergeProtos.BundleFile.newBuilder()
            .setSourceFile(provisioningProfile.get().getExecPathString())
            .setBundlePath(PROVISIONING_PROFILE_BUNDLE_FILE)
            .build());
      }
    } else if (ruleContext.attributes().isAttributeValueExplicitlySpecified(
        ObjcBinaryRule.EXPLICIT_PROVISIONING_PROFILE_ATTR)) {
      ruleContext.attributeError(ObjcBinaryRule.EXPLICIT_PROVISIONING_PROFILE_ATTR,
          SIMULATOR_PROVISIONING_PROFILE_ERROR);
    }

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
        .addOutput(ipaOutput)
        .build());

    ObjcActionsBuilder.registerAll(
        ruleContext, ObjcActionsBuilder.baseActions(ruleContext, objcProvider, xcodeProvider));
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
