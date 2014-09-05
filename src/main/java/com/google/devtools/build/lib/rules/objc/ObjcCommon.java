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

import static com.google.devtools.build.lib.rules.objc.ArtifactListAttribute.ARCHIVES;
import static com.google.devtools.build.lib.rules.objc.ArtifactListAttribute.HDRS;
import static com.google.devtools.build.lib.rules.objc.ArtifactListAttribute.NON_ARC_SRCS;
import static com.google.devtools.build.lib.rules.objc.ArtifactListAttribute.SRCS;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.BUNDLE_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.HEADER;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCASSETS_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCDATAMODEL;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.Runfiles;
import com.google.devtools.build.lib.view.RunfilesProvider;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.BuildSetting;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.DependencyControl;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.TargetControl;

/**
 * Contains information common to multiple objc_* rules, and provides a unified API for extracting
 * and accessing it.
 */
final class ObjcCommon {
  @VisibleForTesting
  static final String NOT_IN_ASSETS_DIR_ERROR_FORMAT = "The following files are specified by the "
      + "asset_catalogs attribute but do not have a parent or ancestor directory named "
      + "*.xcassets: %s";

  @VisibleForTesting
  static final String NOT_IN_XCDATAMODEL_DIR_ERROR_FORMAT = "The following files are specified by "
      + "the datamodels attribute but do not have a parent or ancestor directory named "
      + "*.xcdatamodel[d]: %s";

  @VisibleForTesting
  static final String REQUIRES_AT_LEAST_ONE_LIBRARY_OR_SOURCE_FILE = "At least one library "
      + "dependency or source file is required.";

  private final RuleContext context;
  private final AssetCatalogsInfo assetCatalogsInfo;
  private final ObjcProvider objcProvider;
  private final XcdatamodelsInfo xcdatamodelsInfo;

  private ObjcCommon(RuleContext context, AssetCatalogsInfo assetCatalogsInfo,
      ObjcProvider objcProvider, XcdatamodelsInfo xcdatamodelsInfo) {
    this.context = Preconditions.checkNotNull(context);
    this.assetCatalogsInfo = Preconditions.checkNotNull(assetCatalogsInfo);
    this.objcProvider = Preconditions.checkNotNull(objcProvider);
    this.xcdatamodelsInfo = Preconditions.checkNotNull(xcdatamodelsInfo);
  }

  public RuleContext getContext() {
    return context;
  }

  public AssetCatalogsInfo getAssetCatalogsInfo() {
    return assetCatalogsInfo;
  }

  public ObjcProvider getObjcProvider() {
    return objcProvider;
  }

  /**
   * Reports any known errors to the {@link RuleContext}. This should be called exactly once for
   * a target.
   */
  public void reportErrors() {
    if (!Iterables.isEmpty(assetCatalogsInfo.getNotInXcassetsDir())) {
      context.attributeError("asset_catalogs",
          String.format(NOT_IN_ASSETS_DIR_ERROR_FORMAT,
              Joiner.on(" ").join(Artifact.toExecPaths(assetCatalogsInfo.getNotInXcassetsDir()))));
    }

    if (!xcdatamodelsInfo.getNotInXcdatamodelDir().isEmpty()) {
      context.attributeError("datamodels",
          String.format(NOT_IN_XCDATAMODEL_DIR_ERROR_FORMAT,
              Joiner.on(" ")
                  .join(Artifact.toExecPaths(xcdatamodelsInfo.getNotInXcdatamodelDir()))));
    }

    // TODO(bazel-team): This requirement doesn't make sense in light of libraries that only export
    // resources. See what is the most reasonable requirement we can put here is.
    if (objcProvider.get(LIBRARY).isEmpty() && objcProvider.get(IMPORTED_LIBRARY).isEmpty()) {
      context.ruleError(REQUIRES_AT_LEAST_ONE_LIBRARY_OR_SOURCE_FILE);
    }
  }

  /**
   * Builds an {@code XcodeProvider} which supplies the targets of all transitive dependencies and
   * one Xcode target corresponding to this one.
   */
  private XcodeProvider xcodeProvider(TargetControl thisTarget) {
    NestedSetBuilder<TargetControl> result = NestedSetBuilder.stableOrder();
    for (XcodeProvider depProvider : ObjcRuleClasses.deps(context, XcodeProvider.class)) {
      result.addTransitive(depProvider.getTargets());
    }
    return new XcodeProvider(result.add(thisTarget).build());
  }

  /**
   * Returns an {@code XcodeProvider} for this target.
   * @param maybeInfoplistFile the Info.plist file. Used for applications.
   * @param xcodeDependencies dependencies of the target for this rule in the .xcodeproj file.
   * @param buildSettings additional build settings of this target.
   */
  public XcodeProvider xcodeProvider(
      Optional<Artifact> maybeInfoplistFile,
      Optional<Artifact> maybePchFile,
      Iterable<DependencyControl> xcodeDependencies,
      Iterable<BuildSetting> buildSettings) {
    // TODO(bazel-team): Add provisioning profile information when Xcodegen supports it.
    TargetControl.Builder targetControl = TargetControl.newBuilder()
        .setName(context.getLabel().getName())
        .setLabel(context.getLabel().toString())
        .addAllImportedLibrary(Artifact.toExecPaths(objcProvider.get(IMPORTED_LIBRARY)))
        .addAllHeaderFile(Artifact.toExecPaths(HDRS.get(context)))
        .addAllHeaderFile(Artifact.toExecPaths(maybePchFile.asSet()))
        .addAllSourceFile(Artifact.toExecPaths(SRCS.get(context)))
        .addAllNonArcSourceFile(Artifact.toExecPaths(NON_ARC_SRCS.get(context)))
        .addAllCopt(ObjcRuleClasses.copts(context))
        .addAllBuildSetting(buildSettings)
        .addAllSdkFramework(SdkFramework.names(objcProvider.get(SDK_FRAMEWORK)))
        .addAllXcassetsDir(PathFragment.safePathStrings(objcProvider.get(XCASSETS_DIR)))
        .addAllXcdatamodel(PathFragment.safePathStrings(
            Xcdatamodel.xcdatamodelDirs(objcProvider.get(XCDATAMODEL))))
        .addAllDependency(xcodeDependencies);
    for (BundleableFile file : objcProvider.get(BUNDLE_FILE)) {
      targetControl.addGeneralResourceFile(file.getOriginal().getExecPathString());
    }
    for (Artifact infoplistFile : maybeInfoplistFile.asSet()) {
      targetControl.setInfoplist(infoplistFile.getExecPathString());
    }
    for (Artifact pchFile : maybePchFile.asSet()) {
      targetControl.addBuildSetting(BuildSetting.newBuilder()
          .setName("GCC_PREFIX_HEADER")
          .setValue(pchFile.getExecPathString())
          .build());
    }
    return xcodeProvider(targetControl.build());
  }

  /**
   * Returns an instance based on the rule specified by {@code context}. Also registers the extra
   * {@code SdkFramework}s specified by {@code extraSdkFrameworks}.
   */
  public static ObjcCommon fromContext(RuleContext context,
      Iterable<SdkFramework> extraSdkFrameworks) {
    AssetCatalogsInfo assetCatalogsInfo = AssetCatalogsInfo.fromRule(context);
    XcdatamodelsInfo xcdatamodelsInfo = XcdatamodelsInfo.fromRule(context);

    ObjcProvider objcProvider = new ObjcProvider.Builder()
        .addAll(HEADER, HDRS.get(context))
        .add(assetCatalogsInfo)
        .addAll(LIBRARY, ObjcRuleClasses.outputAFile(context).asSet())
        .addAll(IMPORTED_LIBRARY, ARCHIVES.get(context))
        .addAll(BUNDLE_FILE, BundleableFile.resourceFilesFromRule(context))
        .addAll(BUNDLE_FILE, BundleableFile.stringsFilesFromRule(context))
        .addAll(BUNDLE_FILE, BundleableFile.xibFilesFromRule(context))
        .addAll(SDK_FRAMEWORK, ObjcRuleClasses.sdkFrameworks(context))
        .addAll(SDK_FRAMEWORK, extraSdkFrameworks)
        .addAll(XCDATAMODEL, xcdatamodelsInfo.getXcdatamodels())
        .addTransitive(ObjcRuleClasses.deps(context, ObjcProvider.class))
        .build();

    return new ObjcCommon(context, assetCatalogsInfo, objcProvider, xcdatamodelsInfo);
  }

  /**
   * @param filesToBuild files to build for this target. These also become the data runfiles.
   * @param targetProvider the {@code XcodeTargetProvider} for this target
   */
  public ConfiguredTarget configuredTarget(NestedSet<Artifact> filesToBuild,
      XcodeProvider targetProvider) {
    RunfilesProvider runfilesProvider = RunfilesProvider.withData(
        new Runfiles.Builder()
            .addRunfiles(context, RunfilesProvider.DEFAULT_RUNFILES)
            .build(),
        new Runfiles.Builder().addArtifacts(filesToBuild).build());
    NestedSet<Artifact> allInputs = NestedSetBuilder.<Artifact>stableOrder()
        .addAll(NON_ARC_SRCS.get(context))
        .addAll(SRCS.get(context))
        .addAll(HDRS.get(context))
        .addAll(ARCHIVES.get(context))
        .build();

    return new RuleConfiguredTargetBuilder(context)
        .setFilesToBuild(filesToBuild)
        .add(RunfilesProvider.class, runfilesProvider)
        .addProvider(ObjcProvider.class, objcProvider)
        .addProvider(XcodeProvider.class, targetProvider)
        // TODO(bazel-team): Remove this when legacy dependencies have been removed.
        .addProvider(
            LegacyObjcSourceFileProvider.class, new LegacyObjcSourceFileProvider(allInputs))
        .build();
  }
}
