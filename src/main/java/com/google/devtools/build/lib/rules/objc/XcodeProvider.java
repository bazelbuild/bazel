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

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.BUNDLE_IMPORT_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FORCE_LOAD_FOR_XCODEGEN;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FRAMEWORK_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.GENERAL_RESOURCE_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.INCLUDE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_DYLIB;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.WEAK_SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCASSETS_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCDATAMODEL;

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.TransitiveInfoProvider;
import com.google.devtools.build.xcode.util.Interspersing;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.DependencyControl;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.TargetControl;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.XcodeprojBuildSetting;

/**
 * Provider which provides transitive dependency information that is specific to Xcodegen. In
 * particular, it provides a sequence of targets which can be used to create a self-contained
 * {@code .xcodeproj} file.
 */
@Immutable
final class XcodeProvider implements TransitiveInfoProvider {
  public static final class Builder {
    private Label label;
    private final ImmutableList.Builder<PathFragment> userHeaderSearchPaths =
        new ImmutableList.Builder<>();
    private Optional<InfoplistMerging> infoplistMerging = Optional.absent();
    private final NestedSetBuilder<XcodeProvider> dependencies = NestedSetBuilder.stableOrder();
    private final ImmutableList.Builder<XcodeprojBuildSetting> xcodeprojBuildSettings =
        new ImmutableList.Builder<>();
    private final ImmutableList.Builder<String> copts = new ImmutableList.Builder<>();
    private XcodeProductType productType;
    private final ImmutableList.Builder<Artifact> headers = new ImmutableList.Builder<>();
    private Optional<CompilationArtifacts> compilationArtifacts = Optional.absent();
    private ObjcProvider objcProvider;

    /**
     * Sets the label of the build target which corresponds to this Xcode target.
     */
    public Builder setLabel(Label label) {
      this.label = label;
      return this;
    }

    /**
     * Adds user header search paths for this target.
     */
    public Builder addUserHeaderSearchPaths(Iterable<PathFragment> userHeaderSearchPaths) {
      this.userHeaderSearchPaths.addAll(userHeaderSearchPaths);
      return this;
    }

    /**
     * Sets the Info.plist merging information. Used for applications. May be
     * absent for other bundles.
     */
    public Builder setInfoplistMerging(InfoplistMerging infoplistMerging) {
      this.infoplistMerging = Optional.of(infoplistMerging);
      return this;
    }

    /**
     * Adds {@link XcodeProvider}s corresponding to direct dependencies of this target which should
     * be added in the {@code .xcodeproj} file.
     */
    public Builder addDependencies(Iterable<XcodeProvider> dependencies) {
      for (XcodeProvider dependency : dependencies) {
        this.dependencies.add(dependency);
        this.dependencies.addTransitive(dependency.dependencies);
      }
      return this;
    }

    /**
     * Adds additional build settings of this target.
     */
    public Builder addXcodeprojBuildSettings(
        Iterable<XcodeprojBuildSetting> xcodeprojBuildSettings) {
      this.xcodeprojBuildSettings.addAll(xcodeprojBuildSettings);
      return this;
    }

    /**
     * Sets the copts to use when compiling the Xcode target.
     */
    public Builder addCopts(Iterable<String> copts) {
      this.copts.addAll(copts);
      return this;
    }

    /**
     * Sets the product type for the PBXTarget in the .xcodeproj file.
     */
    public Builder setProductType(XcodeProductType productType) {
      this.productType = productType;
      return this;
    }

    /**
     * Adds to the header files of this target. It needs not to include the header files of
     * dependencies.
     */
    public Builder addHeaders(Iterable<Artifact> headers) {
      this.headers.addAll(headers);
      return this;
    }

    /**
     * The compilation artifacts for this target.
     */
    public Builder setCompilationArtifacts(CompilationArtifacts compilationArtifacts) {
      this.compilationArtifacts = Optional.of(compilationArtifacts);
      return this;
    }

    /**
     * Sets the {@link ObjcProvider} corresponding to this target.
     */
    public Builder setObjcProvider(ObjcProvider objcProvider) {
      this.objcProvider = objcProvider;
      return this;
    }

    public XcodeProvider build() {
      return new XcodeProvider(this);
    }
  }

  private final Label label;
  private final ImmutableList<PathFragment> userHeaderSearchPaths;
  private final Optional<InfoplistMerging> infoplistMerging;
  private final NestedSet<XcodeProvider> dependencies;
  private final ImmutableList<XcodeprojBuildSetting> xcodeprojBuildSettings;
  private final ImmutableList<String> copts;
  private final XcodeProductType productType;
  private final ImmutableList<Artifact> headers;
  private final Optional<CompilationArtifacts> compilationArtifacts;
  private final ObjcProvider objcProvider;

  private XcodeProvider(Builder builder) {
    this.label = Preconditions.checkNotNull(builder.label);
    this.userHeaderSearchPaths = builder.userHeaderSearchPaths.build();
    this.infoplistMerging = builder.infoplistMerging;
    this.dependencies = builder.dependencies.build();
    this.xcodeprojBuildSettings = builder.xcodeprojBuildSettings.build();
    this.copts = builder.copts.build();
    this.productType = Preconditions.checkNotNull(builder.productType);
    this.headers = builder.headers.build();
    this.compilationArtifacts = builder.compilationArtifacts;
    this.objcProvider = Preconditions.checkNotNull(builder.objcProvider);
  }

  /**
   * Returns all the target controls that must be added to the xcodegen control. No other target
   * controls are needed to generate a functional project file. This method creates a new list
   * whenever it is called.
   */
  ImmutableList<TargetControl> targets() {
    ImmutableList.Builder<TargetControl> controls = new ImmutableList.Builder<>();
    controls.add(targetControl());
    for (XcodeProvider dependency : dependencies) {
      controls.add(dependency.targetControl());
    }
    return controls.build();
  }

  private TargetControl targetControl() {
    // TODO(bazel-team): Add provisioning profile information when Xcodegen supports it.
    // TODO(bazel-team): Add -force_load information when Xcodegen supports it.
    TargetControl.Builder targetControl = TargetControl.newBuilder()
        .setName(label.getName())
        .setLabel(label.toString())
        .setProductType(productType.getIdentifier())
        .addAllImportedLibrary(Artifact.toExecPaths(objcProvider.get(IMPORTED_LIBRARY)))
        .addAllUserHeaderSearchPath(PathFragment.safePathStrings(userHeaderSearchPaths))
        .addAllHeaderSearchPath(PathFragment.safePathStrings(objcProvider.get(INCLUDE)))
        .addAllHeaderFile(Artifact.toExecPaths(headers))
        // TODO(bazel-team): Add all build settings information once Xcodegen supports it.
        .addAllCopt(copts)
        .addAllLinkopt(
            Interspersing.beforeEach("-force_load", objcProvider.get(FORCE_LOAD_FOR_XCODEGEN)))
        .addAllLinkopt(IosSdkCommands.DEFAULT_LINKER_FLAGS)
        .addAllLinkopt(Interspersing.beforeEach(
            "-weak_framework", SdkFramework.names(objcProvider.get(WEAK_SDK_FRAMEWORK))))
        .addAllBuildSetting(xcodeprojBuildSettings)
        .addAllBuildSetting(IosSdkCommands.defaultWarningsForXcode())
        .addAllSdkFramework(SdkFramework.names(objcProvider.get(SDK_FRAMEWORK)))
        .addAllFramework(PathFragment.safePathStrings(objcProvider.get(FRAMEWORK_DIR)))
        .addAllXcassetsDir(PathFragment.safePathStrings(objcProvider.get(XCASSETS_DIR)))
        .addAllXcdatamodel(PathFragment.safePathStrings(
            Xcdatamodel.xcdatamodelDirs(objcProvider.get(XCDATAMODEL))))
        .addAllBundleImport(PathFragment.safePathStrings(objcProvider.get(BUNDLE_IMPORT_DIR)))
        .addAllSdkDylib(objcProvider.get(SDK_DYLIB))
        .addAllGeneralResourceFile(Artifact.toExecPaths(objcProvider.get(GENERAL_RESOURCE_FILE)));

    if ((productType == XcodeProductType.APPLICATION) || (productType == XcodeProductType.BUNDLE)) {
      for (XcodeProvider dependency : dependencies) {
        // Only add a target to a binary's dependencies if it has source files to compile. Xcode
        // cannot build targets without a source file in the PBXSourceFilesBuildPhase, so if such a
        // target is present in the control file, it is only to get Xcodegen to put headers and
        // resources not used by the final binary in the Project Navigator.
        for (CompilationArtifacts artifacts : dependency.compilationArtifacts.asSet()) {
          if (artifacts.getArchive().isPresent()) {
            targetControl.addDependency(DependencyControl.newBuilder()
                .setTargetLabel(dependency.label.toString())
                .build());
          }
        }
      }
    }

    for (InfoplistMerging merging : infoplistMerging.asSet()) {
      for (Artifact infoplist : merging.getPlistWithEverything().asSet()) {
        targetControl.setInfoplist(infoplist.getExecPathString());
      }
    }
    for (CompilationArtifacts artifacts : compilationArtifacts.asSet()) {
      targetControl
          .addAllSourceFile(Artifact.toExecPaths(artifacts.getSrcs()))
          .addAllNonArcSourceFile(Artifact.toExecPaths(artifacts.getNonArcSrcs()));

      for (Artifact pchFile : artifacts.getPchFile().asSet()) {
        targetControl
            .setPchPath(pchFile.getExecPathString())
            .addHeaderFile(pchFile.getExecPathString());
      }
    }
    return targetControl.build();
  }
}

