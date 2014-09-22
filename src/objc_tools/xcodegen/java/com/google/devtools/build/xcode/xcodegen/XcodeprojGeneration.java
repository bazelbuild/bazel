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

package com.google.devtools.build.xcode.xcodegen;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static com.google.devtools.build.xcode.common.BuildOptionsUtil.DEFAULT_OPTIONS_NAME;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.xcode.common.XcodeprojPath;
import com.google.devtools.build.xcode.util.Equaling;
import com.google.devtools.build.xcode.util.Mapping;
import com.google.devtools.build.xcode.xcodegen.SourceFile.BuildType;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.Control;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.DependencyControl;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.TargetControl;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.XcodeprojBuildSetting;

import com.dd.plist.NSArray;
import com.dd.plist.NSDictionary;
import com.dd.plist.NSObject;
import com.dd.plist.NSString;
import com.facebook.buck.apple.xcode.GidGenerator;
import com.facebook.buck.apple.xcode.XcodeprojSerializer;
import com.facebook.buck.apple.xcode.xcodeproj.PBXBuildFile;
import com.facebook.buck.apple.xcode.xcodeproj.PBXContainerItemProxy.ProxyType;
import com.facebook.buck.apple.xcode.xcodeproj.PBXFileReference;
import com.facebook.buck.apple.xcode.xcodeproj.PBXFrameworksBuildPhase;
import com.facebook.buck.apple.xcode.xcodeproj.PBXNativeTarget;
import com.facebook.buck.apple.xcode.xcodeproj.PBXProject;
import com.facebook.buck.apple.xcode.xcodeproj.PBXReference;
import com.facebook.buck.apple.xcode.xcodeproj.PBXReference.SourceTree;
import com.facebook.buck.apple.xcode.xcodeproj.PBXSourcesBuildPhase;
import com.facebook.buck.apple.xcode.xcodeproj.PBXTarget.ProductType;
import com.facebook.buck.apple.xcode.xcodeproj.PBXTargetDependency;

import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystem;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;

/**
 * Utility code for generating Xcode project files.
 */
public class XcodeprojGeneration {
  public static final String FILE_TYPE_ARCHIVE_LIBRARY = "archive.ar";
  public static final String FILE_TYPE_WRAPPER_APPLICATION = "wrapper.application";

  @VisibleForTesting
  static final String APP_NEEDS_SOURCE_ERROR =
      "Due to limitations in Xcode, application projects must have at least one source file.";

  private XcodeprojGeneration() {
    throw new UnsupportedOperationException("static-only");
  }

  /**
   * Writes a project to an {@code OutputStream} in the correct encoding.
   */
  public static void write(OutputStream out, PBXProject project) throws IOException {
    XcodeprojSerializer ser = new XcodeprojSerializer(
        new GidGenerator(ImmutableSet.<String>of()), project);
    Writer outWriter = new OutputStreamWriter(out, StandardCharsets.UTF_8);
    // toXMLPropertyList includes an XML encoding specification (UTF-8), which we specify above.
    // Standard Xcodeproj files use the toASCIIPropertyList format, but Xcode will rewrite
    // XML-encoded project files automatically when first opening them. We use XML to prevent
    // encoding issues, since toASCIIPropertyList does not include the UTF-8 encoding comment, and
    // Xcode by default apparently uses MacRoman.
    // This encoding concern is probably why Buck also generates XML project files as well.
    outWriter.write(ser.toPlist().toXMLPropertyList());
    outWriter.flush();
  }

  public static boolean isApp(TargetControl targetControl) {
    return targetControl.hasInfoplist();
  }

  private static class TargetInfo {
    final TargetControl control;
    final PBXNativeTarget nativeTarget;
    final PBXFrameworksBuildPhase frameworksPhase;
    final PBXBuildFile productBuildFile;
    final PBXTargetDependency targetDependency;

    TargetInfo(TargetControl control,
        PBXNativeTarget nativeTarget,
        PBXFrameworksBuildPhase frameworksPhase,
        PBXBuildFile productBuildFile,
        PBXTargetDependency targetDependency) {
      this.control = control;
      this.nativeTarget = nativeTarget;
      this.frameworksPhase = frameworksPhase;
      this.productBuildFile = productBuildFile;
      this.targetDependency = targetDependency;
    }
  }

  private static String google3labelToXcodeTargetName(String label) {
    return label.replace("//", "").replace("/", "_").replace(":", "_");
  }

  private static NSDictionary nonArcCompileSettings() {
    NSDictionary result = new NSDictionary();
    result.put("COMPILER_FLAGS", "-fno-objc-arc");
    return result;
  }

  private static boolean hasAtLeastOneCompilableSource(TargetControl control) {
    return (control.getSourceFileCount() != 0) || (control.getNonArcSourceFileCount() != 0);
  }

  /**
   * Returns the final header search paths to be placed in a build configuration.
   * @param sourceRoot path from the client root to the .xcodeproj containing directory
   * @param paths the header search paths, relative to client root
   * @param resolvedPaths paths to add as-is to the end of the returned array
   * @return an {@link NSArray} of the paths, each relative to the source root
   */
  private static NSArray headerSearchPaths(Path sourceRoot, Iterable<String> paths,
      String... resolvedPaths) {
    ImmutableList.Builder<String> result = new ImmutableList.Builder<>();
    for (String path : paths) {
      result.add(sourceRoot.resolve(path).toString());
    }
    result.add(resolvedPaths);
    return (NSArray) NSObject.wrap(result.build());
  }

  /** Generates a project file. */
  public static PBXProject xcodeproj(Path root, Control control, Grouper grouper) {
    checkArgument(control.hasPbxproj(), "Must set pbxproj field on control proto.");
    FileSystem fileSystem = root.getFileSystem();

    XcodeprojPath<Path> outputPath = XcodeprojPath.converter().fromPath(
        RelativePaths.fromString(fileSystem, control.getPbxproj()));

    int levelsToExecRoot = outputPath.getXcodeprojContainerDir().getNameCount();
    Path sourceRoot = fileSystem.getPath(Joiner
        .on('/')
        .join(Collections.nCopies(levelsToExecRoot, "..")));

    NSDictionary projBuildConfigMap = new NSDictionary();
    projBuildConfigMap.put("ARCHS", new NSArray(
        new NSString("arm7"), new NSString("arm7s"), new NSString("arm64")));
    projBuildConfigMap.put("CLANG_ENABLE_OBJC_ARC", "YES");
    projBuildConfigMap.put("SDKROOT", "iphoneos");
    projBuildConfigMap.put("GCC_WARN_64_TO_32_BIT_CONVERSION", "YES");
    projBuildConfigMap.put("IPHONEOS_DEPLOYMENT_TARGET", "7.0");
    projBuildConfigMap.put("GCC_VERSION", "com.apple.compilers.llvm.clang.1_0");

    PBXProject project = new PBXProject(outputPath.getProjectName());
    project.getMainGroup().setPath(sourceRoot.toString());
    try {
      project
          .getBuildConfigurationList()
          .getBuildConfigurationsByName()
          .get(DEFAULT_OPTIONS_NAME)
          .setBuildSettings(projBuildConfigMap);
    } catch (ExecutionException e) {
      throw new RuntimeException(e);
    }

    Map<String, TargetInfo> targetInfoByLabel = new HashMap<>();

    PBXFileReferences fileReferences = new PBXFileReferences();
    SdkFrameworkObjects frameworkObjects = new SdkFrameworkObjects(fileReferences);
    PBXBuildFiles pbxBuildFiles = new PBXBuildFiles(fileReferences);
    Resources resources =
        Resources.fromTargetControls(fileSystem, pbxBuildFiles, control.getTargetList());
    Xcdatamodels xcdatamodels =
        Xcdatamodels.fromTargetControls(fileSystem, pbxBuildFiles, control.getTargetList());
    // We use a hash set for the Project Navigator files so that the same PBXFileReference does not
    // get added twice. Because PBXFileReference uses equality-by-identity semantics, this requires
    // the PBXFileReferences cache to properly return the same reference for functionally-equivalent
    // files.
    Set<PBXReference> ungroupedProjectNavigatorFiles = new LinkedHashSet<>();
    for (TargetControl targetControl : control.getTargetList()) {
      checkArgument(targetControl.hasName(), "TargetControl requires a name: %s", targetControl);
      checkArgument(targetControl.hasLabel(), "TargetControl requires a label: %s", targetControl);

      String productName = targetControl.getName();

      boolean isApp = isApp(targetControl);
      Preconditions.checkArgument(!isApp || hasAtLeastOneCompilableSource(targetControl),
          APP_NEEDS_SOURCE_ERROR);
      PBXSourcesBuildPhase sourcesBuildPhase = new PBXSourcesBuildPhase();

      for (SourceFile source : SourceFile.allSourceFiles(fileSystem, targetControl)) {
        PBXFileReference fileRef =
            fileReferences.get(FileReference.of(source.path().toString(), SourceTree.GROUP));
        ungroupedProjectNavigatorFiles.add(fileRef);
        if (Equaling.of(source.buildType(), BuildType.NO_BUILD)) {
          continue;
        }
        PBXBuildFile buildFile = new PBXBuildFile(fileRef);
        if (Equaling.of(source.buildType(), BuildType.NON_ARC_BUILD)) {
          buildFile.setSettings(Optional.of(nonArcCompileSettings()));
        }
        sourcesBuildPhase.getFiles().add(buildFile);
      }
      sourcesBuildPhase.getFiles().addAll(xcdatamodels.buildFiles().get(targetControl));

      PBXFileReference productReference;
      if (isApp) {
        productReference = fileReferences.get(
            FileReference.of(String.format("%s.app", productName), SourceTree.BUILT_PRODUCTS_DIR)
                .withExplicitFileType(FILE_TYPE_WRAPPER_APPLICATION));
      } else {
        productReference = fileReferences.get(
            FileReference.of(String.format("lib%s.a", productName), SourceTree.BUILT_PRODUCTS_DIR)
                .withExplicitFileType(FILE_TYPE_ARCHIVE_LIBRARY));
      }
      ungroupedProjectNavigatorFiles.add(productReference);

      NSDictionary targetBuildConfigMap = new NSDictionary();
      // TODO(bazel-team): Stop adding the sourceRoot automatically once the
      // released version of Bazel starts passing it.
      targetBuildConfigMap.put("USER_HEADER_SEARCH_PATHS",
          headerSearchPaths(
              sourceRoot, targetControl.getUserHeaderSearchPathList(), sourceRoot.toString()));
      targetBuildConfigMap.put("HEADER_SEARCH_PATHS",
          headerSearchPaths(sourceRoot, targetControl.getHeaderSearchPathList(), "$(inherited)"));

      targetBuildConfigMap.put("PRODUCT_NAME", productName);
      if (targetControl.hasInfoplist()) {
        Path relative = RelativePaths.fromString(fileSystem, targetControl.getInfoplist());
        targetBuildConfigMap.put("INFOPLIST_FILE", sourceRoot.resolve(relative).toString());
      }

      if (targetControl.getCoptCount() > 0) {
        targetBuildConfigMap.put("OTHER_CFLAGS", NSObject.wrap(targetControl.getCoptList()));
      }
      for (XcodeprojBuildSetting setting : targetControl.getBuildSettingList()) {
        targetBuildConfigMap.put(setting.getName(), setting.getValue());
      }

      PBXNativeTarget target =
          new PBXNativeTarget(google3labelToXcodeTargetName(targetControl.getLabel()));
      try {
        target
            .getBuildConfigurationList()
            .getBuildConfigurationsByName()
            .get(DEFAULT_OPTIONS_NAME)
            .setBuildSettings(targetBuildConfigMap);
      } catch (ExecutionException e) {
        throw new RuntimeException(e);
      }
      target.setProductType(isApp ? ProductType.IOS_BINARY : ProductType.IOS_LIBRARY);
      target.setProductReference(productReference);

      PBXFrameworksBuildPhase frameworksPhase =
          frameworkObjects.newBuildPhase(targetControl.getSdkFrameworkList());
      for (String importedArchive : targetControl.getImportedLibraryList()) {
        PBXFileReference fileReference = fileReferences.get(
            FileReference.of(importedArchive, SourceTree.GROUP)
                .withExplicitFileType(FILE_TYPE_ARCHIVE_LIBRARY));
        ungroupedProjectNavigatorFiles.add(fileReference);
        if (isApp) {
          frameworksPhase.getFiles().add(new PBXBuildFile(fileReference));
        }
      }

      project.getTargets().add(target);

      target.getBuildPhases().add(frameworksPhase);
      target.getBuildPhases().add(sourcesBuildPhase);
      target.getBuildPhases().addAll(resources.resourcesBuildPhase(targetControl).asSet());

      checkState(!Mapping.of(targetInfoByLabel, targetControl.getLabel()).isPresent(),
          "Mapping already exists for target with label %s in map: %s",
          targetControl.getLabel(), targetInfoByLabel);
      targetInfoByLabel.put(
          targetControl.getLabel(),
          new TargetInfo(
              targetControl,
              target,
              frameworksPhase,
              new PBXBuildFile(productReference),
              new LocalPBXTargetDependency(
                  new LocalPBXContainerItemProxy(
                      project, target, ProxyType.TARGET_REFERENCE))));
    }

    for (HasProjectNavigatorFiles references : ImmutableList.of(pbxBuildFiles, frameworkObjects)) {
      Iterables.addAll(ungroupedProjectNavigatorFiles, references.mainGroupReferences());
    }
    Iterables.addAll(
        project.getMainGroup().getChildren(),
        grouper.group(ungroupedProjectNavigatorFiles));
    for (TargetInfo targetInfo : targetInfoByLabel.values()) {
      for (DependencyControl dependency : targetInfo.control.getDependencyList()) {
        TargetInfo dependencyInfo =
            Mapping.of(targetInfoByLabel, dependency.getTargetLabel()).get();
        targetInfo.frameworksPhase.getFiles().add(dependencyInfo.productBuildFile);
        targetInfo.nativeTarget.getDependencies().add(dependencyInfo.targetDependency);
      }
    }

    return project;
  }
}
