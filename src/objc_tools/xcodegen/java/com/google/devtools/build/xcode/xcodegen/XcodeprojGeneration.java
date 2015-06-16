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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.escape.Escaper;
import com.google.common.escape.Escapers;
import com.google.devtools.build.xcode.common.XcodeprojPath;
import com.google.devtools.build.xcode.util.Containing;
import com.google.devtools.build.xcode.util.Equaling;
import com.google.devtools.build.xcode.util.Mapping;
import com.google.devtools.build.xcode.xcodegen.LibraryObjects.BuildPhaseBuilder;
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
import com.facebook.buck.apple.xcode.xcodeproj.PBXCopyFilesBuildPhase;
import com.facebook.buck.apple.xcode.xcodeproj.PBXFileReference;
import com.facebook.buck.apple.xcode.xcodeproj.PBXFrameworksBuildPhase;
import com.facebook.buck.apple.xcode.xcodeproj.PBXNativeTarget;
import com.facebook.buck.apple.xcode.xcodeproj.PBXProject;
import com.facebook.buck.apple.xcode.xcodeproj.PBXReference;
import com.facebook.buck.apple.xcode.xcodeproj.PBXReference.SourceTree;
import com.facebook.buck.apple.xcode.xcodeproj.PBXResourcesBuildPhase;
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
import java.nio.file.Paths;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;

/**
 * Utility code for generating Xcode project files.
 */
public class XcodeprojGeneration {
  public static final String FILE_TYPE_ARCHIVE_LIBRARY = "archive.ar";
  public static final String FILE_TYPE_WRAPPER_APPLICATION = "wrapper.application";
  public static final String FILE_TYPE_WRAPPER_BUNDLE = "wrapper.cfbundle";
  public static final String FILE_TYPE_APP_EXTENSION = "wrapper.app-extension";
  private static final String DEFAULT_OPTIONS_NAME = "Debug";
  private static final Escaper QUOTE_ESCAPER = Escapers.builder().addEscape('"', "\\\"").build();

  @VisibleForTesting
  static final String APP_NEEDS_SOURCE_ERROR =
      "Due to limitations in Xcode, application projects must have at least one source file.";

  private XcodeprojGeneration() {
    throw new UnsupportedOperationException("static-only");
  }

  /**
   * Determines the relative path to the workspace root from the path of the project.pbxproj output
   * file. An absolute path is preferred if available.
   */
  static Path relativeWorkspaceRoot(Path pbxproj) {
    int levelsToExecRoot = pbxproj.getParent().getParent().getNameCount();
    return pbxproj.getFileSystem().getPath(Joiner
        .on('/')
        .join(Collections.nCopies(levelsToExecRoot, "..")));
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

  private static final EnumSet<ProductType> SUPPORTED_PRODUCT_TYPES = EnumSet.of(
      ProductType.STATIC_LIBRARY,
      ProductType.APPLICATION,
      ProductType.BUNDLE,
      ProductType.UNIT_TEST,
      ProductType.APP_EXTENSION);

  private static final EnumSet<ProductType> PRODUCT_TYPES_THAT_HAVE_A_BINARY = EnumSet.of(
      ProductType.APPLICATION,
      ProductType.BUNDLE,
      ProductType.UNIT_TEST,
      ProductType.APP_EXTENSION);

  /**
   * Detects the product type of the given target based on multiple fields in {@code targetControl}.
   * {@code productType} is set as a field on {@code PBXNativeTarget} objects in Xcode project
   * files, and we support three values: {@link ProductType#APPLICATION},
   * {@link ProductType#STATIC_LIBRARY}, and {@link ProductType#BUNDLE}. The product type is not
   * only what xcodegen sets the {@code productType} field to - it also dictates what can be built
   * with this target (e.g. a library cannot be built with resources), what build phase it should be
   * added to of its dependers, and the name and shape of its build output.
   */
  public static ProductType productType(TargetControl targetControl) {
    if (targetControl.hasProductType()) {
      for (ProductType supportedType : SUPPORTED_PRODUCT_TYPES) {
        if (targetControl.getProductType().equals(supportedType.identifier)) {
          return supportedType;
        }
      }
      throw new IllegalArgumentException(
          "Unsupported product type: " + targetControl.getProductType());
    }

    return targetControl.hasInfoplist() ? ProductType.APPLICATION : ProductType.STATIC_LIBRARY;
  }

  private static String productName(TargetControl targetControl) {
    if (Equaling.of(ProductType.STATIC_LIBRARY, productType(targetControl))) {
      // The product names for static libraries must be unique since the final
      // binary is linked with "clang -l${LIBRARY_PRODUCT_NAME}" for each static library.
      // Unlike other product types, a full application may have dozens of static libraries,
      // so rather than just use the target name, we use the full label to generate the product
      // name.
      return labelToXcodeTargetName(targetControl.getLabel());
    } else {
      return targetControl.getName();
    }
  }

  /**
   * Returns the file reference corresponding to the {@code productReference} of the given target.
   * The {@code productReference} is the build output of a target, and its name and file type
   * (stored in the {@link FileReference}) change based on the product type.
   */
  private static FileReference productReference(TargetControl targetControl) {
    ProductType type = productType(targetControl);
    String productName = productName(targetControl);

    switch (type) {
      case APPLICATION:
        return FileReference.of(String.format("%s.app", productName), SourceTree.BUILT_PRODUCTS_DIR)
            .withExplicitFileType(FILE_TYPE_WRAPPER_APPLICATION);
      case STATIC_LIBRARY:
        return FileReference.of(
            String.format("lib%s.a", productName), SourceTree.BUILT_PRODUCTS_DIR)
                .withExplicitFileType(FILE_TYPE_ARCHIVE_LIBRARY);
      case BUNDLE:
        return FileReference.of(
            String.format("%s.bundle", productName), SourceTree.BUILT_PRODUCTS_DIR)
                .withExplicitFileType(FILE_TYPE_WRAPPER_BUNDLE);
      case UNIT_TEST:
        return FileReference.of(
            String.format("%s.xctest", productName), SourceTree.BUILT_PRODUCTS_DIR)
                .withExplicitFileType(FILE_TYPE_WRAPPER_BUNDLE);
      case APP_EXTENSION:
        return FileReference.of(
            String.format("%s.appex", productName), SourceTree.BUILT_PRODUCTS_DIR)
                .withExplicitFileType(FILE_TYPE_APP_EXTENSION);
      default:
        throw new IllegalArgumentException("unknown: " + type);
    }
  }

  private static class TargetInfo {
    final TargetControl control;
    final PBXNativeTarget nativeTarget;
    final PBXFrameworksBuildPhase frameworksPhase;
    final PBXResourcesBuildPhase resourcesPhase;
    final PBXBuildFile productBuildFile;
    final PBXTargetDependency targetDependency;
    final NSDictionary buildConfig;

    TargetInfo(TargetControl control,
        PBXNativeTarget nativeTarget,
        PBXFrameworksBuildPhase frameworksPhase,
        PBXResourcesBuildPhase resourcesPhase,
        PBXBuildFile productBuildFile,
        PBXTargetDependency targetDependency,
        NSDictionary buildConfig) {
      this.control = control;
      this.nativeTarget = nativeTarget;
      this.frameworksPhase = frameworksPhase;
      this.resourcesPhase = resourcesPhase;
      this.productBuildFile = productBuildFile;
      this.targetDependency = targetDependency;
      this.buildConfig = buildConfig;
    }

    /**
     * Returns the path to the built, statically-linked binary for this target. The path contains
     * build-setting variables and may be used in a build setting such as {@code TEST_HOST}.
     *
     * <p>One example return value is {@code $(BUILT_PRODUCTS_DIR)/Foo.app/Foo}.
     */
    String staticallyLinkedBinary() {
      ProductType type = productType(control);
      Preconditions.checkArgument(
          Containing.item(PRODUCT_TYPES_THAT_HAVE_A_BINARY, type),
          "This product type (%s) is not known to have a binary.", type);
      FileReference productReference = productReference(control);
      return String.format("$(%s)/%s/%s",
          productReference.sourceTree().name(),
          productReference.path().or(productReference.name()),
          control.getName());
    }

    /**
     * Adds the given dependency to the list of dependencies, the
     * appropriate build phase if applicable, and the appropriate build setting values if
     * applicable, of this target.
     */
    void addDependencyInfo(
        DependencyControl dependencyControl, Map<String, TargetInfo> targetInfoByLabel) {
      TargetInfo dependencyInfo =
          Mapping.of(targetInfoByLabel, dependencyControl.getTargetLabel()).get();
      if (dependencyControl.getTestHost()) {
        buildConfig.put("TEST_HOST", dependencyInfo.staticallyLinkedBinary());
        buildConfig.put("BUNDLE_LOADER", dependencyInfo.staticallyLinkedBinary());
      } else if (productType(dependencyInfo.control) == ProductType.BUNDLE) {
        resourcesPhase.getFiles().add(dependencyInfo.productBuildFile);
      } else if (productType(dependencyInfo.control) == ProductType.APP_EXTENSION) {
        PBXCopyFilesBuildPhase copyFilesPhase = new PBXCopyFilesBuildPhase(
            PBXCopyFilesBuildPhase.Destination.PLUGINS, /*path=*/"");
        copyFilesPhase.getFiles().add(dependencyInfo.productBuildFile);
        nativeTarget.getBuildPhases().add(copyFilesPhase);
      } else {
        frameworksPhase.getFiles().add(dependencyInfo.productBuildFile);
      }
      nativeTarget.getDependencies().add(dependencyInfo.targetDependency);
    }
  }

  // TODO(bazel-team): Make this a no-op once the released version of Bazel sends the label to
  // xcodegen pre-processed.
  private static String labelToXcodeTargetName(String label) {
    String pathFromWorkspaceRoot =  label.replace("//", "").replace(':', '/');
    List<String> components = Splitter.on('/').splitToList(pathFromWorkspaceRoot);
    return Joiner.on('_').join(Lists.reverse(components));
  }

  private static NSDictionary nonArcCompileSettings() {
    NSDictionary result = new NSDictionary();
    result.put("COMPILER_FLAGS", "-fno-objc-arc");
    return result;
  }

  private static boolean hasAtLeastOneCompilableSource(TargetControl control) {
    return (control.getSourceFileCount() != 0) || (control.getNonArcSourceFileCount() != 0);
  }

  private static <E> Iterable<E> plus(Iterable<E> before, E... rest) {
    return Iterables.concat(before, ImmutableList.copyOf(rest));
  }

  /**
   * Returns the final header search paths to be placed in a build configuration.
   */
  private static NSArray headerSearchPaths(Iterable<String> paths) {
    ImmutableList.Builder<String> result = new ImmutableList.Builder<>();
    for (String path : paths) {
      // TODO(bazel-team): Remove this hack once the released version of Bazel is prepending
      // "$(WORKSPACE_ROOT)/" to every "source rooted" path.
      if (!path.startsWith("$")) {
        path = "$(WORKSPACE_ROOT)/" + path;
      }
      result.add(path);
    }
    return (NSArray) NSObject.wrap(result.build());
  }

  /**
   * Returns the {@code FRAMEWORK_SEARCH_PATHS} array for a target's build config given the list of
   * {@code .framework} directory paths.
   */
  private static NSArray frameworkSearchPaths(Iterable<String> frameworks) {
    ImmutableSet.Builder<NSString> result = new ImmutableSet.Builder<>();
    for (String framework : frameworks) {
      result.add(new NSString("$(WORKSPACE_ROOT)/" + Paths.get(framework).getParent()));
    }
    // This is needed by XcTest targets (and others, just in case) for SenTestingKit.framework.
    result.add(new NSString("$(SDKROOT)/Developer/Library/Frameworks"));
    // This is needed by non-XcTest targets that use XcTest.framework, for instance for test
    // utility libraries packaged as an objc_library.
    result.add(new NSString("$(PLATFORM_DIR)/Developer/Library/Frameworks"));

    return (NSArray) NSObject.wrap(result.build().asList());
  }

  private static PBXFrameworksBuildPhase buildLibraryInfo(
      LibraryObjects libraryObjects, TargetControl target) {
    BuildPhaseBuilder builder = libraryObjects.newBuildPhase();
    if (Containing.item(PRODUCT_TYPES_THAT_HAVE_A_BINARY, productType(target))) {
      for (String dylib : target.getSdkDylibList()) {
        builder.addDylib(dylib);
      }
    }
    for (String sdkFramework : target.getSdkFrameworkList()) {
      builder.addSdkFramework(sdkFramework);
    }
    for (String framework : target.getFrameworkList()) {
      builder.addFramework(framework);
    }
    return builder.build();
  }

  private static ImmutableList<String> otherLdflags(TargetControl targetControl) {
    Iterable<String> givenFlags = targetControl.getLinkoptList();
    ImmutableList.Builder<String> flags = new ImmutableList.Builder<>();
    flags.addAll(givenFlags);
    if (!Equaling.of(ProductType.STATIC_LIBRARY, productType(targetControl))) {
      for (String importedLibrary : targetControl.getImportedLibraryList()) {
        flags.add("$(WORKSPACE_ROOT)/" + importedLibrary);
      }
    }
    return flags.build();
  }

  /** Generates a project file. */
  public static PBXProject xcodeproj(Path workspaceRoot, Control control,
      Iterable<PbxReferencesProcessor> postProcessors) {
    checkArgument(control.hasPbxproj(), "Must set pbxproj field on control proto.");
    FileSystem fileSystem = workspaceRoot.getFileSystem();

    XcodeprojPath<Path> outputPath = XcodeprojPath.converter().fromPath(
        RelativePaths.fromString(fileSystem, control.getPbxproj()));

    NSDictionary projBuildConfigMap = new NSDictionary();
    projBuildConfigMap.put("ARCHS", new NSArray(
        new NSString("armv7"), new NSString("arm64")));
    projBuildConfigMap.put("CLANG_ENABLE_OBJC_ARC", "YES");
    projBuildConfigMap.put("SDKROOT", "iphoneos");
    projBuildConfigMap.put("IPHONEOS_DEPLOYMENT_TARGET", "7.0");
    projBuildConfigMap.put("GCC_VERSION", "com.apple.compilers.llvm.clang.1_0");
    projBuildConfigMap.put("CODE_SIGN_IDENTITY[sdk=iphoneos*]", "iPhone Developer");

    for (XcodeprojBuildSetting projectSetting : control.getBuildSettingList()) {
      projBuildConfigMap.put(projectSetting.getName(), projectSetting.getValue());
    }

    PBXProject project = new PBXProject(outputPath.getProjectName());
    project.getMainGroup().setPath(workspaceRoot.toString());
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
    LibraryObjects libraryObjects = new LibraryObjects(fileReferences);
    PBXBuildFiles pbxBuildFiles = new PBXBuildFiles(fileReferences);
    Resources resources =
        Resources.fromTargetControls(fileSystem, pbxBuildFiles, control.getTargetList());
    Xcdatamodels xcdatamodels =
        Xcdatamodels.fromTargetControls(fileSystem, pbxBuildFiles, control.getTargetList());
    // We use a hash set for the Project Navigator files so that the same PBXFileReference does not
    // get added twice. Because PBXFileReference uses equality-by-identity semantics, this requires
    // the PBXFileReferences cache to properly return the same reference for functionally-equivalent
    // files.
    Set<PBXReference> projectNavigatorFiles = new LinkedHashSet<>();
    for (TargetControl targetControl : control.getTargetList()) {
      checkArgument(targetControl.hasName(), "TargetControl requires a name: %s", targetControl);
      checkArgument(targetControl.hasLabel(), "TargetControl requires a label: %s", targetControl);

      ProductType productType = productType(targetControl);
      Preconditions.checkArgument(
          (productType != ProductType.APPLICATION) || hasAtLeastOneCompilableSource(targetControl),
          APP_NEEDS_SOURCE_ERROR);
      PBXSourcesBuildPhase sourcesBuildPhase = new PBXSourcesBuildPhase();

      for (SourceFile source : SourceFile.allSourceFiles(fileSystem, targetControl)) {
        PBXFileReference fileRef =
            fileReferences.get(FileReference.of(source.path().toString(), SourceTree.GROUP));
        projectNavigatorFiles.add(fileRef);
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

      PBXFileReference productReference = fileReferences.get(productReference(targetControl));
      projectNavigatorFiles.add(productReference);

      NSDictionary targetBuildConfigMap = new NSDictionary();
      // TODO(bazel-team): Stop adding the workspace root automatically once the
      // released version of Bazel starts passing it.
      targetBuildConfigMap.put("USER_HEADER_SEARCH_PATHS",
          headerSearchPaths(
              plus(targetControl.getUserHeaderSearchPathList(), "$(WORKSPACE_ROOT)")));
      targetBuildConfigMap.put("HEADER_SEARCH_PATHS",
          headerSearchPaths(
              plus(targetControl.getHeaderSearchPathList(), "$(inherited)")));
      targetBuildConfigMap.put("FRAMEWORK_SEARCH_PATHS",
          frameworkSearchPaths(targetControl.getFrameworkList()));

      targetBuildConfigMap.put("WORKSPACE_ROOT", workspaceRoot.toString());

      if (targetControl.hasPchPath()) {
        targetBuildConfigMap.put(
            "GCC_PREFIX_HEADER", "$(WORKSPACE_ROOT)/" + targetControl.getPchPath());
      }

      targetBuildConfigMap.put("PRODUCT_NAME", productName(targetControl));
      if (targetControl.hasInfoplist()) {
        targetBuildConfigMap.put(
            "INFOPLIST_FILE", "$(WORKSPACE_ROOT)/" + targetControl.getInfoplist());
      }

      // Double-quotes in copt strings need to be escaped for XCode.
      if (targetControl.getCoptCount() > 0) {
        List<String> escapedCopts = Lists.transform(
            targetControl.getCoptList(), QUOTE_ESCAPER.asFunction());
        targetBuildConfigMap.put("OTHER_CFLAGS", NSObject.wrap(escapedCopts));
      }
      targetBuildConfigMap.put("OTHER_LDFLAGS", NSObject.wrap(otherLdflags(targetControl)));
      for (XcodeprojBuildSetting setting : targetControl.getBuildSettingList()) {
        String name = setting.getName();
        String value = setting.getValue();
        // TODO(bazel-team): Remove this hack after next Bazel release.
        if (name.equals("CODE_SIGN_ENTITLEMENTS") && !value.startsWith("$")) {
          value = "$(WORKSPACE_ROOT)/" + value;
        }
        targetBuildConfigMap.put(name, value);
      }

      PBXNativeTarget target = new PBXNativeTarget(
          labelToXcodeTargetName(targetControl.getLabel()), productType);
      try {
        target
            .getBuildConfigurationList()
            .getBuildConfigurationsByName()
            .get(DEFAULT_OPTIONS_NAME)
            .setBuildSettings(targetBuildConfigMap);
      } catch (ExecutionException e) {
        throw new RuntimeException(e);
      }
      target.setProductReference(productReference);

      PBXFrameworksBuildPhase frameworksPhase = buildLibraryInfo(libraryObjects, targetControl);
      PBXResourcesBuildPhase resourcesPhase = resources.resourcesBuildPhase(targetControl);

      for (String importedArchive : targetControl.getImportedLibraryList()) {
        PBXFileReference fileReference = fileReferences.get(
            FileReference.of(importedArchive, SourceTree.GROUP)
                .withExplicitFileType(FILE_TYPE_ARCHIVE_LIBRARY));
        projectNavigatorFiles.add(fileReference);
      }

      project.getTargets().add(target);

      target.getBuildPhases().add(frameworksPhase);
      target.getBuildPhases().add(sourcesBuildPhase);
      target.getBuildPhases().add(resourcesPhase);

      checkState(!Mapping.of(targetInfoByLabel, targetControl.getLabel()).isPresent(),
          "Mapping already exists for target with label %s in map: %s",
          targetControl.getLabel(), targetInfoByLabel);
      targetInfoByLabel.put(
          targetControl.getLabel(),
          new TargetInfo(
              targetControl,
              target,
              frameworksPhase,
              resourcesPhase,
              new PBXBuildFile(productReference),
              new LocalPBXTargetDependency(
                  new LocalPBXContainerItemProxy(
                      project, target, ProxyType.TARGET_REFERENCE)),
              targetBuildConfigMap));
    }

    for (HasProjectNavigatorFiles references : ImmutableList.of(pbxBuildFiles, libraryObjects)) {
      Iterables.addAll(projectNavigatorFiles, references.mainGroupReferences());
    }

    Iterable<PBXReference> processedProjectFiles = projectNavigatorFiles;
    for (PbxReferencesProcessor postProcessor : postProcessors) {
      processedProjectFiles = postProcessor.process(processedProjectFiles);
    }

    Iterables.addAll(project.getMainGroup().getChildren(), processedProjectFiles);
    for (TargetInfo targetInfo : targetInfoByLabel.values()) {
      for (DependencyControl dependency : targetInfo.control.getDependencyList()) {
        targetInfo.addDependencyInfo(dependency, targetInfoByLabel);
      }
    }

    return project;
  }
}
