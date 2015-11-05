// Copyright 2014 The Bazel Authors. All rights reserved.
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
import com.facebook.buck.apple.xcode.xcodeproj.PBXShellScriptBuildPhase;
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
import java.util.ArrayList;
import java.util.Arrays;
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
  public static final String FILE_TYPE_FRAMEWORK = "wrapper.frawework";
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
      ProductType.APP_EXTENSION,
      ProductType.FRAMEWORK);

  private static final EnumSet<ProductType> PRODUCT_TYPES_THAT_HAVE_A_BINARY = EnumSet.of(
      ProductType.APPLICATION,
      ProductType.BUNDLE,
      ProductType.UNIT_TEST,
      ProductType.APP_EXTENSION,
      ProductType.FRAMEWORK);

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
      return targetControl.getLabel();
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
      case FRAMEWORK:
        return FileReference.of(
            String.format("%s.framework", productName), SourceTree.BUILT_PRODUCTS_DIR)
                .withExplicitFileType(FILE_TYPE_FRAMEWORK);

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

  /**
   * Returns the {@code ARCHS} array for a target's build config given the list of architecture
   * strings. If none is given, an array with default architectures "armv7" and "arm64" will be
   * returned.
   */
  private static NSArray cpuArchitectures(Iterable<String> architectures) {
    if (Iterables.isEmpty(architectures)) {
      return new NSArray(new NSString("armv7"), new NSString("arm64"));
    } else {
      ImmutableSet.Builder<NSString> result = new ImmutableSet.Builder<>();
      for (String architecture : architectures) {
        result.add(new NSString(architecture));
      }
      return (NSArray) NSObject.wrap(result.build().asList());
    }
  }

  private static PBXFrameworksBuildPhase buildFrameworksInfo(
      LibraryObjects libraryObjects, TargetControl target) {
    BuildPhaseBuilder builder = libraryObjects.newBuildPhase();
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
    if (Containing.item(PRODUCT_TYPES_THAT_HAVE_A_BINARY, productType(targetControl))) {
      for (String dylib : targetControl.getSdkDylibList()) {
        if (dylib.startsWith("lib")) {
          dylib = dylib.substring(3);
        }
        flags.add("-l" + dylib);
      }
    }

    return flags.build();
  }

  /**
   * Returns a unique name for the given imported library path, scoped by both the base name and
   * the parent directories. For example, with "foo/bar/lib.a", "lib_bar_foo.a" will be returned.
   */
  private static String uniqueImportedLibraryName(String importedLibrary) {
    String extension = "";
    String pathWithoutExtension = "";
    int i = importedLibrary.lastIndexOf('.');
    if (i > 0) {
      extension = importedLibrary.substring(i);
      pathWithoutExtension = importedLibrary.substring(0, i);
    } else {
      pathWithoutExtension = importedLibrary;
    }

    String[] pathFragments = pathWithoutExtension.replace("-", "_").split("/");
    return Joiner.on("_").join(Lists.reverse(Arrays.asList(pathFragments))) + extension;
  }

  /** Generates a project file. */
  public static PBXProject xcodeproj(Path workspaceRoot, Control control,
      Iterable<PbxReferencesProcessor> postProcessors) {
    checkArgument(control.hasPbxproj(), "Must set pbxproj field on control proto.");
    FileSystem fileSystem = workspaceRoot.getFileSystem();

    XcodeprojPath<Path> outputPath = XcodeprojPath.converter().fromPath(
        RelativePaths.fromString(fileSystem, control.getPbxproj()));

    NSDictionary projBuildConfigMap = new NSDictionary();
    projBuildConfigMap.put("ARCHS", cpuArchitectures(control.getCpuArchitectureList()));
    projBuildConfigMap.put("VALID_ARCHS",
        new NSArray(
            new NSString("armv7"),
            new NSString("armv7s"),
            new NSString("arm64"),
            new NSString("i386"),
            new NSString("x86_64")));
    projBuildConfigMap.put("CLANG_ENABLE_OBJC_ARC", "YES");
    projBuildConfigMap.put("SDKROOT", "iphoneos");
    projBuildConfigMap.put("IPHONEOS_DEPLOYMENT_TARGET", "7.0");
    projBuildConfigMap.put("GCC_VERSION", "com.apple.compilers.llvm.clang.1_0");
    projBuildConfigMap.put("CODE_SIGN_IDENTITY[sdk=iphoneos*]", "iPhone Developer");

    // Disable bitcode for now.
    // TODO(bazel-team): Need to re-enable once we have real Xcode 7 support.
    projBuildConfigMap.put("ENABLE_BITCODE", "NO");

    for (XcodeprojBuildSetting projectSetting : control.getBuildSettingList()) {
      projBuildConfigMap.put(projectSetting.getName(), projectSetting.getValue());
    }

    PBXProject project = new PBXProject(outputPath.getProjectName());
    project.getMainGroup().setPath(workspaceRoot.toString());
    if (workspaceRoot.isAbsolute()) {
      project.getMainGroup().setSourceTree(SourceTree.ABSOLUTE);
    }
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
    List<String> usedTargetNames = new ArrayList<>();
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

      // Note that HFS+ (the Mac filesystem) is usually case insensitive, so we cast all target
      // names to lower case before checking for duplication because otherwise users may end up
      // having duplicated intermediate build directories that can interfere with the build.
      String targetName = targetControl.getName();
      String targetNameInLowerCase = targetName.toLowerCase();
      if (usedTargetNames.contains(targetNameInLowerCase)) {
        // Use the label in the odd case where we have two targets with the same name.
        targetName = targetControl.getLabel();
        targetNameInLowerCase = targetName.toLowerCase();
      }
      checkState(!usedTargetNames.contains(targetNameInLowerCase),
          "Name (case-insensitive) already exists for target with label/name %s/%s in list: %s",
          targetControl.getLabel(), targetControl.getName(), usedTargetNames);
      usedTargetNames.add(targetNameInLowerCase);
      PBXNativeTarget target = new PBXNativeTarget(targetName, productType);
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

      // We only add frameworks here and not dylibs because of differences in how
      // Xcode 6 and Xcode 7 specify dylibs in the project organizer.
      // (Xcode 6 -> *.dylib, Xcode 7 -> *.tbd)
      PBXFrameworksBuildPhase frameworksPhase = buildFrameworksInfo(libraryObjects, targetControl);
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
      TargetControl targetControl = targetInfo.control;
      for (DependencyControl dependency : targetControl.getDependencyList()) {
        targetInfo.addDependencyInfo(dependency, targetInfoByLabel);
      }

      if (!Equaling.of(ProductType.STATIC_LIBRARY, productType(targetControl))
          && !targetControl.getImportedLibraryList().isEmpty()) {
        // We add a script build phase to copy the imported libraries to BUILT_PRODUCT_DIR with
        // unique names before linking them to work around an Xcode issue where imported libraries
        // with duplicated names lead to link errors.
        //
        // Internally Xcode uses linker flag -l{LIBRARY_NAME} to link a particular library and
        // delegates to the linker to locate the actual library using library search paths. So given
        // two imported libraries with the same name: a/b/libfoo.a, c/d/libfoo.a, Xcode uses
        // duplicate linker flag -lfoo to link both of the libraries. Depending on the order of
        // the library search paths, the linker will only be able to locate and link one of the
        // libraries.
        //
        // With this workaround using a script build phase, all imported libraries to link have
        // unique names. For the previous example with a/b/libfoo.a and c/d/libfoo.a, the script
        // build phase will copy them to BUILT_PRODUCTS_DIR with unique names libfoo_b_a.a and
        // libfoo_d_c.a, respectively. The linker flags Xcode uses to link them will be
        // -lfoo_d_c and -lfoo_b_a, with no duplication.
        PBXShellScriptBuildPhase scriptBuildPhase = new PBXShellScriptBuildPhase();
        scriptBuildPhase.setShellScript(
            "for ((i=0; i < ${SCRIPT_INPUT_FILE_COUNT}; i++)) do\n"
            + "  INPUT_FILE=\"SCRIPT_INPUT_FILE_${i}\"\n"
            + "  OUTPUT_FILE=\"SCRIPT_OUTPUT_FILE_${i}\"\n"
            + "  cp -v -f \"${!INPUT_FILE}\" \"${!OUTPUT_FILE}\"\n"
            + "done");
        for (String importedLibrary : targetControl.getImportedLibraryList()) {
          String uniqueImportedLibrary = uniqueImportedLibraryName(importedLibrary);
          scriptBuildPhase.getInputPaths().add("$(WORKSPACE_ROOT)/" + importedLibrary);
          scriptBuildPhase.getOutputPaths().add("$(BUILT_PRODUCTS_DIR)/" + uniqueImportedLibrary);
          FileReference fileReference = FileReference.of(uniqueImportedLibrary,
              SourceTree.BUILT_PRODUCTS_DIR).withExplicitFileType(FILE_TYPE_ARCHIVE_LIBRARY);
          targetInfo.frameworksPhase.getFiles().add(pbxBuildFiles.getStandalone(fileReference));
        }
        targetInfo.nativeTarget.getBuildPhases().add(scriptBuildPhase);
      }
    }

    return project;
  }
}
