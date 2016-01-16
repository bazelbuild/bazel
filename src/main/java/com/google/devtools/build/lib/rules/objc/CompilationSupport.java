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

import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromTemplates;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.DEFINE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FORCE_LOAD_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FRAMEWORK_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FRAMEWORK_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.Flag.USES_CPP;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.Flag.USES_FRAMEWORKS;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.Flag.USES_SWIFT;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.HEADER;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.INCLUDE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.INCLUDE_SYSTEM;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.MODULE_MAP;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_DYLIB;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.WEAK_SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.CLANG;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.CLANG_PLUSPLUS;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.COMPILABLE_SRCS_TYPE;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.DSYMUTIL;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.HEADERS;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.LIBTOOL;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.NON_ARC_SRCS_TYPE;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.SRCS_TYPE;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.STRIP;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.SWIFT;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.intermediateArtifacts;
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.apple.Platform;
import com.google.devtools.build.lib.rules.cpp.CppModuleMap;
import com.google.devtools.build.lib.rules.cpp.CppModuleMapAction;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs;
import com.google.devtools.build.lib.rules.java.J2ObjcConfiguration;
import com.google.devtools.build.lib.rules.objc.J2ObjcSource.SourceType;
import com.google.devtools.build.lib.rules.objc.ObjcCommon.CompilationAttributes;
import com.google.devtools.build.lib.rules.objc.XcodeProvider.Builder;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesCollector.InstrumentationSpec;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesCollector.LocalMetadataCollector;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Support for rules that compile sources. Provides ways to determine files that should be output,
 * registering Xcode settings and generating the various actions that might be needed for
 * compilation.
 *
 * <p>Methods on this class can be called in any order without impacting the result.
 */
public final class CompilationSupport {

  @VisibleForTesting
  static final String ABSOLUTE_INCLUDES_PATH_FORMAT =
      "The path '%s' is absolute, but only relative paths are allowed.";

  @VisibleForTesting
  static final ImmutableList<String> LINKER_COVERAGE_FLAGS =
      ImmutableList.of("-ftest-coverage", "-fprofile-arcs");

  // Flags for clang 6.1(xcode 6.4)
  @VisibleForTesting
  static final ImmutableList<String> CLANG_COVERAGE_FLAGS =
      ImmutableList.of("-fprofile-arcs", "-ftest-coverage");

  /**
   * Returns the location of the xcrunwrapper tool.
   */
  public static final FilesToRunProvider xcrunwrapper(RuleContext ruleContext) {
   return ruleContext.getExecutablePrerequisite("$xcrunwrapper", Mode.HOST);
  }

  /**
   * Files which can be instrumented along with the attributes in which they may occur and the
   * attributes along which they are propagated from dependencies (via
   * {@link InstrumentedFilesProvider}).
   */
  private static final InstrumentationSpec INSTRUMENTATION_SPEC =
      new InstrumentationSpec(
              FileTypeSet.of(
                  ObjcRuleClasses.NON_CPP_SOURCES,
                  ObjcRuleClasses.CPP_SOURCES,
                  ObjcRuleClasses.SWIFT_SOURCES,
                  HEADERS))
          .withSourceAttributes("srcs", "non_arc_srcs", "hdrs")
          .withDependencyAttributes("deps", "data", "binary", "xctest_app");

  private static final String FRAMEWORK_SUFFIX = ".framework";

  private static final Predicate<String> INCLUDE_DIR_OPTION_IN_COPTS =
      new Predicate<String>() {
        @Override
        public boolean apply(String copt) {
          return copt.startsWith("-I") && copt.length() > 2;
        }
      };

  /**
   * Predicate to remove '.inc' files from an iterable.
   */
  private static final Predicate<Artifact> NON_INC_FILES =
      new Predicate<Artifact>() {
        @Override
        public boolean apply(Artifact artifact) {
          return !artifact.getFilename().endsWith(".inc");
        }
      };

  /**
   * Defines a library that contains the transitive closure of dependencies.
   */
  public static final SafeImplicitOutputsFunction FULLY_LINKED_LIB =
      fromTemplates("%{name}_fully_linked.a");

  /**
   * Iterable wrapper providing strong type safety for arguments to binary linking.
   */
  static final class ExtraLinkArgs extends IterableWrapper<String> {
    ExtraLinkArgs(String... args) {
      super(args);
    }
  }

  @VisibleForTesting
  static final String FILE_IN_SRCS_AND_HDRS_WARNING_FORMAT =
      "File '%s' is in both srcs and hdrs.";

  @VisibleForTesting
  static final String FILE_IN_SRCS_AND_NON_ARC_SRCS_ERROR_FORMAT =
      "File '%s' is present in both srcs and non_arc_srcs which is forbidden.";

  static final ImmutableList<String> DEFAULT_COMPILER_FLAGS = ImmutableList.of("-DOS_IOS");

  static final ImmutableList<String> DEFAULT_LINKER_FLAGS = ImmutableList.of("-ObjC");
  
  /**
   * Returns information about the given rule's compilation artifacts.
   */
  // TODO(bazel-team): Remove this information from ObjcCommon and move it internal to this class.
  static CompilationArtifacts compilationArtifacts(RuleContext ruleContext) {
    PrerequisiteArtifacts srcs = ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET)
        .errorsForNonMatching(SRCS_TYPE);
    return new CompilationArtifacts.Builder()
        .addSrcs(srcs.filter(COMPILABLE_SRCS_TYPE).list())
        .addNonArcSrcs(ruleContext.getPrerequisiteArtifacts("non_arc_srcs", Mode.TARGET)
            .errorsForNonMatching(NON_ARC_SRCS_TYPE)
            .list())
        .addPrivateHdrs(srcs.filter(HEADERS).list())
        .setIntermediateArtifacts(ObjcRuleClasses.intermediateArtifacts(ruleContext))
        .setPchFile(Optional.fromNullable(ruleContext.getPrerequisiteArtifact("pch", Mode.TARGET)))
        .build();
  }

  private final RuleContext ruleContext;
  private final CompilationAttributes attributes;

  /**
   * Creates a new compilation support for the given rule.
   */
  public CompilationSupport(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
    this.attributes = new CompilationAttributes(ruleContext);
  }

  /**
   * Registers all actions necessary to compile this rule's sources and archive them.
   *
   * @param common common information about this rule and its dependencies
   * @return this compilation support
   */
  CompilationSupport registerCompileAndArchiveActions(ObjcCommon common)
      throws InterruptedException {
    if (common.getCompilationArtifacts().isPresent()) {
      registerGenerateModuleMapAction(common.getCompilationArtifacts());
      IntermediateArtifacts intermediateArtifacts =
          ObjcRuleClasses.intermediateArtifacts(ruleContext);
      Optional<CppModuleMap> moduleMap;
      if (ObjcRuleClasses.objcConfiguration(ruleContext).moduleMapsEnabled()) {
        moduleMap = Optional.of(intermediateArtifacts.moduleMap());
      } else {
        moduleMap = Optional.absent();
      }
      registerCompileAndArchiveActions(
          common.getCompilationArtifacts().get(),
          intermediateArtifacts,
          common.getObjcProvider(),
          moduleMap,
          ruleContext.getConfiguration().isCodeCoverageEnabled(),
          true);
    }
    return this;
  }

  /**
   * Creates actions to compile each source file individually, and link all the compiled object
   * files into a single archive library.
   */
  private void registerCompileAndArchiveActions(
      CompilationArtifacts compilationArtifacts,
      IntermediateArtifacts intermediateArtifacts,
      ObjcProvider objcProvider,
      Optional<CppModuleMap> moduleMap,
      boolean isCodeCoverageEnabled,
      boolean isFullyLinkEnabled) throws InterruptedException {
    ImmutableList.Builder<Artifact> objFiles = new ImmutableList.Builder<>();
    for (Artifact sourceFile : compilationArtifacts.getSrcs()) {
      Artifact objFile = intermediateArtifacts.objFile(sourceFile);
      objFiles.add(objFile);
      if (ObjcRuleClasses.SWIFT_SOURCES.matches(sourceFile.getFilename())) {
        registerSwiftCompileAction(sourceFile, objFile, intermediateArtifacts, objcProvider);
      } else {
        registerCompileAction(
            sourceFile,
            objFile,
            objcProvider,
            moduleMap,
            intermediateArtifacts,
            compilationArtifacts,
            ImmutableList.of("-fobjc-arc"),
            isCodeCoverageEnabled);
      }
    }
    for (Artifact nonArcSourceFile : compilationArtifacts.getNonArcSrcs()) {
      Artifact objFile = intermediateArtifacts.objFile(nonArcSourceFile);
      objFiles.add(objFile);
      registerCompileAction(
          nonArcSourceFile,
          objFile,
          objcProvider,
          moduleMap,
          intermediateArtifacts,
          compilationArtifacts,
          ImmutableList.of("-fno-objc-arc"),
          isCodeCoverageEnabled);
    }

    if (compilationArtifacts.hasSwiftSources()) {
      registerSwiftModuleMergeAction(intermediateArtifacts, compilationArtifacts, objcProvider);
    }

    for (Artifact archive : compilationArtifacts.getArchive().asSet()) {
      registerArchiveActions(intermediateArtifacts, objFiles, archive);
    }

    if (isFullyLinkEnabled) {
      registerFullyLinkAction(objcProvider);
    }
  }

  /**
   * Adds a source file to a command line, honoring the useAbsolutePathForActions flag.
   */
  private CustomCommandLine.Builder addSource(CustomCommandLine.Builder commandLine,
      ObjcConfiguration objcConfiguration, Artifact sourceFile) {
    PathFragment sourceExecPathFragment = sourceFile.getExecPath();
    String sourcePath = sourceExecPathFragment.getPathString();
    if (!sourceExecPathFragment.isAbsolute() && objcConfiguration.getUseAbsolutePathsForActions()) {
      sourcePath = objcConfiguration.getXcodeWorkspaceRoot() + "/" + sourcePath;
    }
    commandLine.add(sourcePath);
    return commandLine;
  }

  private CustomCommandLine.Builder addSource(String argName, CustomCommandLine.Builder commandLine,
      ObjcConfiguration objcConfiguration, Artifact sourceFile) {
    commandLine.add(argName);
    return addSource(commandLine, objcConfiguration, sourceFile);
  }

  private void registerCompileAction(
      Artifact sourceFile,
      Artifact objFile,
      ObjcProvider objcProvider,
      Optional<CppModuleMap> moduleMap,
      IntermediateArtifacts intermediateArtifacts,
      CompilationArtifacts compilationArtifacts,
      Iterable<String> otherFlags,
      boolean isCodeCoverageEnabled) {
    ObjcConfiguration objcConfiguration = ObjcRuleClasses.objcConfiguration(ruleContext);
    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);
    ImmutableList.Builder<String> coverageFlags = new ImmutableList.Builder<>();
    ImmutableList.Builder<Artifact> gcnoFiles = new ImmutableList.Builder<>();
    ImmutableList.Builder<Artifact> additionalInputs = new ImmutableList.Builder<>();
    if (isCodeCoverageEnabled && ObjcRuleClasses.isInstrumentable(sourceFile)) {
      coverageFlags.addAll(CLANG_COVERAGE_FLAGS);
      gcnoFiles.add(intermediateArtifacts.gcnoFile(sourceFile));
    }
    CustomCommandLine.Builder commandLine = new CustomCommandLine.Builder()
        .add(CLANG);
    if (ObjcRuleClasses.CPP_SOURCES.matches(sourceFile.getExecPath())) {
      commandLine.add("-stdlib=libc++");
    }

    if (compilationArtifacts.hasSwiftSources()) {
      // Add the directory that contains merged TargetName-Swift.h header to search path, in case
      // any of ObjC files use it.
      commandLine.add("-I");
      commandLine.addPath(intermediateArtifacts.swiftHeader().getExecPath().getParentDirectory());
      additionalInputs.add(intermediateArtifacts.swiftHeader());
    }

    // The linker needs full debug symbol information to perform binary dead-code stripping.
    if (objcConfiguration.shouldStripBinary()) {
      commandLine.add("-g");
    }

    Artifact dotdFile = intermediateArtifacts.dotdFile(sourceFile);
    commandLine
        .add(compileFlagsForClang(appleConfiguration))
        .add(commonLinkAndCompileFlagsForClang(objcProvider, objcConfiguration, appleConfiguration))
        .add(objcConfiguration.getCoptsForCompilationMode())
        .addBeforeEachPath(
            "-iquote", ObjcCommon.userHeaderSearchPaths(ruleContext.getConfiguration()))
        .addBeforeEachExecPath("-include", compilationArtifacts.getPchFile().asSet())
        .addBeforeEachPath("-I", objcProvider.get(INCLUDE))
        .addBeforeEachPath("-isystem", objcProvider.get(INCLUDE_SYSTEM))
        .add(otherFlags)
        .addFormatEach("-D%s", objcProvider.get(DEFINE))
        .add(coverageFlags.build())
        .add(objcConfiguration.getCopts())
        .add(attributes.copts())
        .add(attributes.optionsCopts());
    PathFragment sourceExecPathFragment = sourceFile.getExecPath();
    String sourcePath = sourceExecPathFragment.getPathString();
    if (!sourceExecPathFragment.isAbsolute() && objcConfiguration.getUseAbsolutePathsForActions()) {
      sourcePath = objcConfiguration.getXcodeWorkspaceRoot() + "/" + sourcePath;
    }
    commandLine
      .add("-c").add(sourcePath)
      .addExecPath("-o", objFile)
      .add("-MD")
      .addExecPath("-MF", dotdFile);

    if (moduleMap.isPresent()) {
      // -fmodule-map-file only loads the module in Xcode 7, so we add the module maps's directory
      // to the include path instead.
      // TODO(bazel-team): Use -fmodule-map-file when Xcode 6 support is dropped.
      commandLine
          .add(attributes.enableModules() ? "-fmodules" : "-fmodule-maps")
          .add("-iquote")
          .add(
              moduleMap
                  .get()
                  .getArtifact()
                  .getExecPath()
                  .getParentDirectory()
                  .toString())
          .add("-fmodule-name=" + moduleMap.get().getName());
      if (attributes.enableModules()) {
        String cachePath =
            ruleContext.getConfiguration().getGenfilesFragment() + "/_objc_module_cache";
        commandLine.add("-fmodules-cache-path=" + cachePath);
      }
    }

    // TODO(bazel-team): Remote private headers from inputs once they're added to the provider.
    ruleContext.registerAction(ObjcRuleClasses.spawnXcrunActionBuilder(ruleContext)
        .setMnemonic("ObjcCompile")
        .setExecutable(xcrunwrapper(ruleContext))
        .setCommandLine(commandLine.build())
        .addInput(sourceFile)
        .addInputs(additionalInputs.build())
        .addOutput(objFile)
        .addOutputs(gcnoFiles.build())
        .addOutput(dotdFile)
        .addTransitiveInputs(objcProvider.get(HEADER))
        .addTransitiveInputs(objcProvider.get(MODULE_MAP))
        .addInputs(compilationArtifacts.getPrivateHdrs())
        .addTransitiveInputs(objcProvider.get(FRAMEWORK_FILE))
        .addInputs(compilationArtifacts.getPchFile().asSet())
        .build(ruleContext));
  }

  /**
   * Compiles a single swift file.
   *
   * @param sourceFile the artifact to compile
   * @param objFile the resulting object artifact
   * @param intermediateArtifacts intermediary artifacts
   * @param objcProvider ObjcProvider instance for this invocation
   */
  private void registerSwiftCompileAction(
      Artifact sourceFile,
      Artifact objFile,
      IntermediateArtifacts intermediateArtifacts,
      ObjcProvider objcProvider) {
    ObjcConfiguration objcConfiguration = ObjcRuleClasses.objcConfiguration(ruleContext);
    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);

    // Compiling a single swift file requires knowledge of all of the other
    // swift files in the same module. The primary file ({@code sourceFile}) is
    // compiled to an object file, while the remaining files are used to resolve
    // symbols (they behave like c/c++ headers in this context).
    ImmutableSet.Builder<Artifact> otherSwiftSourcesBuilder = ImmutableSet.builder();
    for (Artifact otherSourceFile : compilationArtifacts(ruleContext).getSrcs()) {
      if (ObjcRuleClasses.SWIFT_SOURCES.matches(otherSourceFile.getFilename())
          && otherSourceFile != sourceFile) {
        otherSwiftSourcesBuilder.add(otherSourceFile);
      }
    }
    ImmutableSet<Artifact> otherSwiftSources = otherSwiftSourcesBuilder.build();

    CustomCommandLine.Builder commandLine = new CustomCommandLine.Builder()
        .add(SWIFT)
        .add("-frontend")
        .add("-emit-object")
        .add("-target").add(swiftTarget(appleConfiguration))
        .add("-sdk").add(AppleToolchain.sdkDir())
        .add("-enable-objc-interop");

    if (objcConfiguration.generateDebugSymbols()) {
      commandLine.add("-g");
    }

    commandLine
      .add("-Onone")
      .add("-module-name").add(getModuleName())
      .add("-parse-as-library");
    addSource("-primary-file", commandLine, objcConfiguration, sourceFile)
      .addExecPaths(otherSwiftSources)
      .addExecPath("-o", objFile)
      .addExecPath("-emit-module-path", intermediateArtifacts.swiftModuleFile(sourceFile))
      // The swift compiler will invoke clang itself when compiling module maps. This invocation
      // does not include the current working directory, causing cwd-relative imports to fail.
      // Including the current working directory to the header search paths ensures that these
      // relative imports will work.
      .add("-Xcc").add("-I.");

    // Using addExecPathBefore here adds unnecessary quotes around '-Xcc -I', which trips the
    // compiler. Using two add() calls generates a correctly formed command line.
    for (PathFragment directory : objcProvider.get(INCLUDE).toList()) {
      commandLine.add("-Xcc").add(String.format("-I%s", directory.toString()));
    }

    ImmutableList.Builder<Artifact> inputHeaders = ImmutableList.<Artifact>builder()
        .addAll(attributes.hdrs())
        .addAll(attributes.textualHdrs());

    Optional<Artifact> bridgingHeader = attributes.bridgingHeader();
    if (bridgingHeader.isPresent()) {
      commandLine.addExecPath("-import-objc-header", bridgingHeader.get());
      inputHeaders.add(bridgingHeader.get());
    }

    // Import the Objective-C module map.
    // TODO(bazel-team): Find a way to import the module map directly, instead of the parent
    // directory?
    if (objcConfiguration.moduleMapsEnabled()) {
      PathFragment moduleMapPath = intermediateArtifacts.moduleMap().getArtifact().getExecPath();
      commandLine.add("-I").add(moduleMapPath.getParentDirectory().toString());
      commandLine.add("-import-underlying-module");
    }

    ruleContext.registerAction(
        ObjcRuleClasses.spawnXcrunActionBuilder(ruleContext)
            .setMnemonic("SwiftCompile")
            .setExecutable(xcrunwrapper(ruleContext))
            .setCommandLine(commandLine.build())
            .addInput(sourceFile)
            .addInputs(otherSwiftSources)
            .addInputs(inputHeaders.build())
            .addTransitiveInputs(objcProvider.get(HEADER))
            .addTransitiveInputs(objcProvider.get(MODULE_MAP))
            .addOutput(objFile)
            .addOutput(intermediateArtifacts.swiftModuleFile(sourceFile))
            .build(ruleContext));
  }

  /**
   * Merges multiple .partial_swiftmodule files together. Also produces a swift header that can be
   * used by Objective-C code.
   */
  private void registerSwiftModuleMergeAction(
      IntermediateArtifacts intermediateArtifacts,
      CompilationArtifacts compilationArtifacts,
      ObjcProvider objcProvider) {
    ObjcConfiguration objcConfiguration = ObjcRuleClasses.objcConfiguration(ruleContext);
    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);

    ImmutableList.Builder<Artifact> moduleFiles = new ImmutableList.Builder<>();
    for (Artifact src : compilationArtifacts.getSrcs()) {
      if (ObjcRuleClasses.SWIFT_SOURCES.matches(src.getFilename())) {
        moduleFiles.add(intermediateArtifacts.swiftModuleFile(src));
      }
    }

    CustomCommandLine.Builder commandLine = new CustomCommandLine.Builder()
        .add(SWIFT)
        .add("-frontend")
        .add("-emit-module")
        .add("-sdk").add(AppleToolchain.sdkDir())
        .add("-target").add(swiftTarget(appleConfiguration));

    if (objcConfiguration.generateDebugSymbols()) {
      commandLine.add("-g");
    }

    commandLine
        .add("-module-name").add(getModuleName())
        .add("-parse-as-library")
        .addExecPaths(moduleFiles.build())
        .addExecPath("-o", intermediateArtifacts.swiftModule())
        .addExecPath("-emit-objc-header-path", intermediateArtifacts.swiftHeader())
        // The swift compiler will invoke clang itself when compiling module maps. This invocation
        // does not include the current working directory, causing cwd-relative imports to fail.
        // Including the current working directory to the header search paths ensures that these
        // relative imports will work.
        .add("-Xcc").add("-I.");


    // Using addExecPathBefore here adds unnecessary quotes around '-Xcc -I', which trips the
    // compiler. Using two add() calls generates a correctly formed command line.
    for (PathFragment directory : objcProvider.get(INCLUDE).toList()) {
      commandLine.add("-Xcc").add(String.format("-I%s", directory.toString()));
    }

    // Import the Objective-C module map.
    // TODO(bazel-team): Find a way to import the module map directly, instead of the parent
    // directory?
    if (objcConfiguration.moduleMapsEnabled()) {
      PathFragment moduleMapPath = intermediateArtifacts.moduleMap().getArtifact().getExecPath();
      commandLine.add("-I").add(moduleMapPath.getParentDirectory().toString());
    }

    ruleContext.registerAction(ObjcRuleClasses.spawnXcrunActionBuilder(ruleContext)
        .setMnemonic("SwiftModuleMerge")
        .setExecutable(xcrunwrapper(ruleContext))
        .setCommandLine(commandLine.build())
        .addInputs(moduleFiles.build())
        .addTransitiveInputs(objcProvider.get(HEADER))
        .addTransitiveInputs(objcProvider.get(MODULE_MAP))
        .addOutput(intermediateArtifacts.swiftModule())
        .addOutput(intermediateArtifacts.swiftHeader())
        .build(ruleContext));
  }

  private void registerArchiveActions(IntermediateArtifacts intermediateArtifacts,
      ImmutableList.Builder<Artifact> objFiles, Artifact archive) {
    for (Action action : archiveActions(ruleContext, objFiles.build(), archive,
        ruleContext.getFragment(AppleConfiguration.class),
        intermediateArtifacts.objList())) {
      ruleContext.registerAction(action);
    }
  }

  private Iterable<Action> archiveActions(
      ActionConstructionContext context,
      Iterable<Artifact> objFiles,
      Artifact archive,
      AppleConfiguration appleConfiguration,
      Artifact objList) {

    ImmutableList.Builder<Action> actions = new ImmutableList.Builder<>();

    actions.add(new FileWriteAction(
        context.getActionOwner(),
        objList,
        Artifact.joinExecPaths("\n", objFiles),
        /*makeExecutable=*/ false));

    actions.add(ObjcRuleClasses.spawnXcrunActionBuilder(ruleContext)
        .setMnemonic("ObjcLink")
        .setExecutable(xcrunwrapper(ruleContext))
        .setCommandLine(new CustomCommandLine.Builder()
            .add(LIBTOOL)
            .add("-static")
            .add("-filelist").add(objList.getExecPathString())
            .add("-arch_only").add(appleConfiguration.getIosCpu())
            .add("-syslibroot").add(AppleToolchain.sdkDir())
            .add("-o").add(archive.getExecPathString())
            .build())
        .addInputs(objFiles)
        .addInput(objList)
        .addOutput(archive)
        .build(context));

    return actions.build();
  }

  private void registerFullyLinkAction(ObjcProvider objcProvider) throws InterruptedException {
    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);
    Artifact archive = ruleContext.getImplicitOutputArtifact(FULLY_LINKED_LIB);

    ImmutableList<Artifact> ccLibraries = ccLibraries(objcProvider);
    ruleContext.registerAction(ObjcRuleClasses.spawnXcrunActionBuilder(ruleContext)
        .setMnemonic("ObjcLink")
        .setExecutable(xcrunwrapper(ruleContext))
        .setCommandLine(new CustomCommandLine.Builder()
            .add(LIBTOOL)
            .add("-static")
            .add("-arch_only").add(appleConfiguration.getIosCpu())
            .add("-syslibroot").add(AppleToolchain.sdkDir())
            .add("-o").add(archive.getExecPathString())
            .addExecPaths(objcProvider.get(LIBRARY))
            .addExecPaths(objcProvider.get(IMPORTED_LIBRARY))
            .addExecPaths(ccLibraries)
            .build())
        .addInputs(ccLibraries)
        .addTransitiveInputs(objcProvider.get(LIBRARY))
        .addTransitiveInputs(objcProvider.get(IMPORTED_LIBRARY))
        .addOutput(archive)
        .build(ruleContext));
  }

  /**
   * Returns a provider that collects this target's instrumented sources as well as those of its
   * dependencies.
   *
   * @param common common information about this rule and its dependencies
   * @return an instrumented files provider
   */
  public InstrumentedFilesProvider getInstrumentedFilesProvider(ObjcCommon common) {
    ImmutableList.Builder<Artifact> oFiles = ImmutableList.builder();

    if (common.getCompilationArtifacts().isPresent()) {
      CompilationArtifacts artifacts = common.getCompilationArtifacts().get();
      IntermediateArtifacts intermediateArtifacts = intermediateArtifacts(ruleContext);
      for (Artifact artifact : Iterables.concat(artifacts.getSrcs(), artifacts.getNonArcSrcs())) {
        oFiles.add(intermediateArtifacts.objFile(artifact));
      }
    }

    return InstrumentedFilesCollector.collect(
        ruleContext,
        INSTRUMENTATION_SPEC,
        new ObjcCoverageMetadataCollector(),
        oFiles.build(),
        !TargetUtils.isTestRule(ruleContext.getTarget()));
  }

  /**
   * Registers any actions necessary to link this rule and its dependencies.
   *
   * <p>Dsym bundle and breakpad files are generated if
   * {@link ObjcConfiguration#generateDebugSymbols()} is set.
   *
   * <p>When Bazel flags {@code --compilation_mode=opt} and {@code --objc_enable_binary_stripping}
   * are specified, additional optimizations will be performed on the linked binary: all-symbol
   * stripping (using {@code /usr/bin/strip}) and dead-code stripping (using linker flags:
   * {@code -dead_strip} and {@code -no_dead_strip_inits_and_terms}).
   *
   * @param objcProvider common information about this rule's attributes and its dependencies
   * @param extraLinkArgs any additional arguments to pass to the linker
   * @param extraLinkInputs any additional input artifacts to pass to the link action
   *
   * @return this compilation support
   */
  CompilationSupport registerLinkActions(ObjcProvider objcProvider, ExtraLinkArgs extraLinkArgs,
      Iterable<Artifact> extraLinkInputs) {
    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);
    Optional<Artifact> dsymBundle;
    if (ObjcRuleClasses.objcConfiguration(ruleContext).generateDebugSymbols()) {
      registerDsymActions();
      dsymBundle = Optional.of(intermediateArtifacts.dsymBundle());
    } else {
      dsymBundle = Optional.absent();
    }

    registerLinkAction(objcProvider, extraLinkArgs, extraLinkInputs, dsymBundle);
    return this;
  }

  /**
   * Registers an action that will generate a clang module map for this target, using the hdrs
   * attribute of this rule.
   */
  CompilationSupport registerGenerateModuleMapAction(
      Optional<CompilationArtifacts> compilationArtifacts) {
    if (ObjcRuleClasses.objcConfiguration(ruleContext).moduleMapsEnabled()) {
      // TODO(bazel-team): Include textual headers in the module map when Xcode 6 support is
      // dropped.
      Iterable<Artifact> publicHeaders = attributes.hdrs();
      Iterable<Artifact> privateHeaders = ImmutableList.of();
      if (compilationArtifacts.isPresent()) {
        CompilationArtifacts artifacts = compilationArtifacts.get();
        publicHeaders = Iterables.concat(publicHeaders, artifacts.getAdditionalHdrs());
        privateHeaders = Iterables.concat(privateHeaders, artifacts.getPrivateHdrs());
      }
      CppModuleMap moduleMap = ObjcRuleClasses.intermediateArtifacts(ruleContext).moduleMap();
      registerGenerateModuleMapAction(moduleMap, publicHeaders, privateHeaders);
    }
    return this;
  }

  /**
   * Registers an action that will generate a clang module map.
   *
   * @param moduleMap the module map to generate
   * @param publicHeaders the headers that should be directly accessible by dependers
   * @param privateHeaders the headers that should only be directly accessible by this module
   */
  private void registerGenerateModuleMapAction(
      CppModuleMap moduleMap, Iterable<Artifact> publicHeaders, Iterable<Artifact> privateHeaders) {
    // The current clang (clang-600.0.57) on Darwin doesn't support 'textual', so we can't have
    // '.inc' files in the module map (since they're implictly textual).
    // TODO(bazel-team): Remove filtering once clang-700 is the base clang we support.
    publicHeaders = Iterables.filter(publicHeaders, NON_INC_FILES);
    privateHeaders = Iterables.filter(privateHeaders, NON_INC_FILES);
    ruleContext.registerAction(
        new CppModuleMapAction(
            ruleContext.getActionOwner(),
            moduleMap,
            privateHeaders,
            publicHeaders,
            attributes.moduleMapsForDirectDeps(),
            ImmutableList.<PathFragment>of(),
            /*compiledModule=*/ true,
            /*moduleMapHomeIsCwd=*/ false,
            /*generateSubModules=*/ false,
            /*externDependencies=*/ true));
  }

  private void registerLinkAction(ObjcProvider objcProvider, ExtraLinkArgs extraLinkArgs,
      Iterable<Artifact> extraLinkInputs, Optional<Artifact> dsymBundle) {
    ObjcConfiguration objcConfiguration = ObjcRuleClasses.objcConfiguration(ruleContext);
    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);

    // When compilation_mode=opt and objc_enable_binary_stripping are specified, the unstripped
    // binary containing debug symbols is generated by the linker, which also needs the debug
    // symbols for dead-code removal. The binary is also used to generate dSYM bundle if
    // --objc_generate_debug_symbol is specified. A symbol strip action is later registered to strip
    // the symbol table from the unstripped binary.
    Artifact binaryToLink =
        objcConfiguration.shouldStripBinary()
            ? intermediateArtifacts.unstrippedSingleArchitectureBinary()
            : intermediateArtifacts.strippedSingleArchitectureBinary();

    ImmutableList<Artifact> ccLibraries = ccLibraries(objcProvider);
    ruleContext.registerAction(
        ObjcRuleClasses.spawnXcrunActionBuilder(ruleContext)
            .setMnemonic("ObjcLink")
            .setShellCommand(ImmutableList.of("/bin/bash", "-c"))
            .setCommandLine(
                linkCommandLine(extraLinkArgs, objcProvider, binaryToLink, dsymBundle, ccLibraries))
            .addOutput(binaryToLink)
            .addOutputs(dsymBundle.asSet())
            .addTransitiveInputs(objcProvider.get(LIBRARY))
            .addTransitiveInputs(objcProvider.get(IMPORTED_LIBRARY))
            .addTransitiveInputs(objcProvider.get(FRAMEWORK_FILE))
            .addInputs(ccLibraries)
            .addInputs(extraLinkInputs)
            .addInput(xcrunwrapper(ruleContext).getExecutable())
            .build(ruleContext));

    if (objcConfiguration.shouldStripBinary()) {
      // For test targets, only debug symbols are stripped off, since /usr/bin/strip is not able
      // to strip off all symbols in XCTest bundle.
      boolean isTestTarget = TargetUtils.isTestRule(ruleContext.getRule());
      Iterable<String> stripArgs =
          isTestTarget ? ImmutableList.of("-S") : ImmutableList.<String>of();
      Artifact strippedBinary = intermediateArtifacts.strippedSingleArchitectureBinary();

      ruleContext.registerAction(
          ObjcRuleClasses.spawnXcrunActionBuilder(ruleContext)
              .setMnemonic("ObjcBinarySymbolStrip")
              .setExecutable(xcrunwrapper(ruleContext))
              .setCommandLine(symbolStripCommandLine(stripArgs, binaryToLink, strippedBinary))
              .addOutput(strippedBinary)
              .addInput(binaryToLink)
              .build(ruleContext));
    }
  }

  private ImmutableList<Artifact> ccLibraries(ObjcProvider objcProvider) {
    ImmutableList.Builder<Artifact> ccLibraryBuilder = ImmutableList.builder();
    for (LinkerInputs.LibraryToLink libraryToLink : objcProvider.get(ObjcProvider.CC_LIBRARY)) {
      ccLibraryBuilder.add(libraryToLink.getArtifact());
    }
    return ccLibraryBuilder.build();
  }

  private static CommandLine symbolStripCommandLine(
      Iterable<String> extraFlags, Artifact unstrippedArtifact, Artifact strippedArtifact) {
    return CustomCommandLine.builder()
        .add(STRIP)
        .add(extraFlags)
        .addExecPath("-o", strippedArtifact)
        .addPath(unstrippedArtifact.getExecPath())
        .build();
  }

  private CommandLine linkCommandLine(ExtraLinkArgs extraLinkArgs,
      ObjcProvider objcProvider, Artifact linkedBinary, Optional<Artifact> dsymBundle,
      ImmutableList<Artifact> ccLibraries) {
    ObjcConfiguration objcConfiguration = ObjcRuleClasses.objcConfiguration(ruleContext);
    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);

    CustomCommandLine.Builder commandLine = CustomCommandLine.builder()
        .addPath(xcrunwrapper(ruleContext).getExecutable().getExecPath());

    if (objcProvider.is(USES_CPP)) {
      commandLine
          .add(CLANG_PLUSPLUS)
          .add("-stdlib=libc++");
    } else {
      commandLine.add(CLANG);
    }

    // Do not perform code stripping on tests because XCTest binary is linked not as an executable
    // but as a bundle without any entry point.
    boolean isTestTarget = TargetUtils.isTestRule(ruleContext.getRule());
    if (objcConfiguration.shouldStripBinary() && !isTestTarget) {
      commandLine.add("-dead_strip").add("-no_dead_strip_inits_and_terms");
    }

    commandLine
        .add(commonLinkAndCompileFlagsForClang(objcProvider, objcConfiguration, appleConfiguration))
        .add("-Xlinker")
        .add("-objc_abi_version")
        .add("-Xlinker")
        .add("2")
        .add("-fobjc-link-runtime")
        .add(DEFAULT_LINKER_FLAGS)
        .addBeforeEach("-framework", frameworkNames(objcProvider))
        .addBeforeEach("-weak_framework", SdkFramework.names(objcProvider.get(WEAK_SDK_FRAMEWORK)))
        .addFormatEach("-l%s", libraryNames(objcProvider))
        .addExecPath("-o", linkedBinary)
        .addExecPaths(objcProvider.get(LIBRARY))
        .addExecPaths(objcProvider.get(IMPORTED_LIBRARY))
        .addExecPaths(ccLibraries)
        .addBeforeEach("-force_load", Artifact.toExecPaths(objcProvider.get(FORCE_LOAD_LIBRARY)))
        .add(extraLinkArgs)
        .add(objcProvider.get(ObjcProvider.LINKOPT))
        .build();

    if (ruleContext.getConfiguration().isCodeCoverageEnabled()) {
      commandLine.add(LINKER_COVERAGE_FLAGS);
    }

    if (objcProvider.is(USES_SWIFT)) {
      commandLine.add("-L").add(AppleToolchain.swiftLibDir(appleConfiguration));
    }

    if (objcProvider.is(USES_SWIFT) || objcProvider.is(USES_FRAMEWORKS)) {
      // Enable loading bundled frameworks.
      commandLine
          .add("-Xlinker").add("-rpath")
          .add("-Xlinker").add("@executable_path/Frameworks");
    }

    // Call to dsymutil for debug symbol generation must happen in the link action.
    // All debug symbol information is encoded in object files inside archive files. To generate
    // the debug symbol bundle, dsymutil will look inside the linked binary for the encoded
    // absolute paths to archive files, which are only valid in the link action.
    if (dsymBundle.isPresent()) {
      PathFragment dsymPath = FileSystemUtils.removeExtension(dsymBundle.get().getExecPath());
      commandLine
          .add("&&")
          .addPath(xcrunwrapper(ruleContext).getExecutable().getExecPath())
          .add(DSYMUTIL)
          .add(linkedBinary.getExecPathString())
          .add("-o " + dsymPath)
          .add("&& zipped_bundle=${PWD}/" + dsymBundle.get().getShellEscapedExecPathString())
          .add("&& cd " + dsymPath)
          .add("&& /usr/bin/zip -q -r \"${zipped_bundle}\" .");
    }

    return new SingleArgCommandLine(commandLine.build());
  }

  /**
   * Command line that converts its input's arg array to a single input.
   *
   * <p>Required as a hack to the link command line because that may contain two commands, which are
   * then passed to {@code /bin/bash -c}, and accordingly need to be a single argument.
   */
  private static class SingleArgCommandLine extends CommandLine {

    private final CommandLine original;

    private SingleArgCommandLine(CommandLine original) {
      this.original = original;
    }

    @Override
    public Iterable<String> arguments() {
      return ImmutableList.of(Joiner.on(' ').join(original.arguments()));
    }
  }

  private Iterable<String> libraryNames(ObjcProvider objcProvider) {
    ImmutableList.Builder<String> args = new ImmutableList.Builder<>();
    for (String dylib : objcProvider.get(SDK_DYLIB)) {
      if (dylib.startsWith("lib")) {
        // remove lib prefix if it exists which is standard
        // for libraries (libxml.dylib -> -lxml).
        dylib = dylib.substring(3);
      }
      args.add(dylib);
    }
    return args.build();
  }

  /**
   * All framework names to pass to the linker using {@code -framework} flags. For a framework in
   * the directory foo/bar.framework, the name is "bar". Each framework is found without using the
   * full path by means of the framework search paths. The search paths are added by
   * {@link #commonLinkAndCompileFlagsForClang(ObjcProvider, ObjcConfiguration,
   * AppleConfiguration)}).
   *
   * <p>It's awful that we can't pass the full path to the framework and avoid framework search
   * paths, but this is imposed on us by clang. clang does not support passing the full path to the
   * framework, so Bazel cannot do it either.
   */
  private Iterable<String> frameworkNames(ObjcProvider provider) {
    List<String> names = new ArrayList<>();
    Iterables.addAll(names, SdkFramework.names(provider.get(SDK_FRAMEWORK)));
    for (PathFragment frameworkDir : provider.get(FRAMEWORK_DIR)) {
      String segment = frameworkDir.getBaseName();
      Preconditions.checkState(segment.endsWith(FRAMEWORK_SUFFIX),
          "expect %s to end with %s, but it does not", segment, FRAMEWORK_SUFFIX);
      names.add(segment.substring(0, segment.length() - FRAMEWORK_SUFFIX.length()));
    }
    return names;
  }

  /**
   * Registers actions that compile and archive j2Objc dependencies of this rule.
   *
   * @param objcProvider common information about this rule's attributes and its dependencies
   *
   * @return this compilation support
   */
  CompilationSupport registerJ2ObjcCompileAndArchiveActions(ObjcProvider objcProvider)
      throws InterruptedException {
    J2ObjcSrcsProvider provider = J2ObjcSrcsProvider.buildFrom(ruleContext);
    Iterable<J2ObjcSource> j2ObjcSources = provider.getSrcs();
    J2ObjcConfiguration j2objcConfiguration = ruleContext.getFragment(J2ObjcConfiguration.class);

    // Only perform J2ObjC dead code stripping if flag --j2objc_dead_code_removal is specified and
    // users have specified entry classes.
    boolean stripJ2ObjcDeadCode = j2objcConfiguration.removeDeadCode()
        && !provider.getEntryClasses().isEmpty();

    if (stripJ2ObjcDeadCode) {
      registerJ2ObjcDeadCodeRemovalActions(j2ObjcSources, provider.getEntryClasses());
    }

    for (J2ObjcSource j2ObjcSource : j2ObjcSources) {
      if (j2ObjcSource.hasSourceFiles()) {
        J2ObjcSource sourceToCompile =
            j2ObjcSource.getSourceType() == SourceType.JAVA && stripJ2ObjcDeadCode
                ? j2ObjcSource.toPrunedSource(ruleContext)
                : j2ObjcSource;
        IntermediateArtifacts intermediateArtifacts =
            ObjcRuleClasses.j2objcIntermediateArtifacts(ruleContext, sourceToCompile);
        CompilationArtifacts compilationArtifact = new CompilationArtifacts.Builder()
            .addNonArcSrcs(sourceToCompile.getObjcSrcs())
            .setIntermediateArtifacts(intermediateArtifacts)
            .setPchFile(Optional.<Artifact>absent())
            .build();
        // The module map that the above intermediateArtifacts would point to is a combination of
        // the current target and the java target. Instead, we want the one generated for that
        // source, but there is no easy way to get that. So for now, compile the source with the
        // module map (and therefore module name) for this rule.
        // TODO(bazel-team): Generate module maps along with the J2Objc sources, and use that here.
        Optional<CppModuleMap> moduleMap;
        if (ObjcRuleClasses.objcConfiguration(ruleContext).moduleMapsEnabled()) {
          moduleMap = Optional.of(ObjcRuleClasses.intermediateArtifacts(ruleContext).moduleMap());
        } else {
          moduleMap = Optional.absent();
        }
        registerCompileAndArchiveActions(
            compilationArtifact,
            intermediateArtifacts,
            objcProvider.toJ2ObjcOnlyProvider(),
            moduleMap,
            ruleContext.getConfiguration().isCodeCoverageEnabled(),
            false);
      }
    }

    return this;
  }

  /**
   * Registers actions that generates a module map for all {@link J2ObjcSource}s in
   * {@link J2ObjcSrcsProvider}.
   *
   * @return this compilation support
   */
  public CompilationSupport registerJ2ObjcGenerateModuleMapAction(J2ObjcSrcsProvider provider) {
    if (ObjcRuleClasses.objcConfiguration(ruleContext).moduleMapsEnabled()) {
      CppModuleMap moduleMap = ObjcRuleClasses.intermediateArtifacts(ruleContext).moduleMap();
      ImmutableSet.Builder<Artifact> headers = ImmutableSet.builder();
      for (J2ObjcSource j2ObjcSource : provider.getSrcs()) {
        headers.addAll(j2ObjcSource.getObjcHdrs());
      }
      registerGenerateModuleMapAction(moduleMap, headers.build(), ImmutableList.<Artifact>of());
    }
    return this;
  }

  private void registerJ2ObjcDeadCodeRemovalActions(Iterable<J2ObjcSource> j2ObjcSources,
      Iterable<String> entryClasses) {
    Artifact pruner = ruleContext.getPrerequisiteArtifact("$j2objc_dead_code_pruner", Mode.HOST);
    J2ObjcMappingFileProvider provider = ObjcRuleClasses.j2ObjcMappingFileProvider(ruleContext);
    NestedSet<Artifact> j2ObjcDependencyMappingFiles = provider.getDependencyMappingFiles();
    NestedSet<Artifact> j2ObjcHeaderMappingFiles = provider.getHeaderMappingFiles();

    for (J2ObjcSource j2ObjcSource : j2ObjcSources) {
      if (j2ObjcSource.getSourceType() == SourceType.JAVA
          && j2ObjcSource.hasSourceFiles()) {
        Iterable<Artifact> sourceArtifacts = j2ObjcSource.getObjcSrcs();
        Iterable<Artifact> prunedSourceArtifacts =
            j2ObjcSource.toPrunedSource(ruleContext).getObjcSrcs();
        PathFragment paramFilePath = FileSystemUtils.replaceExtension(
            j2ObjcSource.getTargetLabel().toPathFragment(), ".param.j2objc");
        Artifact paramFile = ruleContext.getUniqueDirectoryArtifact(
            "_j2objc_pruned",
            paramFilePath,
            ruleContext.getBinOrGenfilesDirectory());
        PathFragment objcFilePath = j2ObjcSource.getObjcFilePath();
        CustomCommandLine commandLine = CustomCommandLine.builder()
            .addJoinExecPaths("--input_files", ",", sourceArtifacts)
            .addJoinExecPaths("--output_files", ",", prunedSourceArtifacts)
            .addJoinExecPaths("--dependency_mapping_files", ",", j2ObjcDependencyMappingFiles)
            .addJoinExecPaths("--header_mapping_files", ",", j2ObjcHeaderMappingFiles)
            .add("--entry_classes").add(Joiner.on(",").join(entryClasses))
            .add("--objc_file_path").add(objcFilePath.getPathString())
            .build();

        ruleContext.registerAction(new ParameterFileWriteAction(
            ruleContext.getActionOwner(),
            paramFile,
            commandLine,
            ParameterFile.ParameterFileType.UNQUOTED, ISO_8859_1));
        ruleContext.registerAction(new SpawnAction.Builder()
            .setMnemonic("DummyPruner")
            .setExecutable(pruner)
            .addInput(pruner)
            .addInput(paramFile)
            .addInputs(sourceArtifacts)
            .addTransitiveInputs(j2ObjcDependencyMappingFiles)
            .addTransitiveInputs(j2ObjcHeaderMappingFiles)
            .setCommandLine(CustomCommandLine.builder()
                .addPaths("@%s", paramFile.getExecPath())
                .build())
            .addOutputs(prunedSourceArtifacts)
            .build(ruleContext));
      }
    }
  }

  /**
   * Sets compilation-related Xcode project information on the given provider builder.
   *
   * @param common common information about this rule's attributes and its dependencies
   * @return this compilation support
   */
  CompilationSupport addXcodeSettings(Builder xcodeProviderBuilder, ObjcCommon common) {
    ObjcConfiguration objcConfiguration = ObjcRuleClasses.objcConfiguration(ruleContext);
    for (CompilationArtifacts artifacts : common.getCompilationArtifacts().asSet()) {
      xcodeProviderBuilder.setCompilationArtifacts(artifacts);
    }

    // The include directory options ("-I") are parsed out of copts. The include directories are
    // added as non-propagated header search paths local to the associated Xcode target.
    Iterable<String> copts = Iterables.concat(
        objcConfiguration.getCopts(), attributes.copts(), attributes.optionsCopts());
    Iterable<String> includeDirOptions = Iterables.filter(copts, INCLUDE_DIR_OPTION_IN_COPTS);
    Iterable<String> coptsWithoutIncludeDirs = Iterables.filter(
        copts, Predicates.not(INCLUDE_DIR_OPTION_IN_COPTS));
    ImmutableList.Builder<PathFragment> nonPropagatedHeaderSearchPaths =
        new ImmutableList.Builder<>();
    for (String includeDirOption : includeDirOptions) {
      nonPropagatedHeaderSearchPaths.add(new PathFragment(includeDirOption.substring(2)));
    }

    xcodeProviderBuilder
        .addHeaders(attributes.hdrs())
        .addHeaders(attributes.textualHdrs())
        .addUserHeaderSearchPaths(ObjcCommon.userHeaderSearchPaths(ruleContext.getConfiguration()))
        .addHeaderSearchPaths("$(WORKSPACE_ROOT)", attributes.headerSearchPaths())
        .addHeaderSearchPaths("$(SDKROOT)/usr/include", attributes.sdkIncludes())
        .addNonPropagatedHeaderSearchPaths(
            "$(WORKSPACE_ROOT)", nonPropagatedHeaderSearchPaths.build())
        .addCompilationModeCopts(objcConfiguration.getCoptsForCompilationMode())
        .addCopts(coptsWithoutIncludeDirs);

    return this;
  }

  /**
   * Validates compilation-related attributes on this rule.
   *
   * @return this compilation support
   */
  CompilationSupport validateAttributes() {
    for (PathFragment absoluteInclude :
        Iterables.filter(attributes.includes(), PathFragment.IS_ABSOLUTE)) {
      ruleContext.attributeError(
          "includes", String.format(ABSOLUTE_INCLUDES_PATH_FORMAT, absoluteInclude));
    }

    if (ruleContext.attributes().has("srcs", BuildType.LABEL_LIST)) {
      Set<Artifact> hdrsSet = new HashSet<>(attributes.hdrs());
      Set<Artifact> srcsSet =
          new HashSet<>(ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET).list());

      // Check for overlap between srcs and hdrs.
      for (Artifact header : Sets.intersection(hdrsSet, srcsSet)) {
        String path = header.getRootRelativePath().toString();
        ruleContext.attributeWarning(
            "srcs", String.format(FILE_IN_SRCS_AND_HDRS_WARNING_FORMAT, path));
      }

      // Check for overlap between srcs and non_arc_srcs.
      Set<Artifact> nonArcSrcsSet =
          new HashSet<>(ruleContext.getPrerequisiteArtifacts("non_arc_srcs", Mode.TARGET).list());
      for (Artifact conflict : Sets.intersection(nonArcSrcsSet, srcsSet)) {
        String path = conflict.getRootRelativePath().toString();
        ruleContext.attributeError(
            "srcs", String.format(FILE_IN_SRCS_AND_NON_ARC_SRCS_ERROR_FORMAT, path));
      }
    }

    return this;
  }

  private CompilationSupport registerDsymActions() {
    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);
    ObjcConfiguration objcConfiguration = ObjcRuleClasses.objcConfiguration(ruleContext);

    Artifact dsymBundle = intermediateArtifacts.dsymBundle();
    Artifact linkedBinary =
        objcConfiguration.shouldStripBinary()
            ? intermediateArtifacts.unstrippedSingleArchitectureBinary()
            : intermediateArtifacts.strippedSingleArchitectureBinary();
    Artifact debugSymbolFile = intermediateArtifacts.dsymSymbol();
    Artifact dsymPlist = intermediateArtifacts.dsymPlist();

    PathFragment dsymOutputDir =
        replaceSuffix(
            dsymBundle.getExecPath(), IntermediateArtifacts.TMP_DSYM_BUNDLE_SUFFIX, ".app.dSYM");
    PathFragment dsymPlistZipEntry = dsymPlist.getExecPath().relativeTo(dsymOutputDir);
    PathFragment debugSymbolFileZipEntry =
        debugSymbolFile
            .getExecPath()
            .replaceName(linkedBinary.getFilename())
            .relativeTo(dsymOutputDir);

    StringBuilder unzipDsymCommand =
        new StringBuilder()
            .append(
                String.format(
                    "unzip -p %s %s > %s",
                    dsymBundle.getExecPathString(),
                    dsymPlistZipEntry,
                    dsymPlist.getExecPathString()))
            .append(
                String.format(
                    " && unzip -p %s %s > %s",
                    dsymBundle.getExecPathString(),
                    debugSymbolFileZipEntry,
                    debugSymbolFile.getExecPathString()));

    ruleContext.registerAction(
        new SpawnAction.Builder()
            .setMnemonic("UnzipDsym")
            .setShellCommand(unzipDsymCommand.toString())
            .addInput(dsymBundle)
            .addOutput(dsymPlist)
            .addOutput(debugSymbolFile)
            .build(ruleContext));

    Artifact dumpsyms = ruleContext.getPrerequisiteArtifact(":dumpsyms", Mode.HOST);
    Artifact breakpadFile = intermediateArtifacts.breakpadSym();
    ruleContext.registerAction(ObjcRuleClasses.spawnOnDarwinActionBuilder()
        .setMnemonic("GenBreakpad")
        .setProgressMessage("Generating breakpad file: " + ruleContext.getLabel())
        .setShellCommand(ImmutableList.of("/bin/bash", "-c"))
        .addInput(dumpsyms)
        .addInput(debugSymbolFile)
        .addArgument(String.format("%s %s > %s",
            ShellUtils.shellEscape(dumpsyms.getExecPathString()),
            ShellUtils.shellEscape(debugSymbolFile.getExecPathString()),
            ShellUtils.shellEscape(breakpadFile.getExecPathString())))
        .addOutput(breakpadFile)
        .build(ruleContext));
    return this;
  }

  private PathFragment replaceSuffix(PathFragment path, String suffix, String newSuffix) {
    // TODO(bazel-team): Throw instead of returning null?
    String name = path.getBaseName();
    if (name.endsWith(suffix)) {
      return path.replaceName(name.substring(0, name.length() - suffix.length()) + newSuffix);
    } else {
      return null;
    }
  }

  /**
   * Returns the name of Swift module for this target.
   */
  private String getModuleName() {
    // If we have module maps support, we need to use the generated module name, this way
    // clang can properly load objc part of the module via -import-underlying-module command.
    if (ObjcRuleClasses.objcConfiguration(ruleContext).moduleMapsEnabled()) {
      return ObjcRuleClasses.intermediateArtifacts(ruleContext).moduleMap().getName();
    }
    // Otherwise, just use target name, it doesn't matter.
    return ruleContext.getLabel().getName();
  }

  /**
   * Collector that, given a list of output artifacts, finds and registers coverage notes metadata
   * for any compilation action.
   */
  private static class ObjcCoverageMetadataCollector extends LocalMetadataCollector {

    @Override
    public void collectMetadataArtifacts(
        Iterable<Artifact> artifacts,
        AnalysisEnvironment analysisEnvironment,
        NestedSetBuilder<Artifact> metadataFilesBuilder) {
      for (Artifact artifact : artifacts) {
        Action action = analysisEnvironment.getLocalGeneratingAction(artifact);
        if (action.getMnemonic().equals("ObjcCompile")) {
          addOutputs(metadataFilesBuilder, action, ObjcRuleClasses.COVERAGE_NOTES);
        }
      }
    }
  }
  
  private static Iterable<PathFragment> uniqueParentDirectories(Iterable<PathFragment> paths) {
    ImmutableSet.Builder<PathFragment> parents = new ImmutableSet.Builder<>();
    for (PathFragment path : paths) {
      parents.add(path.getParentDirectory());
    }
    return parents.build();
  }

  /**
   * Returns the target string for swift compiler. For example, "x86_64-apple-ios8.2"
   */
  @VisibleForTesting
  static String swiftTarget(AppleConfiguration configuration) {
    return configuration.getIosCpu() + "-apple-ios" + configuration.getIosSdkVersion();
  }
  
  /**
   * Returns a list of clang flags used for all link and compile actions executed through clang.
   */
  private static List<String> commonLinkAndCompileFlagsForClang(
      ObjcProvider provider, ObjcConfiguration objcConfiguration,
      AppleConfiguration appleConfiguration) {
    ImmutableList.Builder<String> builder = new ImmutableList.Builder<>();
    Platform platform = Platform.forIosArch(appleConfiguration.getIosCpu());
    if (platform == Platform.IOS_SIMULATOR) {
      builder.add("-mios-simulator-version-min=" + objcConfiguration.getMinimumOs());
    } else {
      builder.add("-miphoneos-version-min=" + objcConfiguration.getMinimumOs());
    }

    if (objcConfiguration.generateDebugSymbols()) {
      builder.add("-g");
    }

    return builder
        .add("-arch", appleConfiguration.getIosCpu())
        .add("-isysroot", AppleToolchain.sdkDir())
        // TODO(bazel-team): Pass framework search paths to Xcodegen.
        .add("-F", AppleToolchain.sdkFrameworkDir(platform, appleConfiguration))
        // As of sdk8.1, XCTest is in a base Framework dir
        .add("-F", AppleToolchain.platformDeveloperFrameworkDir(appleConfiguration))
        // Add custom (non-SDK) framework search paths. For each framework foo/bar.framework,
        // include "foo" as a search path.
        .addAll(Interspersing.beforeEach(
            "-F",
            PathFragment.safePathStrings(uniqueParentDirectories(provider.get(FRAMEWORK_DIR)))))
        .build();
  }

  private static Iterable<String> compileFlagsForClang(AppleConfiguration configuration) {
    return Iterables.concat(
        AppleToolchain.DEFAULT_WARNINGS.values(),
        platformSpecificCompileFlagsForClang(configuration),
        configuration.getBitcodeMode().getCompilerFlags(),
        DEFAULT_COMPILER_FLAGS
    );
  }

  private static List<String> platformSpecificCompileFlagsForClang(
      AppleConfiguration configuration) {
    switch (Platform.forIosArch(configuration.getIosCpu())) {
      case IOS_DEVICE:
        return ImmutableList.of();
      case IOS_SIMULATOR:
        // These are added by Xcode when building, because the simulator is built on OSX
        // frameworks so we aim compile to match the OSX objc runtime.
        return ImmutableList.of(
            "-fexceptions",
            "-fasm-blocks",
            "-fobjc-abi-version=2",
            "-fobjc-legacy-dispatch");
      default:
        throw new AssertionError();
    }
  }
}
