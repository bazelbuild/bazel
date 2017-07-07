// Copyright 2016 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.DEFINE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.DYNAMIC_FRAMEWORK_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FORCE_LOAD_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.Flag.USES_CPP;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.HEADER;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.INCLUDE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.INCLUDE_SYSTEM;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LINK_INPUTS;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.MODULE_MAP;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.STATIC_FRAMEWORK_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.WEAK_SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.CLANG;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.CLANG_PLUSPLUS;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.COMPILABLE_SRCS_TYPE;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.DSYMUTIL;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.HEADERS;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.NON_ARC_SRCS_TYPE;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.PRECOMPILED_SRCS_TYPE;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.SRCS_TYPE;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnActionTemplate;
import com.google.devtools.build.lib.analysis.actions.SpawnActionTemplate.OutputPathMapper;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions.AppleBitcodeMode;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.rules.apple.Platform;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction.DotdFile;
import com.google.devtools.build.lib.rules.cpp.CppModuleMap;
import com.google.devtools.build.lib.rules.cpp.FdoSupportProvider;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Constructs command lines for objc compilation, archiving, and linking.  Uses hard-coded
 * command line templates.
 * 
 * TODO(b/28403953): Deprecate in favor of {@link CrosstoolCompilationSupport} in all objc rules.
 */
public class LegacyCompilationSupport extends CompilationSupport {

  /**
   * Frameworks implicitly linked to iOS, watchOS, and tvOS binaries when using legacy compilation.
   */
  @VisibleForTesting
  static final ImmutableList<SdkFramework> AUTOMATIC_SDK_FRAMEWORKS =
      ImmutableList.of(new SdkFramework("Foundation"), new SdkFramework("UIKit"));

  /**
   * A mapper that maps input ObjC source {@link Artifact.TreeFileArtifact}s to output object file
   * {@link Artifact.TreeFileArtifact}s.
   */
  private static final OutputPathMapper COMPILE_ACTION_TEMPLATE_OUTPUT_PATH_MAPPER =
      new OutputPathMapper() {
        @Override
        public PathFragment parentRelativeOutputPath(TreeFileArtifact inputTreeFileArtifact) {
          return FileSystemUtils.replaceExtension(
              inputTreeFileArtifact.getParentRelativePath(), ".o");
        }
      };

  /**
   * Returns information about the given rule's compilation artifacts. Dependencies specified in the
   * current rule's attributes are obtained via {@code ruleContext}. Output locations are determined
   * using the given {@code intermediateArtifacts} object. The fact that these are distinct objects
   * allows the caller to generate compilation actions pertaining to a configuration separate from
   * the current rule's configuration.
   */
  static CompilationArtifacts compilationArtifacts(
      RuleContext ruleContext, IntermediateArtifacts intermediateArtifacts) {
    PrerequisiteArtifacts srcs =
        ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET).errorsForNonMatching(SRCS_TYPE);
    return new CompilationArtifacts.Builder()
        .addSrcs(srcs.filter(COMPILABLE_SRCS_TYPE).list())
        .addNonArcSrcs(
            ruleContext
                .getPrerequisiteArtifacts("non_arc_srcs", Mode.TARGET)
                .errorsForNonMatching(NON_ARC_SRCS_TYPE)
                .list())
        .addPrivateHdrs(srcs.filter(HEADERS).list())
        .addPrecompiledSrcs(srcs.filter(PRECOMPILED_SRCS_TYPE).list())
        .setIntermediateArtifacts(intermediateArtifacts)
        .build();
  }

  /**
   * Creates a new legacy compilation support for the given rule and build configuration.
   *
   * <p>All actions will be created under the given build configuration, which may be different than
   * the current rule context configuration.
   *
   * <p>The compilation and linking flags will be retrieved from the given compilation attributes.
   * The names of the generated artifacts will be retrieved from the given intermediate artifacts.
   *
   * <p>By instantiating multiple compilation supports for the same rule but with intermediate
   * artifacts with different output prefixes, multiple archives can be compiled for the same rule
   * context.
   */
  LegacyCompilationSupport(
      RuleContext ruleContext,
      BuildConfiguration buildConfiguration,
      IntermediateArtifacts intermediateArtifacts,
      CompilationAttributes compilationAttributes,
      boolean useDeps,
      Map<String, NestedSet<Artifact>> outputGroupCollector,
      CcToolchainProvider toolchain,
      boolean isTestRule,
      boolean usePch) {
    super(
        ruleContext,
        buildConfiguration,
        intermediateArtifacts,
        compilationAttributes,
        useDeps,
        outputGroupCollector,
        toolchain,
        isTestRule,
        usePch);
  }

  @Override
  CompilationSupport registerCompileAndArchiveActions(
      CompilationArtifacts compilationArtifacts, ObjcProvider objcProvider,
      ExtraCompileArgs extraCompileArgs, Iterable<PathFragment> priorityHeaders,
      @Nullable CcToolchainProvider ccToolchain, @Nullable FdoSupportProvider fdoSupport) {
    registerGenerateModuleMapAction(compilationArtifacts);
    Optional<CppModuleMap> moduleMap;
    if (objcConfiguration.moduleMapsEnabled()) {
      moduleMap = Optional.of(intermediateArtifacts.moduleMap());
    } else {
      moduleMap = Optional.absent();
    }
    registerCompileAndArchiveActions(
        compilationArtifacts,
        objcProvider,
        extraCompileArgs,
        priorityHeaders,
        moduleMap);
    return this;
  }

  /**
   * Creates actions to compile each source file individually, and link all the compiled object
   * files into a single archive library.
   */
  private void registerCompileAndArchiveActions(
      CompilationArtifacts compilationArtifacts,
      ObjcProvider objcProvider,
      ExtraCompileArgs extraCompileArgs,
      Iterable<PathFragment> priorityHeaders,
      Optional<CppModuleMap> moduleMap) {
    ImmutableList.Builder<Artifact> objFilesBuilder = ImmutableList.builder();
    ImmutableList.Builder<ObjcHeaderThinningInfo> objcHeaderThinningInfos = ImmutableList.builder();

    for (Artifact sourceFile : compilationArtifacts.getSrcs()) {
      Artifact objFile = intermediateArtifacts.objFile(sourceFile);
      objFilesBuilder.add(objFile);

      if (objFile.isTreeArtifact()) {
        registerCompileActionTemplate(
            sourceFile,
            objFile,
            objcProvider,
            priorityHeaders,
            moduleMap,
            compilationArtifacts,
            Iterables.concat(extraCompileArgs, ImmutableList.of("-fobjc-arc")));
      } else {
        ObjcHeaderThinningInfo objcHeaderThinningInfo =
            registerCompileAction(
                sourceFile,
                objFile,
                objcProvider,
                priorityHeaders,
                moduleMap,
                compilationArtifacts,
                Iterables.concat(extraCompileArgs, ImmutableList.of("-fobjc-arc")));
        if (objcHeaderThinningInfo != null) {
          objcHeaderThinningInfos.add(objcHeaderThinningInfo);
        }
      }
    }
    for (Artifact nonArcSourceFile : compilationArtifacts.getNonArcSrcs()) {
      Artifact objFile = intermediateArtifacts.objFile(nonArcSourceFile);
      objFilesBuilder.add(objFile);
      if (objFile.isTreeArtifact()) {
        registerCompileActionTemplate(
            nonArcSourceFile,
            objFile,
            objcProvider,
            priorityHeaders,
            moduleMap,
            compilationArtifacts,
            Iterables.concat(extraCompileArgs, ImmutableList.of("-fno-objc-arc")));
      } else {
        ObjcHeaderThinningInfo objcHeaderThinningInfo =
            registerCompileAction(
                nonArcSourceFile,
                objFile,
                objcProvider,
                priorityHeaders,
                moduleMap,
                compilationArtifacts,
                Iterables.concat(extraCompileArgs, ImmutableList.of("-fno-objc-arc")));
        if (objcHeaderThinningInfo != null) {
          objcHeaderThinningInfos.add(objcHeaderThinningInfo);
        }
      }
    }

    objFilesBuilder.addAll(compilationArtifacts.getPrecompiledSrcs());

    ImmutableList<Artifact> objFiles = objFilesBuilder.build();
    outputGroupCollector.put(
        OutputGroupProvider.FILES_TO_COMPILE,
        NestedSetBuilder.<Artifact>stableOrder().addAll(objFiles).build());

    for (Artifact archive : compilationArtifacts.getArchive().asSet()) {
      registerArchiveActions(objFiles, archive);
    }

    registerHeaderScanningActions(
        objcHeaderThinningInfos.build(), objcProvider, compilationArtifacts);
  }

  private CustomCommandLine compileActionCommandLine(
      Artifact sourceFile,
      Artifact objFile,
      ObjcProvider objcProvider,
      Iterable<PathFragment> priorityHeaders,
      Optional<CppModuleMap> moduleMap,
      Optional<Artifact> pchFile,
      Optional<Artifact> dotdFile,
      Iterable<String> otherFlags,
      boolean collectCodeCoverage,
      boolean isCPlusPlusSource) {
    CustomCommandLine.Builder commandLine = new CustomCommandLine.Builder().add(CLANG);

    if (isCPlusPlusSource) {
      commandLine.add("-stdlib=libc++");
      commandLine.add("-std=gnu++11");
    }

    // The linker needs full debug symbol information to perform binary dead-code stripping.
    if (objcConfiguration.shouldStripBinary()) {
      commandLine.add("-g");
    }

    List<String> coverageFlags = ImmutableList.of();
    if (collectCodeCoverage) {
      if (buildConfiguration.isLLVMCoverageMapFormatEnabled()) {
        coverageFlags = CLANG_LLVM_COVERAGE_FLAGS;
      } else {
        coverageFlags = CLANG_GCOV_COVERAGE_FLAGS;
      }
    }

    commandLine
        .add(compileFlagsForClang(appleConfiguration))
        .add(commonLinkAndCompileFlagsForClang(objcProvider, objcConfiguration, appleConfiguration))
        .add(objcConfiguration.getCoptsForCompilationMode())
        .addBeforeEachPath(
            "-iquote", ObjcCommon.userHeaderSearchPaths(objcProvider, buildConfiguration))
        .addBeforeEachExecPath("-include", pchFile.asSet())
        .addBeforeEachPath("-I", priorityHeaders)
        .addBeforeEachPath("-I", objcProvider.get(INCLUDE))
        .addBeforeEachPath("-isystem", objcProvider.get(INCLUDE_SYSTEM))
        .add(otherFlags)
        .addFormatEach("-D%s", objcProvider.get(DEFINE))
        .add(coverageFlags)
        .add(getCompileRuleCopts());

    // Add input source file arguments
    commandLine.add("-c");
    if (sourceFile.isTreeArtifact()) {
      commandLine.addPlaceholderTreeArtifactExecPath(sourceFile);
    } else {
      commandLine.addPath(sourceFile.getExecPath());
    }

    // Add output object file arguments.
    commandLine.add("-o");
    if (objFile.isTreeArtifact()) {
      commandLine.addPlaceholderTreeArtifactExecPath(objFile);
    } else {
      commandLine.addPath(objFile.getExecPath());
    }

    // Add Dotd file arguments.
    if (dotdFile.isPresent()) {
      commandLine.add("-MD").addExecPath("-MF", dotdFile.get());
    }

    // Add module map arguments.
    if (moduleMap.isPresent()) {
      // If modules are enabled for the rule, -fmodules is added to the copts already. (This implies
      // module map usage). Otherwise, we need to pass -fmodule-maps.
      if (!attributes.enableModules()) {
        commandLine.add("-fmodule-maps");
      }
      // -fmodule-map-file only loads the module in Xcode 7, so we add the module maps's directory
      // to the include path instead.
      // TODO(bazel-team): Use -fmodule-map-file when Xcode 6 support is dropped.
      commandLine
          .add("-iquote")
          .add(moduleMap.get().getArtifact().getExecPath().getParentDirectory().toString())
          .add("-fmodule-name=" + moduleMap.get().getName());
    }

    return commandLine.build();
  }

  @Nullable
  private ObjcHeaderThinningInfo registerCompileAction(
      Artifact sourceFile,
      Artifact objFile,
      ObjcProvider objcProvider,
      Iterable<PathFragment> priorityHeaders,
      Optional<CppModuleMap> moduleMap,
      CompilationArtifacts compilationArtifacts,
      Iterable<String> otherFlags) {
    boolean isCPlusPlusSource = ObjcRuleClasses.CPP_SOURCES.matches(sourceFile.getExecPath());
    boolean runCodeCoverage =
        buildConfiguration.isCodeCoverageEnabled() && ObjcRuleClasses.isInstrumentable(sourceFile);
    DotdFile dotdFile = intermediateArtifacts.dotdFile(sourceFile);

    CustomCommandLine commandLine =
        compileActionCommandLine(
            sourceFile,
            objFile,
            objcProvider,
            priorityHeaders,
            moduleMap,
            getPchFile(),
            Optional.of(dotdFile.artifact()),
            otherFlags,
            runCodeCoverage,
            isCPlusPlusSource);

    Optional<Artifact> gcnoFile = Optional.absent();
    if (runCodeCoverage && !buildConfiguration.isLLVMCoverageMapFormatEnabled()) {
      gcnoFile = Optional.of(intermediateArtifacts.gcnoFile(sourceFile));
    }

    NestedSet<Artifact> moduleMapInputs = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    if (objcConfiguration.moduleMapsEnabled()) {
      moduleMapInputs = objcProvider.get(MODULE_MAP);
    }

    // TODO(bazel-team): Remove private headers from inputs once they're added to the provider.
    ObjcCompileAction.Builder compileBuilder =
        ObjcCompileAction.Builder.createObjcCompileActionBuilderWithAppleEnv(
                appleConfiguration, appleConfiguration.getSingleArchPlatform())
            .setDotdPruningPlan(objcConfiguration.getDotdPruningPlan())
            .setSourceFile(sourceFile)
            .addTransitiveHeaders(objcProvider.get(HEADER))
            .addHeaders(compilationArtifacts.getPrivateHdrs())
            .addTransitiveMandatoryInputs(moduleMapInputs)
            .addTransitiveMandatoryInputs(objcProvider.get(STATIC_FRAMEWORK_FILE))
            .addTransitiveMandatoryInputs(objcProvider.get(DYNAMIC_FRAMEWORK_FILE))
            .setDotdFile(dotdFile)
            .addMandatoryInputs(getPchFile().asSet());

    Artifact headersListFile = null;
    if (isHeaderThinningEnabled()
        && SOURCES_FOR_HEADER_THINNING.matches(sourceFile.getFilename())) {
      headersListFile = intermediateArtifacts.headersListFile(sourceFile);
      compileBuilder.setHeadersListFile(headersListFile);
    }

    ruleContext.registerAction(
        compileBuilder
            .setMnemonic("ObjcCompile")
            .setExecutable(xcrunwrapper(ruleContext))
            .setCommandLine(commandLine)
            .addOutput(objFile)
            .addOutputs(gcnoFile.asSet())
            .addOutput(dotdFile.artifact())
            .build(ruleContext));

    return headersListFile == null
        ? null
        : new ObjcHeaderThinningInfo(
            sourceFile, headersListFile, ImmutableList.copyOf(commandLine.arguments()));
  }

  /**
   * Registers a SpawnActionTemplate to compile the source file tree artifact, {@code sourceFiles},
   * which can contain multiple concrete source files unknown at analysis time. At execution time,
   * the SpawnActionTemplate will register one ObjcCompile action for each individual source file
   * under {@code sourceFiles}.
   *
   * <p>Note that this method currently does not support code coverage and sources other than ObjC
   * sources.
   *
   * @param sourceFiles tree artifact containing source files to compile
   * @param objFiles tree artifact containing object files compiled from {@code sourceFiles}
   * @param objcProvider ObjcProvider instance for this invocation
   * @param priorityHeaders priority headers to be included before the dependency headers
   * @param moduleMap the module map generated from the associated headers
   * @param compilationArtifacts the CompilationArtifacts instance for this invocation
   * @param otherFlags extra compilation flags to add to the compile action command line
   */
  private void registerCompileActionTemplate(
      Artifact sourceFiles,
      Artifact objFiles,
      ObjcProvider objcProvider,
      Iterable<PathFragment> priorityHeaders,
      Optional<CppModuleMap> moduleMap,
      CompilationArtifacts compilationArtifacts,
      Iterable<String> otherFlags) {
    CustomCommandLine commandLine =
        compileActionCommandLine(
            sourceFiles,
            objFiles,
            objcProvider,
            priorityHeaders,
            moduleMap,
            getPchFile(),
            Optional.<Artifact>absent(),
            otherFlags,
            /* runCodeCoverage=*/ false,
            /* isCPlusPlusSource=*/ false);

    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);
    Platform platform = appleConfiguration.getSingleArchPlatform();

    NestedSet<Artifact> moduleMapInputs = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    if (objcConfiguration.moduleMapsEnabled()) {
      moduleMapInputs = objcProvider.get(MODULE_MAP);
    }

    ruleContext.registerAction(
        new SpawnActionTemplate.Builder(sourceFiles, objFiles)
            .setMnemonics("ObjcCompileActionTemplate", "ObjcCompile")
            .setExecutable(xcrunwrapper(ruleContext))
            .setCommandLineTemplate(commandLine)
            .setEnvironment(ObjcRuleClasses.appleToolchainEnvironment(appleConfiguration, platform))
            .setExecutionInfo(ObjcRuleClasses.darwinActionExecutionRequirement())
            .setOutputPathMapper(COMPILE_ACTION_TEMPLATE_OUTPUT_PATH_MAPPER)
            .addCommonTransitiveInputs(objcProvider.get(HEADER))
            .addCommonTransitiveInputs(moduleMapInputs)
            .addCommonInputs(compilationArtifacts.getPrivateHdrs())
            .addCommonTransitiveInputs(objcProvider.get(STATIC_FRAMEWORK_FILE))
            .addCommonTransitiveInputs(objcProvider.get(DYNAMIC_FRAMEWORK_FILE))
            .addCommonInputs(getPchFile().asSet())
            .build(ruleContext.getActionOwner()));
  }

  private void registerArchiveActions(List<Artifact> objFiles, Artifact archive) {
    Artifact objList = intermediateArtifacts.archiveObjList();
    registerObjFilelistAction(objFiles, objList);
    registerArchiveAction(objFiles, archive);
  }

  private void registerArchiveAction(
      Iterable<Artifact> objFiles,
      Artifact archive) {
    Artifact objList = intermediateArtifacts.archiveObjList();
    ruleContext.registerAction(ObjcRuleClasses.spawnAppleEnvActionBuilder(
                appleConfiguration, appleConfiguration.getSingleArchPlatform())
            .setMnemonic("ObjcLink")
            .setExecutable(libtool(ruleContext))
            .setCommandLine(new CustomCommandLine.Builder()
                    .add("-static")
                    .add("-filelist").add(objList.getExecPathString())
                    .add("-arch_only").add(appleConfiguration.getSingleArchitecture())
                    .add("-syslibroot").add(AppleToolchain.sdkDir())
                    .add("-o").add(archive.getExecPathString())
                    .build())
            .addInputs(objFiles)
            .addInput(objList)
            .addOutput(archive)
            .build(ruleContext));
  }

  @Override
  protected CompilationSupport registerFullyLinkAction(
      ObjcProvider objcProvider, Iterable<Artifact> inputArtifacts, Artifact outputArchive,
      @Nullable CcToolchainProvider ccToolchain, @Nullable FdoSupportProvider fdoSupport) {
    ruleContext.registerAction(
        ObjcRuleClasses.spawnAppleEnvActionBuilder(
                appleConfiguration, appleConfiguration.getSingleArchPlatform())
            .setMnemonic("ObjcLink")
            .setExecutable(libtool(ruleContext))
            .setCommandLine(
                new CustomCommandLine.Builder()
                    .add("-static")
                    .add("-arch_only").add(appleConfiguration.getSingleArchitecture())
                    .add("-syslibroot").add(AppleToolchain.sdkDir())
                    .add("-o").add(outputArchive.getExecPathString())
                    .addExecPaths(inputArtifacts)
                    .build())
            .addInputs(inputArtifacts)
            .addOutput(outputArchive)
            .build(ruleContext));
    return this;
  }

  @Override
  CompilationSupport registerLinkActions(
      ObjcProvider objcProvider,
      J2ObjcMappingFileProvider j2ObjcMappingFileProvider,
      J2ObjcEntryClassProvider j2ObjcEntryClassProvider,
      ExtraLinkArgs extraLinkArgs,
      Iterable<Artifact> extraLinkInputs,
      DsymOutputType dsymOutputType,
      CcToolchainProvider toolchain) {
    Optional<Artifact> dsymBundleZip;
    Optional<Artifact> linkmap;
    Optional<Artifact> bitcodeSymbolMap;
    if (objcConfiguration.generateDsym()) {
      registerDsymActions(dsymOutputType);
      dsymBundleZip = Optional.of(intermediateArtifacts.tempDsymBundleZip(dsymOutputType));
    } else {
      dsymBundleZip = Optional.absent();
    }

    Iterable<Artifact> prunedJ2ObjcArchives =
        computeAndStripPrunedJ2ObjcArchives(
            j2ObjcEntryClassProvider, j2ObjcMappingFileProvider, objcProvider);

    if (objcConfiguration.generateLinkmap()) {
      linkmap = Optional.of(intermediateArtifacts.linkmap());
    } else {
      linkmap = Optional.absent();
    }

    if (appleConfiguration.getBitcodeMode() == AppleBitcodeMode.EMBEDDED) {
      bitcodeSymbolMap = Optional.of(intermediateArtifacts.bitcodeSymbolMap());
    } else {
      bitcodeSymbolMap = Optional.absent();
    }

    registerLinkAction(
        objcProvider,
        extraLinkArgs,
        extraLinkInputs,
        dsymBundleZip,
        prunedJ2ObjcArchives,
        linkmap,
        bitcodeSymbolMap);
    return this;
  }

  private StrippingType getStrippingType(CommandLine commandLine) {
    return Iterables.contains(commandLine.arguments(), "-dynamiclib")
        ? StrippingType.DYNAMIC_LIB
        : StrippingType.DEFAULT;
  }

  private void registerLinkAction(
      ObjcProvider objcProvider,
      ExtraLinkArgs extraLinkArgs,
      Iterable<Artifact> extraLinkInputs,
      Optional<Artifact> dsymBundleZip,
      Iterable<Artifact> prunedJ2ObjcArchives,
      Optional<Artifact> linkmap,
      Optional<Artifact> bitcodeSymbolMap) {
    Artifact binaryToLink = getBinaryToLink();

    ImmutableList<Artifact> objcLibraries = objcProvider.getObjcLibraries();
    ImmutableList<Artifact> ccLibraries = objcProvider.getCcLibraries();
    ImmutableList<Artifact> bazelBuiltLibraries = Iterables.isEmpty(prunedJ2ObjcArchives)
            ? objcLibraries : substituteJ2ObjcPrunedLibraries(objcProvider);
    CommandLine commandLine =
        linkCommandLine(
            extraLinkArgs,
            objcProvider,
            binaryToLink,
            dsymBundleZip,
            ccLibraries,
            bazelBuiltLibraries,
            linkmap,
            bitcodeSymbolMap);
    ruleContext.registerAction(
        ObjcRuleClasses.spawnAppleEnvActionBuilder(
                appleConfiguration, appleConfiguration.getSingleArchPlatform())
            .setMnemonic("ObjcLink")
            .setShellCommand(ImmutableList.of("/bin/bash", "-c"))
            .setCommandLine(new SingleArgCommandLine(commandLine))
            .addOutput(binaryToLink)
            .addOutputs(dsymBundleZip.asSet())
            .addOutputs(linkmap.asSet())
            .addOutputs(bitcodeSymbolMap.asSet())
            .addInputs(bazelBuiltLibraries)
            .addTransitiveInputs(objcProvider.get(IMPORTED_LIBRARY))
            .addTransitiveInputs(objcProvider.get(STATIC_FRAMEWORK_FILE))
            .addTransitiveInputs(objcProvider.get(DYNAMIC_FRAMEWORK_FILE))
            .addTransitiveInputs(objcProvider.get(LINK_INPUTS))
            .addInputs(ccLibraries)
            .addInputs(extraLinkInputs)
            .addInputs(prunedJ2ObjcArchives)
            .addInput(intermediateArtifacts.linkerObjList())
            .addInput(xcrunwrapper(ruleContext).getExecutable())
            .build(ruleContext));

    if (objcConfiguration.shouldStripBinary()) {
      registerBinaryStripAction(binaryToLink, getStrippingType(commandLine));
    }
  }

  @Override
  protected Set<String> frameworkNames(ObjcProvider objcProvider) {
    Set<String> names = new LinkedHashSet<>();
    Iterables.addAll(names, SdkFramework.names(AUTOMATIC_SDK_FRAMEWORKS));
    names.addAll(super.frameworkNames(objcProvider));
    return names;
  }

  private CommandLine linkCommandLine(
      ExtraLinkArgs extraLinkArgs,
      ObjcProvider objcProvider,
      Artifact linkedBinary,
      Optional<Artifact> dsymBundleZip,
      Iterable<Artifact> ccLibraries,
      Iterable<Artifact> bazelBuiltLibraries,
      Optional<Artifact> linkmap,
      Optional<Artifact> bitcodeSymbolMap) {
    Iterable<String> libraryNames = libraryNames(objcProvider);

    CustomCommandLine.Builder commandLine = CustomCommandLine.builder()
            .addPath(xcrunwrapper(ruleContext).getExecutable().getExecPath());
    if (objcProvider.is(USES_CPP)) {
      commandLine
        .add(CLANG_PLUSPLUS)
        .add("-stdlib=libc++")
        .add("-std=gnu++11");
    } else {
      commandLine.add(CLANG);
    }

    // TODO(b/36562173): Replace the "!isTestRule" condition with the presence of "-bundle" in
    // the command line.
    if (objcConfiguration.shouldStripBinary() && !isTestRule) {
      commandLine.add("-dead_strip").add("-no_dead_strip_inits_and_terms");
    }

    Iterable<Artifact> ccLibrariesToForceLoad =
        Iterables.filter(ccLibraries, ALWAYS_LINKED_CC_LIBRARY);

    ImmutableSet<Artifact> forceLinkArtifacts = ImmutableSet.<Artifact>builder()
            .addAll(objcProvider.get(FORCE_LOAD_LIBRARY))
            .addAll(ccLibrariesToForceLoad).build();

    Artifact inputFileList = intermediateArtifacts.linkerObjList();
    Iterable<Artifact> objFiles =
        Iterables.concat(bazelBuiltLibraries, objcProvider.get(IMPORTED_LIBRARY), ccLibraries);
    // Clang loads archives specified in filelists and also specified as -force_load twice,
    // resulting in duplicate symbol errors unless they are deduped.
    objFiles = Iterables.filter(objFiles, Predicates.not(Predicates.in(forceLinkArtifacts)));

    registerObjFilelistAction(objFiles, inputFileList);

    if (objcConfiguration.shouldPrioritizeStaticLibs()) {
      commandLine.add("-filelist").add(inputFileList.getExecPathString());
    }

    AppleBitcodeMode bitcodeMode = appleConfiguration.getBitcodeMode();
    commandLine.add(bitcodeMode.getCompileAndLinkFlags());

    if (bitcodeMode == AppleBitcodeMode.EMBEDDED) {
      commandLine.add("-Xlinker").add("-bitcode_verify");
      commandLine.add("-Xlinker").add("-bitcode_hide_symbols");
      commandLine
          .add("-Xlinker")
          .add("-bitcode_symbol_map")
          .add("-Xlinker")
          .add(bitcodeSymbolMap.get().getExecPathString());
    }

    commandLine
        .add(commonLinkAndCompileFlagsForClang(objcProvider, objcConfiguration, appleConfiguration))
        .add("-Xlinker")
        .add("-objc_abi_version")
        .add("-Xlinker")
        .add("2")
        // Set the rpath so that at runtime dylibs can be loaded from the bundle root's "Frameworks"
        // directory.
        .add("-Xlinker")
        .add("-rpath")
        .add("-Xlinker")
        .add("@executable_path/Frameworks")
        .add("-fobjc-link-runtime")
        .add(DEFAULT_LINKER_FLAGS)
        .addBeforeEach("-framework", frameworkNames(objcProvider))
        .addBeforeEach("-weak_framework", SdkFramework.names(objcProvider.get(WEAK_SDK_FRAMEWORK)))
        .addFormatEach("-l%s", libraryNames);

    if (!objcConfiguration.shouldPrioritizeStaticLibs()) {
      commandLine.add("-filelist").add(inputFileList.getExecPathString());
    }

    commandLine
        .addExecPath("-o", linkedBinary)
        .addBeforeEachExecPath("-force_load", forceLinkArtifacts)
        .add(extraLinkArgs)
        .add(objcProvider.get(ObjcProvider.LINKOPT));

    if (buildConfiguration.isCodeCoverageEnabled()) {
      if (buildConfiguration.isLLVMCoverageMapFormatEnabled()) {
        commandLine.add(LINKER_LLVM_COVERAGE_FLAGS);
      } else {
        commandLine.add(LINKER_COVERAGE_FLAGS);
      }
    }

    for (String linkopt : attributes.linkopts()) {
      commandLine.add("-Wl," + linkopt);
    }

    if (linkmap.isPresent()) {
      commandLine
        .add("-Xlinker -map")
        .add("-Xlinker " + linkmap.get().getExecPath());
    }

    // Call to dsymutil for debug symbol generation must happen in the link action.
    // All debug symbol information is encoded in object files inside archive files. To generate
    // the debug symbol bundle, dsymutil will look inside the linked binary for the encoded
    // absolute paths to archive files, which are only valid in the link action.
    if (dsymBundleZip.isPresent()) {
      PathFragment dsymPath = FileSystemUtils.removeExtension(dsymBundleZip.get().getExecPath());
      commandLine
          .add("&&")
          .addPath(xcrunwrapper(ruleContext).getExecutable().getExecPath())
          .add(DSYMUTIL)
          .add(linkedBinary.getExecPathString())
          .add("-o " + dsymPath)
          .add("&& zipped_bundle=${PWD}/" + dsymBundleZip.get().getShellEscapedExecPathString())
          .add("&& cd " + dsymPath)
          .add("&& /usr/bin/zip -q -r \"${zipped_bundle}\" .");
    }

    return commandLine.build();
  }

  /**
   * Command line that converts its input's arg array to a single input.
   *
   * <p>Required as a hack to the link command line because that may contain two commands, which are
   * then passed to {@code /bin/bash -c}, and accordingly need to be a single argument.
   */
  @Immutable // if original is immutable
  private static final class SingleArgCommandLine extends CommandLine {
    private final CommandLine original;

    private SingleArgCommandLine(CommandLine original) {
      this.original = original;
    }

    @Override
    public Iterable<String> arguments() {
      return ImmutableList.of(Joiner.on(' ').join(original.arguments()));
    }
  }

  /** Returns a list of clang flags used for all link and compile actions executed through clang. */
  private List<String> commonLinkAndCompileFlagsForClang(
      ObjcProvider provider, ObjcConfiguration objcConfiguration,
      AppleConfiguration appleConfiguration) {
    ImmutableList.Builder<String> builder = new ImmutableList.Builder<>();
    Platform platform = appleConfiguration.getSingleArchPlatform();
    String minOSVersionArg;
    switch (platform) {
      case IOS_SIMULATOR:
        minOSVersionArg = "-mios-simulator-version-min";
        break;
      case IOS_DEVICE:
        minOSVersionArg = "-miphoneos-version-min";
        break;
      case WATCHOS_SIMULATOR:
        minOSVersionArg = "-mwatchos-simulator-version-min";
        break;
      case WATCHOS_DEVICE:
        minOSVersionArg = "-mwatchos-version-min";
        break;
      case TVOS_SIMULATOR:
        minOSVersionArg = "-mtvos-simulator-version-min";
        break;
      case TVOS_DEVICE:
        minOSVersionArg = "-mtvos-version-min";
        break;
      default:
        throw new IllegalArgumentException("Unhandled platform " + platform);
    }
    DottedVersion minOSVersion = appleConfiguration.getMinimumOsForPlatformType(platform.getType());
    builder.add(minOSVersionArg + "=" + minOSVersion);

    if (objcConfiguration.generateDsym()) {
      builder.add("-g");
    }

    return builder
        .add("-arch", appleConfiguration.getSingleArchitecture())
        .add("-isysroot", AppleToolchain.sdkDir())
        // TODO(bazel-team): Pass framework search paths to Xcodegen.
        .addAll(commonFrameworkFlags(provider, appleConfiguration))
        .build();
  }

  private static Iterable<String> compileFlagsForClang(AppleConfiguration configuration) {
    return Iterables.concat(
        AppleToolchain.DEFAULT_WARNINGS.values(),
        platformSpecificCompileFlagsForClang(configuration),
        configuration.getBitcodeMode().getCompileAndLinkFlags(),
        DEFAULT_COMPILER_FLAGS
    );
  }

  private static List<String> platformSpecificCompileFlagsForClang(
      AppleConfiguration configuration) {
    switch (configuration.getSingleArchPlatform()) {
      case IOS_DEVICE:
      case WATCHOS_DEVICE:
      case TVOS_DEVICE:
        return ImmutableList.of();
      case IOS_SIMULATOR:
      case WATCHOS_SIMULATOR:
      case TVOS_SIMULATOR:
        return SIMULATOR_COMPILE_FLAGS;
      default:
        throw new AssertionError();
    }
  }
}
