// Copyright 2015 Google Inc. All rights reserved.
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

import static com.google.devtools.build.lib.rules.objc.J2ObjcSource.SourceType;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.DEFINE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FORCE_LOAD_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FRAMEWORK_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FRAMEWORK_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.Flag.USES_CPP;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.HEADER;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.INCLUDE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_DYLIB;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.WEAK_SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.CLANG;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.CLANG_PLUSPLUS;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.DSYMUTIL;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.NON_ARC_SRCS_TYPE;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.SRCS_TYPE;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs;
import com.google.devtools.build.lib.rules.objc.ObjcCommon.CompilationAttributes;
import com.google.devtools.build.lib.rules.objc.XcodeProvider.Builder;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.List;

/**
 * Support for rules that compile sources. Provides ways to determine files that should be output,
 * registering Xcode settings and generating the various actions that might be needed for
 * compilation.
 *
 * <p>Methods on this class can be called in any order without impacting the result.
 */
final class CompilationSupport {

  @VisibleForTesting
  static final String ABSOLUTE_INCLUDES_PATH_FORMAT =
      "The path '%s' is absolute, but only relative paths are allowed.";

  @VisibleForTesting
  static final ImmutableList<String> LINKER_COVERAGE_FLAGS =
      ImmutableList.of("-ftest-coverage", "-fprofile-arcs");

  @VisibleForTesting
  static final ImmutableList<String> CLANG_COVERAGE_FLAGS =
      ImmutableList.of("-fprofile-arcs", "-ftest-coverage", "-fprofile-dir=./coverage_output");

  private static final String FRAMEWORK_SUFFIX = ".framework";

  /**
   * Iterable wrapper providing strong type safety for arguments to binary linking.
   */
  static final class ExtraLinkArgs extends IterableWrapper<String> {
    ExtraLinkArgs(String... args) {
      super(args);
    }
  }

  /**
   * Returns information about the given rule's compilation artifacts.
   */
  // TODO(bazel-team): Remove this information from ObjcCommon and move it internal to this class.
  static CompilationArtifacts compilationArtifacts(RuleContext ruleContext) {
    return new CompilationArtifacts.Builder()
        .addSrcs(ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET)
            .errorsForNonMatching(SRCS_TYPE)
            .list())
        .addNonArcSrcs(ruleContext.getPrerequisiteArtifacts("non_arc_srcs", Mode.TARGET)
            .errorsForNonMatching(NON_ARC_SRCS_TYPE)
            .list())
        .setIntermediateArtifacts(ObjcRuleClasses.intermediateArtifacts(ruleContext))
        .setPchFile(Optional.fromNullable(ruleContext.getPrerequisiteArtifact("pch", Mode.TARGET)))
        .build();
  }

  private final RuleContext ruleContext;
  private final CompilationAttributes attributes;

  /**
   * Creates a new compilation support for the given rule.
   */
  CompilationSupport(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
    this.attributes = new CompilationAttributes(ruleContext);
  }

  /**
   * Registers all actions necessary to compile this rule's sources and archive them.
   *
   * @param common common information about this rule and its dependencies
   * @return this compilation support
   */
  CompilationSupport registerCompileAndArchiveActions(ObjcCommon common) {
    if (common.getCompilationArtifacts().isPresent()) {
      registerCompileAndArchiveActions(
          common.getCompilationArtifacts().get(),
          ObjcRuleClasses.intermediateArtifacts(ruleContext),
          common.getObjcProvider(),
          ruleContext.getConfiguration().isCodeCoverageEnabled());
    }
    return this;
  }

  /**
   * Creates actions to compile each source file individually, and link all the compiled object
   * files into a single archive library.
   */
  private void registerCompileAndArchiveActions(CompilationArtifacts compilationArtifacts,
      IntermediateArtifacts intermediateArtifacts, ObjcProvider objcProvider,
      boolean isCodeCoverageEnabled) {
    ImmutableList.Builder<Artifact> objFiles = new ImmutableList.Builder<>();
    for (Artifact sourceFile : compilationArtifacts.getSrcs()) {
      Artifact objFile = intermediateArtifacts.objFile(sourceFile);
      objFiles.add(objFile);
      registerCompileAction(sourceFile, objFile, compilationArtifacts.getPchFile(),
          objcProvider, intermediateArtifacts, ImmutableList.of("-fobjc-arc"),
          isCodeCoverageEnabled);
    }
    for (Artifact nonArcSourceFile : compilationArtifacts.getNonArcSrcs()) {
      Artifact objFile = intermediateArtifacts.objFile(nonArcSourceFile);
      objFiles.add(objFile);
      registerCompileAction(nonArcSourceFile, objFile, compilationArtifacts.getPchFile(),
          objcProvider, intermediateArtifacts, ImmutableList.of("-fno-objc-arc"),
          isCodeCoverageEnabled);
    }
    for (Artifact archive : compilationArtifacts.getArchive().asSet()) {
      registerArchiveActions(intermediateArtifacts, objFiles, archive);
    }
  }

  private void registerCompileAction(
      Artifact sourceFile,
      Artifact objFile,
      Optional<Artifact> pchFile,
      ObjcProvider objcProvider,
      IntermediateArtifacts intermediateArtifacts,
      Iterable<String> otherFlags,
      boolean isCodeCoverageEnabled) {
    ObjcConfiguration objcConfiguration = ObjcRuleClasses.objcConfiguration(ruleContext);
    ImmutableList.Builder<String> coverageFlags = new ImmutableList.Builder<>();
    ImmutableList.Builder<Artifact> gcnoFiles = new ImmutableList.Builder<>();
    if (isCodeCoverageEnabled) {
      coverageFlags.addAll(CLANG_COVERAGE_FLAGS);
      gcnoFiles.add(intermediateArtifacts.gcnoFile(sourceFile));
    }
    CustomCommandLine.Builder commandLine = new CustomCommandLine.Builder();
    if (ObjcRuleClasses.CPP_SOURCES.matches(sourceFile.getExecPath())) {
      commandLine.add("-stdlib=libc++");
    }
    commandLine
        .add(IosSdkCommands.compileFlagsForClang(objcConfiguration))
        .add(IosSdkCommands.commonLinkAndCompileFlagsForClang(
            objcProvider, objcConfiguration))
        .add(objcConfiguration.getCoptsForCompilationMode())
        .addBeforeEachPath(
            "-iquote", ObjcCommon.userHeaderSearchPaths(ruleContext.getConfiguration()))
        .addBeforeEachExecPath("-include", pchFile.asSet())
        .addBeforeEachPath("-I", objcProvider.get(INCLUDE))
        .add(otherFlags)
        .addFormatEach("-D%s", objcProvider.get(DEFINE))
        .add(coverageFlags.build())
        .add(objcConfiguration.getCopts())
        .add(attributes.copts())
        .add(attributes.optionsCopts())
        .addExecPath("-c", sourceFile)
        .addExecPath("-o", objFile);

    ruleContext.registerAction(ObjcRuleClasses.spawnOnDarwinActionBuilder()
        .setMnemonic("ObjcCompile")
        .setExecutable(CLANG)
        .setCommandLine(commandLine.build())
        .addInput(sourceFile)
        .addOutput(objFile)
        .addOutputs(gcnoFiles.build())
        .addTransitiveInputs(objcProvider.get(HEADER))
        .addTransitiveInputs(objcProvider.get(FRAMEWORK_FILE))
        .addInputs(pchFile.asSet())
        .build(ruleContext));
  }

  private void registerArchiveActions(IntermediateArtifacts intermediateArtifacts,
      ImmutableList.Builder<Artifact> objFiles, Artifact archive) {
    for (Action action : archiveActions(ruleContext, objFiles.build(), archive,
        ObjcRuleClasses.objcConfiguration(ruleContext),
        intermediateArtifacts.objList())) {
      ruleContext.registerAction(action);
    }
  }

  private static Iterable<Action> archiveActions(
      ActionConstructionContext context,
      Iterable<Artifact> objFiles,
      Artifact archive,
      ObjcConfiguration objcConfiguration,
      Artifact objList) {

    ImmutableList.Builder<Action> actions = new ImmutableList.Builder<>();

    actions.add(new FileWriteAction(
        context.getActionOwner(),
        objList,
        Artifact.joinExecPaths("\n", objFiles),
        /*makeExecutable=*/ false));

    actions.add(ObjcRuleClasses.spawnOnDarwinActionBuilder()
        .setMnemonic("ObjcLink")
        .setExecutable(ObjcRuleClasses.LIBTOOL)
        .setCommandLine(new CustomCommandLine.Builder()
            .add("-static")
            .add("-filelist").add(objList.getExecPathString())
            .add("-arch_only").add(objcConfiguration.getIosCpu())
            .add("-syslibroot").add(IosSdkCommands.sdkDir(objcConfiguration))
            .add("-o").add(archive.getExecPathString())
            .build())
        .addInputs(objFiles)
        .addInput(objList)
        .addOutput(archive)
        .build(context));

    return actions.build();
  }

  /**
   * Registers any actions necessary to link this rule and its dependencies. Debug symbols are
   * generated if {@link ObjcConfiguration#generateDebugSymbols()} is set.
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

  private void registerLinkAction(ObjcProvider objcProvider, ExtraLinkArgs extraLinkArgs,
      Iterable<Artifact> extraLinkInputs, Optional<Artifact> dsymBundle) {
    Artifact linkedBinary =
        ObjcRuleClasses.intermediateArtifacts(ruleContext).singleArchitectureBinary();

    ImmutableList<Artifact> ccLibraries = ccLibraries(objcProvider);
    ruleContext.registerAction(
        ObjcRuleClasses.spawnOnDarwinActionBuilder()
            .setMnemonic("ObjcLink")
            .setShellCommand(ImmutableList.of("/bin/bash", "-c"))
            .setCommandLine(
                linkCommandLine(extraLinkArgs, objcProvider, linkedBinary, dsymBundle, ccLibraries))
            .addOutput(linkedBinary)
            .addOutputs(dsymBundle.asSet())
            .addTransitiveInputs(objcProvider.get(LIBRARY))
            .addTransitiveInputs(objcProvider.get(IMPORTED_LIBRARY))
            .addTransitiveInputs(objcProvider.get(FRAMEWORK_FILE))
            .addInputs(ccLibraries)
            .addInputs(extraLinkInputs)
            .build(ruleContext));
  }

  private ImmutableList<Artifact> ccLibraries(ObjcProvider objcProvider) {
    ImmutableList.Builder<Artifact> ccLibraryBuilder = ImmutableList.builder();
    for (LinkerInputs.LibraryToLink libraryToLink : objcProvider.get(ObjcProvider.CC_LIBRARY)) {
      ccLibraryBuilder.add(libraryToLink.getArtifact());
    }
    return ccLibraryBuilder.build();
  }

  private CommandLine linkCommandLine(ExtraLinkArgs extraLinkArgs,
      ObjcProvider objcProvider, Artifact linkedBinary, Optional<Artifact> dsymBundle,
      ImmutableList<Artifact> ccLibraries) {
    ObjcConfiguration objcConfiguration = ObjcRuleClasses.objcConfiguration(ruleContext);

    CustomCommandLine.Builder commandLine = CustomCommandLine.builder();

    if (objcProvider.is(USES_CPP)) {
      commandLine
          .addPath(CLANG_PLUSPLUS)
          .add("-stdlib=libc++");
    } else {
      commandLine.addPath(CLANG);
    }
    commandLine
        .add(IosSdkCommands.commonLinkAndCompileFlagsForClang(objcProvider, objcConfiguration))
        .add("-Xlinker").add("-objc_abi_version")
        .add("-Xlinker").add("2")
        .add("-fobjc-link-runtime")
        .add(IosSdkCommands.DEFAULT_LINKER_FLAGS)
        .addBeforeEach("-framework", frameworkNames(objcProvider))
        .addBeforeEach("-weak_framework", SdkFramework.names(objcProvider.get(WEAK_SDK_FRAMEWORK)))
        .addExecPath("-o", linkedBinary)
        .addExecPaths(objcProvider.get(LIBRARY))
        .addExecPaths(objcProvider.get(IMPORTED_LIBRARY))
        .addExecPaths(ccLibraries)
        .add(dylibPaths(objcProvider))
        .addBeforeEach("-force_load", Artifact.toExecPaths(objcProvider.get(FORCE_LOAD_LIBRARY)))
        .add(extraLinkArgs)
        .build();

    if (ruleContext.getConfiguration().isCodeCoverageEnabled()) {
      commandLine.add(LINKER_COVERAGE_FLAGS);
    }

    // Call to dsymutil for debug symbol generation must happen in the link action.
    // All debug symbol information is encoded in object files inside archive files. To generate
    // the debug symbol bundle, dsymutil will look inside the linked binary for the encoded
    // absolute paths to archive files, which are only valid in the link action.
    if (dsymBundle.isPresent()) {
      commandLine
          .add("&&")
          .addPath(DSYMUTIL)
          .add(linkedBinary.getExecPathString())
          .addExecPath("-o", dsymBundle.get());
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

  private Iterable<String> dylibPaths(ObjcProvider objcProvider) {
    ObjcConfiguration objcConfiguration = ObjcRuleClasses.objcConfiguration(ruleContext);
    ImmutableList.Builder<String> args = new ImmutableList.Builder<>();
    for (String dylib : objcProvider.get(SDK_DYLIB)) {
      args.add(String.format(
          "%s/usr/lib/%s.dylib", IosSdkCommands.sdkDir(objcConfiguration), dylib));
    }
    return args.build();
  }

  /**
   * All framework names to pass to the linker using {@code -framework} flags. For a framework in
   * the directory foo/bar.framework, the name is "bar". Each framework is found without using the
   * full path by means of the framework search paths. The search paths are added by
   * {@link IosSdkCommands#commonLinkAndCompileFlagsForClang(ObjcProvider, ObjcConfiguration)}).
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
  CompilationSupport registerJ2ObjcCompileAndArchiveActions(ObjcProvider objcProvider) {
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
      registerCompileAndArchiveActions(compilationArtifact, intermediateArtifacts, objcProvider,
          ruleContext.getConfiguration().isCodeCoverageEnabled());
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
      if (j2ObjcSource.getSourceType() == SourceType.JAVA) {
        Iterable<Artifact> sourceArtifacts = j2ObjcSource.getObjcSrcs();
        Iterable<Artifact> prunedSourceArtifacts =
            j2ObjcSource.toPrunedSource(ruleContext).getObjcSrcs();
        PathFragment objcFilePath = j2ObjcSource.getObjcFilePath();
        ruleContext.registerAction(new SpawnAction.Builder()
            .setMnemonic("DummyPruner")
            .setExecutable(pruner)
            .addInput(pruner)
            .addInputs(sourceArtifacts)
            .addTransitiveInputs(j2ObjcDependencyMappingFiles)
            .addTransitiveInputs(j2ObjcHeaderMappingFiles)
            .setCommandLine(CustomCommandLine.builder()
                .addJoinExecPaths("--input_files", ",", sourceArtifacts)
                .addJoinExecPaths("--output_files", ",", prunedSourceArtifacts)
                .addJoinExecPaths("--dependency_mapping_files", ",", j2ObjcDependencyMappingFiles)
                .addJoinExecPaths("--header_mapping_files", ",", j2ObjcHeaderMappingFiles)
                .add("--entry_classes").add(Joiner.on(",").join(entryClasses))
                .add("--objc_file_path").add(objcFilePath.getPathString())
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
    xcodeProviderBuilder
        .addHeaders(attributes.hdrs())
        .addUserHeaderSearchPaths(ObjcCommon.userHeaderSearchPaths(ruleContext.getConfiguration()))
        .addHeaderSearchPaths("$(WORKSPACE_ROOT)", attributes.headerSearchPaths())
        .addHeaderSearchPaths("$(SDKROOT)/usr/include", attributes.sdkIncludes())
        .addCompilationModeCopts(objcConfiguration.getCoptsForCompilationMode())
        .addCopts(objcConfiguration.getCopts())
        .addCopts(attributes.copts())
        .addCopts(attributes.optionsCopts());
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

    return this;
  }

  private CompilationSupport registerDsymActions() {
    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);
    Artifact dsymBundle = intermediateArtifacts.dsymBundle();
    Artifact debugSymbolFile = intermediateArtifacts.dsymSymbol();
    ruleContext.registerAction(new SpawnAction.Builder()
        .setMnemonic("UnzipDsym")
        .setProgressMessage("Unzipping dSYM file: " + ruleContext.getLabel())
        .setExecutable(new PathFragment("/usr/bin/unzip"))
        .addInput(dsymBundle)
        .setCommandLine(CustomCommandLine.builder()
            .add(dsymBundle.getExecPathString())
            .add("-d")
            .add(stripSuffix(dsymBundle.getExecPathString(),
                IntermediateArtifacts.TMP_DSYM_BUNDLE_SUFFIX) + ".app.dSYM")
            .build())
        .addOutput(intermediateArtifacts.dsymPlist())
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

  private String stripSuffix(String str, String suffix) {
    // TODO(bazel-team): Throw instead of returning null?
    return str.endsWith(suffix) ? str.substring(0, str.length() - suffix.length()) : null;
  }
}
