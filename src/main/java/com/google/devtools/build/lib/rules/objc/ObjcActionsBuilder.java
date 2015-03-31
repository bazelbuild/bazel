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

import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.io.ByteSource;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionRegistry;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.BinaryFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.util.LazyString;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import javax.annotation.CheckReturnValue;

/**
 * Object that creates actions used by Objective-C rules.
 */
final class ObjcActionsBuilder {
  private final ActionConstructionContext context;
  private final IntermediateArtifacts intermediateArtifacts;
  private final ObjcConfiguration objcConfiguration;
  private final BuildConfiguration buildConfiguration;
  private final ActionRegistry actionRegistry;

  ObjcActionsBuilder(ActionConstructionContext context, IntermediateArtifacts intermediateArtifacts,
      ObjcConfiguration objcConfiguration, BuildConfiguration buildConfiguration,
      ActionRegistry actionRegistry) {
    this.context = Preconditions.checkNotNull(context);
    this.intermediateArtifacts = Preconditions.checkNotNull(intermediateArtifacts);
    this.objcConfiguration = Preconditions.checkNotNull(objcConfiguration);
    this.buildConfiguration = Preconditions.checkNotNull(buildConfiguration);
    this.actionRegistry = Preconditions.checkNotNull(actionRegistry);
  }

  /**
   * Creates a new spawn action builder that requires a darwin architecture to run.
   */
  // TODO(bazel-team): Use everywhere we currently set the execution info manually.
  static SpawnAction.Builder spawnOnDarwinActionBuilder() {
    return new SpawnAction.Builder()
        .setExecutionInfo(ImmutableMap.of(ExecutionRequirements.REQUIRES_DARWIN, ""));
  }

  static final PathFragment JAVA = new PathFragment("/usr/bin/java");
  static final PathFragment CLANG = new PathFragment(BIN_DIR + "/clang");
  static final PathFragment CLANG_PLUSPLUS = new PathFragment(BIN_DIR + "/clang++");
  static final PathFragment LIBTOOL = new PathFragment(BIN_DIR + "/libtool");
  static final PathFragment IBTOOL = new PathFragment(IosSdkCommands.IBTOOL_PATH);
  static final PathFragment DSYMUTIL = new PathFragment(BIN_DIR + "/dsymutil");
  static final PathFragment LIPO = new PathFragment(BIN_DIR + "/lipo");
  static final ImmutableList<String> CLANG_COVERAGE_FLAGS =
      ImmutableList.of("-fprofile-arcs", "-ftest-coverage", "-fprofile-dir=./coverage_output");

  // TODO(bazel-team): Reference a rule target rather than a jar file when Darwin runfiles work
  // better.
  static SpawnAction.Builder spawnJavaOnDarwinActionBuilder(Artifact deployJarArtifact) {
    return spawnOnDarwinActionBuilder()
        .setExecutable(JAVA)
        .addExecutableArguments("-jar", deployJarArtifact.getExecPathString())
        .addInput(deployJarArtifact);
  }

  private void registerCompileAction(
      Artifact sourceFile,
      Artifact objFile,
      Optional<Artifact> pchFile,
      ObjcProvider objcProvider,
      Iterable<String> otherFlags,
      OptionsProvider optionsProvider,
      boolean isCodeCoverageEnabled) {
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
        .addBeforeEachPath("-iquote", ObjcCommon.userHeaderSearchPaths(buildConfiguration))
        .addBeforeEachExecPath("-include", pchFile.asSet())
        .addBeforeEachPath("-I", objcProvider.get(INCLUDE))
        .add(otherFlags)
        .addFormatEach("-D%s", objcProvider.get(DEFINE))
        .add(coverageFlags.build())
        .add(objcConfiguration.getCopts())
        .add(optionsProvider.getCopts())
        .addExecPath("-c", sourceFile)
        .addExecPath("-o", objFile);

    register(spawnOnDarwinActionBuilder()
        .setMnemonic("ObjcCompile")
        .setExecutable(CLANG)
        .setCommandLine(commandLine.build())
        .addInput(sourceFile)
        .addOutput(objFile)
        .addOutputs(gcnoFiles.build())
        .addTransitiveInputs(objcProvider.get(HEADER))
        .addTransitiveInputs(objcProvider.get(FRAMEWORK_FILE))
        .addInputs(pchFile.asSet())
        .build(context));
  }

  private static final ImmutableList<String> ARC_ARGS = ImmutableList.of("-fobjc-arc");
  private static final ImmutableList<String> NON_ARC_ARGS = ImmutableList.of("-fno-objc-arc");

  /**
   * Creates actions to compile each source file individually, and link all the compiled object
   * files into a single archive library.
   */
  void registerCompileAndArchiveActions(CompilationArtifacts compilationArtifacts,
      ObjcProvider objcProvider, OptionsProvider optionsProvider, boolean isCodeCoverageEnabled) {
    ImmutableList.Builder<Artifact> objFiles = new ImmutableList.Builder<>();
    for (Artifact sourceFile : compilationArtifacts.getSrcs()) {
      Artifact objFile = intermediateArtifacts.objFile(sourceFile);
      objFiles.add(objFile);
      registerCompileAction(sourceFile, objFile, compilationArtifacts.getPchFile(),
          objcProvider, ARC_ARGS, optionsProvider, isCodeCoverageEnabled);
    }
    for (Artifact nonArcSourceFile : compilationArtifacts.getNonArcSrcs()) {
      Artifact objFile = intermediateArtifacts.objFile(nonArcSourceFile);
      objFiles.add(objFile);
      registerCompileAction(nonArcSourceFile, objFile, compilationArtifacts.getPchFile(),
          objcProvider, NON_ARC_ARGS, optionsProvider, isCodeCoverageEnabled);
    }
    for (Artifact archive : compilationArtifacts.getArchive().asSet()) {
      registerAll(archiveActions(context, objFiles.build(), archive, objcConfiguration,
          intermediateArtifacts.objList()));
    }
  }

  private static Iterable<Action> archiveActions(
      ActionConstructionContext context,
      final Iterable<Artifact> objFiles,
      final Artifact archive,
      final ObjcConfiguration objcConfiguration,
      final Artifact objList) {

    ImmutableList.Builder<Action> actions = new ImmutableList.Builder<>();

    actions.add(new FileWriteAction(
        context.getActionOwner(), objList, joinExecPaths(objFiles), /*makeExecutable=*/ false));

    actions.add(spawnOnDarwinActionBuilder()
        .setMnemonic("ObjcLink")
        .setExecutable(LIBTOOL)
        .setCommandLine(new CommandLine() {
            @Override
            public Iterable<String> arguments() {
              return new ImmutableList.Builder<String>()
                  .add("-static")
                  .add("-filelist").add(objList.getExecPathString())
                  .add("-arch_only").add(objcConfiguration.getIosCpu())
                  .add("-syslibroot").add(IosSdkCommands.sdkDir(objcConfiguration))
                  .add("-o").add(archive.getExecPathString())
                  .build();
            }
          })
        .addInputs(objFiles)
        .addInput(objList)
        .addOutput(archive)
        .build(context));

    return actions.build();
  }

  private void register(Action... action) {
    actionRegistry.registerAction(action);
  }

  private void registerAll(Iterable<? extends Action> actions) {
    for (Action action : actions) {
      actionRegistry.registerAction(action);
    }
  }

  private static ByteSource xcodegenControlFileBytes(
      final Artifact pbxproj, final XcodeProvider.Project project, final String minimumOs) {
    return new ByteSource() {
      @Override
      public InputStream openStream() {
        return XcodeGenProtos.Control.newBuilder()
            .setPbxproj(pbxproj.getExecPathString())
            .addAllTarget(project.targets())
            .addBuildSetting(XcodeGenProtos.XcodeprojBuildSetting.newBuilder()
                .setName("IPHONEOS_DEPLOYMENT_TARGET")
                .setValue(minimumOs)
                .build())
            .build()
            .toByteString()
            .newInput();
      }
    };
  }

  /**
   * Generates actions needed to create an Xcode project file.
   */
  void registerXcodegenActions(
      ObjcRuleClasses.Tools baseTools, Artifact pbxproj, XcodeProvider.Project project) {
    Artifact controlFile = intermediateArtifacts.pbxprojControlArtifact();
    register(new BinaryFileWriteAction(
        context.getActionOwner(),
        controlFile,
        xcodegenControlFileBytes(pbxproj, project, objcConfiguration.getMinimumOs()),
        /*makeExecutable=*/false));
    register(new SpawnAction.Builder()
        .setMnemonic("GenerateXcodeproj")
        .setExecutable(baseTools.xcodegen())
        .addArgument("--control")
        .addInputArgument(controlFile)
        .addOutput(pbxproj)
        .addTransitiveInputs(project.getInputsToXcodegen())
        .build(context));
  }

  static final class ExtraActoolArgs extends IterableWrapper<String> {
    ExtraActoolArgs(Iterable<String> args) {
      super(args);
    }

    ExtraActoolArgs(String... args) {
      super(args);
    }
  }


  private static final String FRAMEWORK_SUFFIX = ".framework";

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
  private static Iterable<String> frameworkNames(ObjcProvider provider) {
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

  static final class ExtraLinkArgs extends IterableWrapper<String> {
    ExtraLinkArgs(Iterable<String> args) {
      super(args);
    }

    ExtraLinkArgs(String... args) {
      super(args);
    }

    /*
     * Returns an ExtraLinkArgs with the parameter appended to this instance's contents. This
     * function does not modify this instance.
     */
    @CheckReturnValue
    public ExtraLinkArgs appendedWith(Iterable<String> extraLinkArgs) {
      return new ExtraLinkArgs(Iterables.concat(this, extraLinkArgs));
    }
  }

  static final class ExtraLinkInputs extends IterableWrapper<Artifact> {
    ExtraLinkInputs(Artifact... inputs) {
      super(inputs);
    }
  }

  private static final class LinkCommandLine extends CommandLine {
    private static final Joiner commandJoiner = Joiner.on(' ');
    private final ObjcProvider objcProvider;
    private final ObjcConfiguration objcConfiguration;
    private final Artifact linkedBinary;
    private final Optional<Artifact> dsymBundle;
    private final ExtraLinkArgs extraLinkArgs;

    LinkCommandLine(ObjcConfiguration objcConfiguration, ExtraLinkArgs extraLinkArgs,
        ObjcProvider objcProvider, Artifact linkedBinary, Optional<Artifact> dsymBundle) {
      this.objcConfiguration = Preconditions.checkNotNull(objcConfiguration);
      this.extraLinkArgs = Preconditions.checkNotNull(extraLinkArgs);
      this.objcProvider = Preconditions.checkNotNull(objcProvider);
      this.linkedBinary = Preconditions.checkNotNull(linkedBinary);
      this.dsymBundle = Preconditions.checkNotNull(dsymBundle);
    }

    Iterable<String> dylibPaths() {
      ImmutableList.Builder<String> args = new ImmutableList.Builder<>();
      for (String dylib : objcProvider.get(SDK_DYLIB)) {
        args.add(String.format(
            "%s/usr/lib/%s.dylib", IosSdkCommands.sdkDir(objcConfiguration), dylib));
      }
      return args.build();
    }

    @Override
    public Iterable<String> arguments() {
      StringBuilder argumentStringBuilder = new StringBuilder();

      Iterable<String> archiveExecPaths = Artifact.toExecPaths(
          Iterables.concat(objcProvider.get(LIBRARY), objcProvider.get(IMPORTED_LIBRARY)));
      commandJoiner.appendTo(argumentStringBuilder, new ImmutableList.Builder<String>()
          .add(objcProvider.is(USES_CPP) ? CLANG_PLUSPLUS.toString() : CLANG.toString())
          .addAll(objcProvider.is(USES_CPP)
              ? ImmutableList.of("-stdlib=libc++") : ImmutableList.<String>of())
          .addAll(IosSdkCommands.commonLinkAndCompileFlagsForClang(objcProvider, objcConfiguration))
          .add("-Xlinker", "-objc_abi_version")
          .add("-Xlinker", "2")
          .add("-fobjc-link-runtime")
          .addAll(IosSdkCommands.DEFAULT_LINKER_FLAGS)
          .addAll(Interspersing.beforeEach("-framework", frameworkNames(objcProvider)))
          .addAll(Interspersing.beforeEach(
              "-weak_framework", SdkFramework.names(objcProvider.get(WEAK_SDK_FRAMEWORK))))
          .add("-o", linkedBinary.getExecPathString())
          .addAll(archiveExecPaths)
          .addAll(dylibPaths())
          .addAll(extraLinkArgs)
          .build());

      // Call to dsymutil for debug symbol generation must happen in the link action.
      // All debug symbol information is encoded in object files inside archive files. To generate
      // the debug symbol bundle, dsymutil will look inside the linked binary for the encoded
      // absolute paths to archive files, which are only valid in the link action.
      for (Artifact justDsymBundle : dsymBundle.asSet()) {
        argumentStringBuilder.append(" ");
        commandJoiner.appendTo(argumentStringBuilder, new ImmutableList.Builder<String>()
            .add("&&")
            .add(DSYMUTIL.toString())
            .add(linkedBinary.getExecPathString())
            .add("-o").add(justDsymBundle.getExecPathString())
            .build());
      }

      return ImmutableList.of(argumentStringBuilder.toString());
    }
  }

  /**
   * Generates an action to link a binary.
   */
  void registerLinkAction(Artifact linkedBinary, ObjcProvider objcProvider,
      ExtraLinkArgs extraLinkArgs, ExtraLinkInputs extraLinkInputs, Optional<Artifact> dsymBundle) {
    extraLinkArgs = new ExtraLinkArgs(Iterables.concat(
        Interspersing.beforeEach(
            "-force_load", Artifact.toExecPaths(objcProvider.get(FORCE_LOAD_LIBRARY))),
        extraLinkArgs));
    register(spawnOnDarwinActionBuilder()
        .setMnemonic("ObjcLink")
        .setShellCommand(ImmutableList.of("/bin/bash", "-c"))
        .setCommandLine(
            new LinkCommandLine(objcConfiguration, extraLinkArgs, objcProvider, linkedBinary,
                dsymBundle))
        .addOutput(linkedBinary)
        .addOutputs(dsymBundle.asSet())
        .addTransitiveInputs(objcProvider.get(LIBRARY))
        .addTransitiveInputs(objcProvider.get(IMPORTED_LIBRARY))
        .addTransitiveInputs(objcProvider.get(FRAMEWORK_FILE))
        .addInputs(extraLinkInputs)
        .build(context));
  }

  static LazyString joinExecPaths(final Iterable<Artifact> artifacts) {
    return new LazyString() {
      @Override
      public String toString() {
        return Artifact.joinExecPaths("\n", artifacts);
      }
    };
  }
}
