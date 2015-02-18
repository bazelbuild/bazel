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
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.ASSET_CATALOG;
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
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCASSETS_DIR;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
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
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.xcode.common.TargetDeviceFamily;
import com.google.devtools.build.xcode.util.Interspersing;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;

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
  static final ImmutableList<String> CLANG_COVERAGE_FLAGS = ImmutableList.<String>of(
      "-fprofile-arcs", "-ftest-coverage", "-fprofile-dir=./coverage_output");

  // TODO(bazel-team): Reference a rule target rather than a jar file when Darwin runfiles work
  // better.
  private static SpawnAction.Builder spawnJavaOnDarwinActionBuilder(Artifact deployJarArtifact) {
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
        .add(IosSdkCommands.compileArgsForClang(objcConfiguration))
        .add(IosSdkCommands.commonLinkAndCompileArgsForClang(
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

  /**
   * Creates actions to convert all files specified by the strings attribute into binary format.
   */
  private static Iterable<Action> convertStringsActions(
      ActionConstructionContext context,
      ObjcRuleClasses.Tools baseTools,
      StringsFiles stringsFiles) {
    ImmutableList.Builder<Action> result = new ImmutableList.Builder<>();
    for (CompiledResourceFile stringsFile : stringsFiles) {
      final Artifact original = stringsFile.getOriginal();
      final Artifact bundled = stringsFile.getBundled().getBundled();
      result.add(new SpawnAction.Builder()
          .setMnemonic("ConvertStringsPlist")
          .setExecutable(baseTools.plmerge())
          .setCommandLine(new CommandLine() {
            @Override
            public Iterable<String> arguments() {
              return ImmutableList.of("--source_file", original.getExecPathString(),
                  "--out_file", bundled.getExecPathString());
            }
          })
          .addInput(original)
          .addOutput(bundled)
          .build(context));
    }
    return result.build();
  }

  private Action[] ibtoolzipAction(ObjcRuleClasses.Tools baseTools, String mnemonic, Artifact input,
      Artifact zipOutput, String archiveRoot) {
    return spawnJavaOnDarwinActionBuilder(baseTools.actooloribtoolzipDeployJar())
        .setMnemonic(mnemonic)
        .setCommandLine(new CustomCommandLine.Builder()
            // The next three arguments are positional, i.e. they don't have flags before them.
            .addPath(zipOutput.getExecPath())
            .add(archiveRoot)
            .addPath(IBTOOL)

            .add("--minimum-deployment-target").add(objcConfiguration.getMinimumOs())
            .addPath(input.getExecPath())
            .build())
        .addOutput(zipOutput)
        .addInput(input)
        .build(context);
  }

  /**
   * Creates actions to convert all files specified by the xibs attribute into nib format.
   */
  private Iterable<Action> convertXibsActions(ObjcRuleClasses.Tools baseTools, XibFiles xibFiles) {
    ImmutableList.Builder<Action> result = new ImmutableList.Builder<>();
    for (Artifact original : xibFiles) {
      Artifact zipOutput = intermediateArtifacts.compiledXibFileZip(original);
      String archiveRoot = BundleableFile.bundlePath(
          FileSystemUtils.replaceExtension(original.getExecPath(), ".nib"));
      result.add(ibtoolzipAction(baseTools, "XibCompile", original, zipOutput, archiveRoot));
    }
    return result.build();
  }

  /**
   * Outputs of an {@code actool} action besides the zip file.
   */
  static final class ExtraActoolOutputs extends IterableWrapper<Artifact> {
    ExtraActoolOutputs(Artifact... extraActoolOutputs) {
      super(extraActoolOutputs);
    }
  }

  static final class ExtraActoolArgs extends IterableWrapper<String> {
    ExtraActoolArgs(Iterable<String> args) {
      super(args);
    }

    ExtraActoolArgs(String... args) {
      super(args);
    }
  }

  void registerActoolzipAction(
      ObjcRuleClasses.Tools tools,
      ObjcProvider provider,
      Artifact zipOutput,
      ExtraActoolOutputs extraActoolOutputs,
      ExtraActoolArgs extraActoolArgs,
      Set<TargetDeviceFamily> families) {
    // TODO(bazel-team): Do not use the deploy jar explicitly here. There is currently a bug where
    // we cannot .setExecutable({java_binary target}) and set REQUIRES_DARWIN in the execution info.
    // Note that below we set the archive root to the empty string. This means that the generated
    // zip file will be rooted at the bundle root, and we have to prepend the bundle root to each
    // entry when merging it with the final .ipa file.
    register(spawnJavaOnDarwinActionBuilder(tools.actooloribtoolzipDeployJar())
        .setMnemonic("AssetCatalogCompile")
        .addTransitiveInputs(provider.get(ASSET_CATALOG))
        .addOutput(zipOutput)
        .addOutputs(extraActoolOutputs)
        .setCommandLine(actoolzipCommandLine(
            objcConfiguration,
            provider,
            zipOutput,
            extraActoolArgs,
            ImmutableSet.copyOf(families)))
        .build(context));
  }

  private static CommandLine actoolzipCommandLine(
      final ObjcConfiguration objcConfiguration,
      final ObjcProvider provider,
      final Artifact zipOutput,
      final ExtraActoolArgs extraActoolArgs,
      final ImmutableSet<TargetDeviceFamily> families) {
    return new CommandLine() {
      @Override
      public Iterable<String> arguments() {
        ImmutableList.Builder<String> args = new ImmutableList.Builder<String>()
            // The next three arguments are positional, i.e. they don't have flags before them.
            .add(zipOutput.getExecPathString())
            .add("") // archive root
            .add(IosSdkCommands.ACTOOL_PATH)
            .add("--platform")
            .add(objcConfiguration.getPlatform().getLowerCaseNameInPlist())
            .add("--minimum-deployment-target").add(objcConfiguration.getMinimumOs());
        for (TargetDeviceFamily targetDeviceFamily : families) {
          args.add("--target-device").add(targetDeviceFamily.name().toLowerCase(Locale.US));
        }
        return args
            .addAll(PathFragment.safePathStrings(provider.get(XCASSETS_DIR)))
            .addAll(extraActoolArgs)
            .build();
      }
    };
  }

  void registerIbtoolzipAction(ObjcRuleClasses.Tools tools, Artifact input, Artifact outputZip) {
    String archiveRoot = BundleableFile.bundlePath(input.getExecPath()) + "c";
    register(ibtoolzipAction(tools, "StoryboardCompile", input, outputZip, archiveRoot));
  }

  @VisibleForTesting
  static Iterable<String> commonMomczipArguments(ObjcConfiguration configuration) {
    return ImmutableList.of(
        "-XD_MOMC_SDKROOT=" + IosSdkCommands.sdkDir(configuration),
        "-XD_MOMC_IOS_TARGET_VERSION=" + configuration.getMinimumOs(),
        "-MOMC_PLATFORMS", configuration.getPlatform().getLowerCaseNameInPlist(),
        "-XD_MOMC_TARGET_VERSION=10.6");
  }

  private static Iterable<Action> momczipActions(ActionConstructionContext context,
      ObjcRuleClasses.Tools baseTools, final ObjcConfiguration objcConfiguration,
      Iterable<Xcdatamodel> datamodels) {
    ImmutableList.Builder<Action> result = new ImmutableList.Builder<>();
    for (Xcdatamodel datamodel : datamodels) {
      final Artifact outputZip = datamodel.getOutputZip();
      final String archiveRoot = datamodel.archiveRootForMomczip();
      final String container = datamodel.getContainer().getSafePathString();
      result.add(spawnJavaOnDarwinActionBuilder(baseTools.momczipDeployJar())
          .setMnemonic("MomCompile")
          .addOutput(outputZip)
          .addInputs(datamodel.getInputs())
          .setCommandLine(new CommandLine() {
            @Override
            public Iterable<String> arguments() {
              return new ImmutableList.Builder<String>()
                  .add(outputZip.getExecPathString())
                  .add(archiveRoot)
                  .add(IosSdkCommands.MOMC_PATH)
                  .addAll(commonMomczipArguments(objcConfiguration))
                  .add(container)
                  .build();
            }
          })
          .build(context));
    }
    return result.build();
  }

  private static final String FRAMEWORK_SUFFIX = ".framework";

  /**
   * All framework names to pass to the linker using {@code -framework} flags. For a framework in
   * the directory foo/bar.framework, the name is "bar". Each framework is found without using the
   * full path by means of the framework search paths. The search paths are added by
   * {@link IosSdkCommands#commonLinkAndCompileArgsForClang(ObjcProvider, ObjcConfiguration)}).
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
     * Returns an ExtraLinkArgs with the parameter prepended to this instance's contents. This
     * function does not modify this instance.
     */
    @CheckReturnValue
    public ExtraLinkArgs prependedWith(Iterable<String> extraLinkArgs) {
      return new ExtraLinkArgs(Iterables.concat(extraLinkArgs, this));
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
    ExtraLinkInputs(Iterable<Artifact> inputs) {
      super(inputs);
    }

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
          .addAll(IosSdkCommands.commonLinkAndCompileArgsForClang(objcProvider, objcConfiguration))
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

  static final class StringsFiles extends IterableWrapper<CompiledResourceFile> {
    StringsFiles(Iterable<CompiledResourceFile> files) {
      super(files);
    }
  }

  /**
   * Registers actions for resource conversion that are needed by all rules that inherit from
   * {@link ObjcBase}.
   */
  void registerResourceActions(ObjcRuleClasses.Tools baseTools, StringsFiles stringsFiles,
      XibFiles xibFiles, Iterable<Xcdatamodel> datamodels) {
    registerAll(convertStringsActions(context, baseTools, stringsFiles));
    registerAll(convertXibsActions(baseTools, xibFiles));
    registerAll(momczipActions(context, baseTools, objcConfiguration, datamodels));
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
