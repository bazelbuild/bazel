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
import static com.google.devtools.build.lib.rules.objc.IosSdkCommands.MINIMUM_OS_VERSION;
import static com.google.devtools.build.lib.rules.objc.IosSdkCommands.TARGET_DEVICE_FAMILIES;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.ASSET_CATALOG;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FRAMEWORK_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FRAMEWORK_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.Flag.USES_CPP;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.HEADER;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.INCLUDE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_DYLIB;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCASSETS_DIR;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.io.ByteSource;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionRegistry;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.util.LazyString;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.actions.ActionConstructionContext;
import com.google.devtools.build.lib.view.actions.BinaryFileWriteAction;
import com.google.devtools.build.lib.view.actions.CommandLine;
import com.google.devtools.build.lib.view.actions.FileWriteAction;
import com.google.devtools.build.lib.view.actions.SpawnAction;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.xcode.common.TargetDeviceFamily;
import com.google.devtools.build.xcode.util.Interspersing;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

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

  private static SpawnAction.Builder spawnOnDarwinActionBuilder(ActionConstructionContext context) {
    return new SpawnAction.Builder(context)
        .setExecutionInfo(ImmutableMap.of(ExecutionRequirements.REQUIRES_DARWIN, ""));
  }

  static final PathFragment JAVA = new PathFragment("/usr/bin/java");
  static final PathFragment CLANG = new PathFragment(BIN_DIR + "/clang");
  static final PathFragment CLANG_PLUSPLUS = new PathFragment(BIN_DIR + "/clang++");
  static final PathFragment LIBTOOL = new PathFragment(BIN_DIR + "/libtool");
  static final PathFragment IBTOOL = new PathFragment(IosSdkCommands.IBTOOL_PATH);
  static final PathFragment DSYMUTIL = new PathFragment(BIN_DIR + "/dsymutil");

  // TODO(bazel-team): Reference a rule target rather than a jar file when Darwin runfiles work
  // better.
  private static SpawnAction.Builder spawnJavaOnDarwinActionBuilder(
      ActionConstructionContext context, Artifact deployJarArtifact) {
    return spawnOnDarwinActionBuilder(context)
        .setExecutable(JAVA)
        .addExecutableArguments("-jar", deployJarArtifact.getExecPathString())
        .addInput(deployJarArtifact);
  }

  private static Action compileAction(
      ActionConstructionContext context,
      final Artifact sourceFile,
      final Artifact objFile,
      final Optional<Artifact> pchFile,
      final ObjcProvider objcProvider,
      final Iterable<String> otherFlags,
      final OptionsProvider optionsProvider,
      final ObjcConfiguration objcConfiguration,
      final BuildConfiguration buildConfiguration) {
    return spawnOnDarwinActionBuilder(context)
        .setMnemonic("Compile")
        .setExecutable(CLANG)
        .setCommandLine(new CommandLine() {
          @Override
          public Iterable<String> arguments() {
            return new ImmutableList.Builder<String>()
                .addAll(IosSdkCommands.compileArgsForClang(objcConfiguration))
                .addAll(IosSdkCommands.commonLinkAndCompileArgsForClang(
                    objcProvider, objcConfiguration))
                .addAll(Interspersing.beforeEach(
                    "-iquote",
                    PathFragment.safePathStrings(
                        ObjcCommon.userHeaderSearchPaths(buildConfiguration))))
                .addAll(Interspersing.beforeEach("-include", Artifact.asExecPaths(pchFile.asSet())))
                .addAll(Interspersing.beforeEach(
                    "-I", PathFragment.safePathStrings(objcProvider.get(INCLUDE))))
                .addAll(otherFlags)
                .addAll(optionsProvider.getCopts())
                .add("-c").add(sourceFile.getExecPathString())
                .add("-o").add(objFile.getExecPathString())
                .build();
          }
        })
        .addInput(sourceFile)
        .addOutput(objFile)
        .addTransitiveInputs(objcProvider.get(HEADER))
        .addTransitiveInputs(objcProvider.get(FRAMEWORK_FILE))
        .addInputs(pchFile.asSet())
        .build();
  }

  private static final ImmutableList<String> ARC_ARGS = ImmutableList.of("-fobjc-arc");
  private static final ImmutableList<String> NON_ARC_ARGS = ImmutableList.of("-fno-objc-arc");

  /**
   * Creates actions to compile each source file individually, and link all the compiled object
   * files into a single archive library.
   */
  void registerCompileAndArchiveActions(CompilationArtifacts compilationArtifacts,
      ObjcProvider objcProvider, OptionsProvider optionsProvider) {
    ImmutableList.Builder<Artifact> objFiles = new ImmutableList.Builder<>();
    for (Artifact sourceFile : compilationArtifacts.getSrcs()) {
      Artifact objFile = intermediateArtifacts.objFile(sourceFile);
      objFiles.add(objFile);
      register(
          compileAction(context, sourceFile, objFile,
              compilationArtifacts.getPchFile(), objcProvider, ARC_ARGS, optionsProvider,
              objcConfiguration, buildConfiguration));
    }
    for (Artifact nonArcSourceFile : compilationArtifacts.getNonArcSrcs()) {
      Artifact objFile = intermediateArtifacts.objFile(nonArcSourceFile);
      objFiles.add(objFile);
      register(
          compileAction(context, nonArcSourceFile, objFile,
              compilationArtifacts.getPchFile(), objcProvider, NON_ARC_ARGS, optionsProvider,
              objcConfiguration, buildConfiguration));
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
    LazyString objListContent = new LazyString() {
      @Override
      public String toString() {
        return Artifact.joinExecPaths("\n", objFiles);
      }
    };

    ImmutableList.Builder<Action> actions = new ImmutableList.Builder<>();

    actions.add(new FileWriteAction(
        context.getActionOwner(), objList, objListContent, /*makeExecutable=*/ false));

    actions.add(spawnOnDarwinActionBuilder(context)
        .setMnemonic("Link")
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
        .build());

    return actions.build();
  }

  private void register(Action action) {
    actionRegistry.registerAction(action);
  }

  private void registerAll(Iterable<? extends Action> actions) {
    for (Action action : actions) {
      actionRegistry.registerAction(action);
    }
  }

  private static ByteSource xcodegenControlFileBytes(
      final Artifact pbxproj, final XcodeProvider xcodeProvider) {
    return new ByteSource() {
      @Override
      public InputStream openStream() {
        return XcodeGenProtos.Control.newBuilder()
            .setPbxproj(pbxproj.getExecPathString())
            .addAllTarget(xcodeProvider.targets())
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
      ObjcRuleClasses.Tools baseTools, Artifact pbxproj, XcodeProvider xcodeProvider) {
    Artifact controlFile = intermediateArtifacts.pbxprojControlArtifact();
    register(new BinaryFileWriteAction(
        context.getActionOwner(),
        controlFile,
        xcodegenControlFileBytes(pbxproj, xcodeProvider),
        /*makeExecutable=*/false));
    register(new SpawnAction.Builder(context)
        .setMnemonic("Generate project")
        .setExecutable(baseTools.xcodegen())
        .addArgument("--control")
        .addInputArgument(controlFile)
        .addOutput(pbxproj)
        .build());
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
      result.add(new SpawnAction.Builder(context)
          .setMnemonic("Convert plist to binary")
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
          .build());
    }
    return result.build();
  }

  /**
   * Creates actions to convert all files specified by the xibs attribute into nib format.
   */
  private static Iterable<Action> convertXibsActions(
      ActionConstructionContext context, XibFiles xibFiles) {
    ImmutableList.Builder<Action> result = new ImmutableList.Builder<>();
    for (CompiledResourceFile xibFile : xibFiles) {
      final Artifact bundled = xibFile.getBundled().getBundled();
      final Artifact original = xibFile.getOriginal();
      result.add(spawnOnDarwinActionBuilder(context)
          .setMnemonic("Compile xib")
          .setExecutable(IBTOOL)
          .setCommandLine(new CommandLine() {
            @Override
            public Iterable<String> arguments() {
              return ImmutableList.of(
                  "--minimum-deployment-target", MINIMUM_OS_VERSION,
                  "--compile", bundled.getExecPathString(),
                  original.getExecPathString());
            }
          })
          .addOutput(bundled)
          .addInput(original)
          .build());
    }
    return result.build();
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
      Artifact actoolzipOutput,
      ExtraActoolArgs extraActoolArgs) {
    // TODO(bazel-team): Do not use the deploy jar explicitly here. There is currently a bug where
    // we cannot .setExecutable({java_binary target}) and set REQUIRES_DARWIN in the execution info.
    // Note that below we set the archive root to the empty string. This means that the generated
    // zip file will be rooted at the bundle root, and we have to prepend the bundle root to each
    // entry when merging it with the final .ipa file.
    register(spawnJavaOnDarwinActionBuilder(context, tools.actooloribtoolzipDeployJar())
        .setMnemonic("Compile asset catalogs")
        .addTransitiveInputs(provider.get(ASSET_CATALOG))
        .addOutput(actoolzipOutput)
        .setCommandLine(actoolzipCommandLine(
            objcConfiguration,
            provider,
            actoolzipOutput,
            extraActoolArgs))
        .build());
  }

  private static CommandLine actoolzipCommandLine(
      final ObjcConfiguration objcConfiguration,
      final ObjcProvider provider,
      final Artifact output,
      final ExtraActoolArgs extraActoolArgs) {
    return new CommandLine() {
      @Override
      public Iterable<String> arguments() {
        ImmutableList.Builder<String> args = new ImmutableList.Builder<String>()
            // The next three arguments are positional, i.e. they don't have flags before them.
            .add(output.getExecPathString())
            .add("") // archive root
            .add(IosSdkCommands.ACTOOL_PATH)
            .add("--platform")
            .add(objcConfiguration.getPlatform().getLowerCaseNameInPlist())
            .add("--minimum-deployment-target").add(MINIMUM_OS_VERSION);
        for (TargetDeviceFamily targetDeviceFamily : TARGET_DEVICE_FAMILIES) {
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
    register(spawnJavaOnDarwinActionBuilder(context, tools.actooloribtoolzipDeployJar())
        .setMnemonic("Compile storyboard")
        .addInput(input)
        .addOutput(outputZip)
        .setCommandLine(Storyboards.ibtoolzipCommandLine(input, outputZip))
        .build());
  }

  @VisibleForTesting
  static Iterable<String> commonMomczipArguments(ObjcConfiguration configuration) {
    return ImmutableList.of(
        "-XD_MOMC_SDKROOT=" + IosSdkCommands.sdkDir(configuration),
        "-XD_MOMC_IOS_TARGET_VERSION=" + IosSdkCommands.MINIMUM_OS_VERSION,
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
      result.add(spawnJavaOnDarwinActionBuilder(context, baseTools.momczipDeployJar())
          .addOutput(outputZip)
          .addInputs(datamodel.getInputs())
          .setCommandLine(new CommandLine() {
            @Override
            public Iterable<String> arguments() {
              return new ImmutableList.Builder<String>()
                  .add(outputZip.getExecPathString())
                  .add(archiveRoot)
                  .add(IosSdkCommands.momcPath(objcConfiguration))
                  .addAll(commonMomczipArguments(objcConfiguration))
                  .add(container)
                  .build();
            }
          })
          .build());
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
  }

  private static final class LinkCommandLine extends CommandLine {
    private final ObjcProvider objcProvider;
    private final ObjcConfiguration objcConfiguration;
    private final Artifact linkedBinary;
    private final ExtraLinkArgs extraLinkArgs;

    LinkCommandLine(ObjcConfiguration objcConfiguration, ExtraLinkArgs extraLinkArgs,
        ObjcProvider objcProvider, Artifact linkedBinary) {
      this.objcConfiguration = Preconditions.checkNotNull(objcConfiguration);
      this.extraLinkArgs = Preconditions.checkNotNull(extraLinkArgs);
      this.objcProvider = Preconditions.checkNotNull(objcProvider);
      this.linkedBinary = Preconditions.checkNotNull(linkedBinary);
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
      return new ImmutableList.Builder<String>()
          .addAll(objcProvider.is(USES_CPP)
              ? ImmutableList.of("-stdlib=libc++") : ImmutableList.<String>of())
          .addAll(IosSdkCommands.commonLinkAndCompileArgsForClang(objcProvider, objcConfiguration))
          .add("-Xlinker", "-objc_abi_version")
          .add("-Xlinker", "2")
          .add("-fobjc-link-runtime")
          .add("-ObjC")
          .addAll(Interspersing.beforeEach("-framework", frameworkNames(objcProvider)))
          .add("-o", linkedBinary.getExecPathString())
          .addAll(Artifact.toExecPaths(objcProvider.get(LIBRARY)))
          .addAll(Artifact.toExecPaths(objcProvider.get(IMPORTED_LIBRARY)))
          .addAll(dylibPaths())
          .addAll(extraLinkArgs)
          .build();
    }
  }

  /**
   * Generates an action to link a binary.
   */
  void registerLinkAction(ActionConstructionContext context, Artifact linkedBinary,
      ObjcProvider objcProvider, ExtraLinkArgs extraLinkArgs) {
    register(spawnOnDarwinActionBuilder(context)
        .setMnemonic("Link")
        .setExecutable(objcProvider.is(USES_CPP) ? CLANG_PLUSPLUS : CLANG)
        .setCommandLine(
            new LinkCommandLine(objcConfiguration, extraLinkArgs, objcProvider, linkedBinary))
        .addOutput(linkedBinary)
        .addTransitiveInputs(objcProvider.get(LIBRARY))
        .addTransitiveInputs(objcProvider.get(IMPORTED_LIBRARY))
        .addTransitiveInputs(objcProvider.get(FRAMEWORK_FILE))
        .build());
  }

  static final class StringsFiles extends IterableWrapper<CompiledResourceFile> {
    StringsFiles(Iterable<CompiledResourceFile> files) {
      super(files);
    }
  }

  static final class XibFiles extends IterableWrapper<CompiledResourceFile> {
    XibFiles(Iterable<CompiledResourceFile> files) {
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
    registerAll(convertXibsActions(context, xibFiles));
    registerAll(momczipActions(context, baseTools, objcConfiguration, datamodels));
  }
}
