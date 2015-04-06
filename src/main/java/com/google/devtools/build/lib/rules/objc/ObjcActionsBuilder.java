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

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FORCE_LOAD_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FRAMEWORK_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FRAMEWORK_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.Flag.USES_CPP;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
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
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
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

  private static final Joiner COMMAND_JOINER = Joiner.on(' ');

  private final ActionConstructionContext context;
  private final IntermediateArtifacts intermediateArtifacts;
  private final ObjcConfiguration objcConfiguration;
  private final ActionRegistry actionRegistry;
  private final PathFragment clang;
  private final PathFragment clangPlusPlus;
  private final PathFragment dsymutil;

  /**
   * @param context {@link ActionConstructionContext} of the rule
   * @param intermediateArtifacts provides intermediate output paths for this rule
   * @param objcConfiguration configuration for this rule
   * @param actionRegistry registry with which to register new actions
   * @param clang path to clang binary to use for compilation. This will soon be deprecated, and
   *     replaced with an Artifact from an objc_toolchain.
   * @param clangPlusPlus path to clang++ binary to use for compilation. This will soon be
   *     deprecated and replaced with an Artifact from an objc_toolchain.
   * @param dsymutil path to the dsymutil binary to use when generating debug symbols. This will
   *     soon be deprecated and replaced with an Aritifact from an objc_toolchain.
   */
  ObjcActionsBuilder(
      ActionConstructionContext context,
      IntermediateArtifacts intermediateArtifacts,
      ObjcConfiguration objcConfiguration,
      ActionRegistry actionRegistry,
      PathFragment clang,
      PathFragment clangPlusPlus,
      PathFragment dsymutil) {
    this.context = Preconditions.checkNotNull(context);
    this.intermediateArtifacts = Preconditions.checkNotNull(intermediateArtifacts);
    this.objcConfiguration = Preconditions.checkNotNull(objcConfiguration);
    this.actionRegistry = Preconditions.checkNotNull(actionRegistry);
    this.clang = clang;
    this.clangPlusPlus = clangPlusPlus;
    this.dsymutil = dsymutil;
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
  static final PathFragment IBTOOL = new PathFragment(IosSdkCommands.IBTOOL_PATH);

  // TODO(bazel-team): Reference a rule target rather than a jar file when Darwin runfiles work
  // better.
  static SpawnAction.Builder spawnJavaOnDarwinActionBuilder(Artifact deployJarArtifact) {
    return spawnOnDarwinActionBuilder()
        .setExecutable(JAVA)
        .addExecutableArguments("-jar", deployJarArtifact.getExecPathString())
        .addInput(deployJarArtifact);
  }

  private void register(Action... action) {
    actionRegistry.registerAction(action);
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

  private final class LinkCommandLine extends CommandLine {
    private final ObjcProvider objcProvider;
    private final Artifact linkedBinary;
    private final Optional<Artifact> dsymBundle;
    private final ExtraLinkArgs extraLinkArgs;

    LinkCommandLine(ExtraLinkArgs extraLinkArgs,
        ObjcProvider objcProvider, Artifact linkedBinary, Optional<Artifact> dsymBundle) {
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
      COMMAND_JOINER.appendTo(argumentStringBuilder, new ImmutableList.Builder<String>()
          .add(objcProvider.is(USES_CPP) ? clangPlusPlus.toString() : clang.toString())
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
        COMMAND_JOINER.appendTo(argumentStringBuilder, new ImmutableList.Builder<String>()
            .add("&&")
            .add(dsymutil.toString())
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
            new LinkCommandLine(extraLinkArgs, objcProvider, linkedBinary, dsymBundle))
        .addOutput(linkedBinary)
        .addOutputs(dsymBundle.asSet())
        .addTransitiveInputs(objcProvider.get(LIBRARY))
        .addTransitiveInputs(objcProvider.get(IMPORTED_LIBRARY))
        .addTransitiveInputs(objcProvider.get(FRAMEWORK_FILE))
        .addInputs(extraLinkInputs)
        .build(context));
  }
}
