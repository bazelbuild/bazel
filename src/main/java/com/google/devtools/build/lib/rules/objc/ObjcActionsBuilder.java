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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.ByteSource;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionRegistry;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.BinaryFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos;

import java.io.InputStream;

/**
 * Object that creates actions used by Objective-C rules.
 */
final class ObjcActionsBuilder {

  private final ActionConstructionContext context;
  private final IntermediateArtifacts intermediateArtifacts;
  private final ObjcConfiguration objcConfiguration;
  private final ActionRegistry actionRegistry;

  /**
   * @param context {@link ActionConstructionContext} of the rule
   * @param intermediateArtifacts provides intermediate output paths for this rule
   * @param objcConfiguration configuration for this rule
   * @param actionRegistry registry with which to register new actions
   */
  ObjcActionsBuilder(
      ActionConstructionContext context,
      IntermediateArtifacts intermediateArtifacts,
      ObjcConfiguration objcConfiguration,
      ActionRegistry actionRegistry) {
    this.context = Preconditions.checkNotNull(context);
    this.intermediateArtifacts = Preconditions.checkNotNull(intermediateArtifacts);
    this.objcConfiguration = Preconditions.checkNotNull(objcConfiguration);
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
}
