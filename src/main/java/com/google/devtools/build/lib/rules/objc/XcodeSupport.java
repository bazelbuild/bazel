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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.io.ByteSource;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.BinaryFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.objc.ReleaseBundlingSupport.SplitArchTransition.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.objc.XcodeProvider.Builder;
import com.google.devtools.build.lib.rules.objc.XcodeProvider.Project;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.XcodeprojBuildSetting;

import java.io.InputStream;
import java.util.List;

/**
 * Support for Objc rule types that export an Xcode provider or generate xcode project files.
 *
 * <p>Methods on this class can be called in any order without impacting the result.
 *
 * <p>These objects should not outlast the analysis phase. Do not pass them to {@link Action}
 * objects or other persistent objects. There are internal tests to ensure that XcodeSupport objects
 * are not persisted that check the name of this class, so update those tests if you change this
 * class's name.
 */
public final class XcodeSupport {

  /**
   * Template for a target's xcode project.
   */
  public static final SafeImplicitOutputsFunction PBXPROJ =
      fromTemplates("%{name}.xcodeproj/project.pbxproj");

  private final RuleContext ruleContext;
  private final IntermediateArtifacts intermediateArtifacts;
  private final Label xcodeTargetLabel;

  /**
   * Creates a new xcode support for the given context.
   */
  XcodeSupport(RuleContext ruleContext)  {
    this(ruleContext, ObjcRuleClasses.intermediateArtifacts(ruleContext), ruleContext.getLabel());
  }

  /**
   * Creates a new xcode support for the given context and {@link IntermediateArtifacts} with given
   * target label.
   */
  public XcodeSupport(
      RuleContext ruleContext, IntermediateArtifacts intermediateArtifacts,
      Label xcodeTargetLabel) {
    this.ruleContext = ruleContext;
    this.intermediateArtifacts = intermediateArtifacts;
    this.xcodeTargetLabel = xcodeTargetLabel;
  }

  /**
   * Adds xcode project files to the given builder.
   *
   * @return this xcode support
   * @throws InterruptedException 
   */
  XcodeSupport addFilesToBuild(NestedSetBuilder<Artifact> filesToBuild)
      throws InterruptedException {
    filesToBuild.add(ruleContext.getImplicitOutputArtifact(PBXPROJ));
    return this;
  }

  /**
   * Adds a dummy source file to the Xcode target. This is needed if the target does not have any
   * source files but Xcode requires one.
   *
   * @return this xcode support
   */
  XcodeSupport addDummySource(XcodeProvider.Builder xcodeProviderBuilder) {
    ruleContext.registerAction(new SymlinkAction(
        ruleContext.getActionOwner(),
        ruleContext.getPrerequisiteArtifact("$dummy_source", Mode.TARGET),
        intermediateArtifacts.dummySource(),
        "Symlinking dummy artifact"));

    xcodeProviderBuilder.addAdditionalSources(intermediateArtifacts.dummySource());
    return this;
  }

  /**
   * Registers actions that generate the rule's Xcode project.
   *
   * @param xcodeProvider information about this rule's xcode settings and that of its dependencies
   * @return this xcode support
   * @throws InterruptedException 
   */
  XcodeSupport registerActions(XcodeProvider xcodeProvider) throws InterruptedException {
    registerXcodegenActions(XcodeProvider.Project.fromTopLevelTarget(xcodeProvider));
    return this;
  }

  /**
   * Registers actions that generate the rule's Xcode project.
   *
   * @param xcodeProviders information about several rules' xcode settings
   * @return this xcode support
   * @throws InterruptedException 
   */
  XcodeSupport registerActions(Iterable<XcodeProvider> xcodeProviders) throws InterruptedException {
    registerXcodegenActions(Project.fromTopLevelTargets(xcodeProviders));
    return this;
  }

  /**
   * Adds common xcode settings to the given provider builder.
   *
   * @param objcProvider provider containing all dependencies' information as well as some of this
   *    rule's
   * @param productType type of this rule's Xcode target
   *
   * @return this xcode support
   */
  XcodeSupport addXcodeSettings(XcodeProvider.Builder xcodeProviderBuilder,
      ObjcProvider objcProvider, XcodeProductType productType) {
    ObjcConfiguration objcConfiguration = ObjcRuleClasses.objcConfiguration(ruleContext);
    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);
    return addXcodeSettings(xcodeProviderBuilder, objcProvider, productType,
        appleConfiguration.getIosCpu(), objcConfiguration.getConfigurationDistinguisher());
  }

  /**
   * Adds common xcode settings to the given provider builder, explicitly specifying architecture
   * to use.
   *
   * @param objcProvider provider containing all dependencies' information as well as some of this
   *    rule's
   * @param productType type of this rule's Xcode target
   * @param architecture architecture to filter all dependencies with (only matching ones will be
   *    included in the final targets generated)
   * @param configurationDistinguisher distinguisher that will cause this target's xcode provider to
   *    discard any dependencies from sources that are tagged with a different distinguisher
   * @return this xcode support
   */
  XcodeSupport addXcodeSettings(Builder xcodeProviderBuilder,
      ObjcProvider objcProvider, XcodeProductType productType, String architecture,
      ConfigurationDistinguisher configurationDistinguisher) {
    xcodeProviderBuilder
        .setLabel(xcodeTargetLabel)
        .setArchitecture(architecture)
        .setConfigurationDistinguisher(configurationDistinguisher)
        .setObjcProvider(objcProvider)
        .setProductType(productType)
        .addXcodeprojBuildSettings(XcodeSupport.defaultXcodeSettings());
    return this;
  }

  /**
   * Adds dependencies to the given provider builder from the given attribute.
   *
   * @return this xcode support
   */
  XcodeSupport addDependencies(Builder xcodeProviderBuilder, Attribute attribute) {
    xcodeProviderBuilder.addPropagatedDependencies(
        ruleContext.getPrerequisites(
            attribute.getName(), attribute.getAccessMode(), XcodeProvider.class));
    return this;
  }

  /**
   * Adds non-propagated dependencies to the given provider builder from the given attribute.
   *
   * <p>A non-propagated dependency will not be linked into the final app bundle and can only serve
   * as a compile-only dependency for its direct dependent.
   *
   * @return this xcode support
   */
  XcodeSupport addNonPropagatedDependencies(Builder xcodeProviderBuilder, Attribute attribute) {
    xcodeProviderBuilder.addNonPropagatedDependencies(
        ruleContext.getPrerequisites(
            attribute.getName(), attribute.getAccessMode(), XcodeProvider.class));
    return this;
  }

  /**
   * Generates an extra {@link XcodeProductType#LIBRARY_STATIC} Xcode target with the same
   * compilation artifacts as the main Xcode target associated with this Xcode support. The extra
   * Xcode library target, instead of the main Xcode target, will act as a dependency for all
   * dependent Xcode targets.
   *
   * <p>This is needed to build the Xcode binary target generated by ios_application in XCode.
   * Currently there is an Xcode target dependency between the binary target from ios_application
   * and the binary target from objc_binary. But Xcode does not link in compiled artifacts from
   * binary dependencies, so any sources specified on objc_binary rules will not be compiled and
   * linked into the app bundle in dependent binary targets associated with ios_application in
   * XCode.
   */
  // TODO(bazel-team): Remove this when the binary rule types and bundling rule types are merged.
  XcodeSupport generateCompanionLibXcodeTarget(Builder xcodeProviderBuilder) {
    xcodeProviderBuilder.generateCompanionLibTarget();
    return this;
  }

  private void registerXcodegenActions(XcodeProvider.Project project) throws InterruptedException {
    Artifact controlFile = intermediateArtifacts.pbxprojControlArtifact();

    ruleContext.registerAction(new BinaryFileWriteAction(
        ruleContext.getActionOwner(),
        controlFile,
        xcodegenControlFileBytes(project),
        /*makeExecutable=*/false));

    ruleContext.registerAction(new SpawnAction.Builder()
        .setMnemonic("GenerateXcodeproj")
        .setExecutable(ruleContext.getExecutablePrerequisite("$xcodegen", Mode.HOST))
        .addArgument("--control")
        .addInputArgument(controlFile)
        .addOutput(ruleContext.getImplicitOutputArtifact(XcodeSupport.PBXPROJ))
        .addTransitiveInputs(project.getInputsToXcodegen())
        .addTransitiveInputs(project.getAdditionalSources())
        .build(ruleContext));
  }

  /**
   * Static class to avoid keeping references to configurations and this XcodeSupport object during
   * execution.
   */
  private static class XcodegenControlFileBytes extends ByteSource {
    private final XcodeProvider.Project project;
    private final Artifact pbxproj;
    private final String workspaceRoot;
    private final List<String> appleCpus;
    private final String minimumOs;
    private final boolean generateDebugSymbols;

    XcodegenControlFileBytes(
        ObjcConfiguration objcConfiguration,
        AppleConfiguration appleConfiguration,
        Project project,
        Artifact pbxproj) {
      this.project = project;
      this.pbxproj = pbxproj;
      this.workspaceRoot = objcConfiguration.getXcodeWorkspaceRoot();
      List<String> multiCpus = appleConfiguration.getIosMultiCpus();
      if (multiCpus.isEmpty()) {
        this.appleCpus = ImmutableList.of(appleConfiguration.getIosCpu());
      } else {
        this.appleCpus = multiCpus;
      }
      this.minimumOs = objcConfiguration.getMinimumOs().toString();
      this.generateDebugSymbols =
          objcConfiguration.generateDebugSymbols() || objcConfiguration.generateDsym();
    }

    @Override
    public InputStream openStream() {
      XcodeGenProtos.Control.Builder builder = XcodeGenProtos.Control.newBuilder();
      if (workspaceRoot != null) {
        builder.setWorkspaceRoot(workspaceRoot);
      }

      builder.addAllCpuArchitecture(appleCpus);

      return builder
          .setPbxproj(pbxproj.getExecPathString())
          .addAllTarget(project.targets())
          .addBuildSetting(
              XcodeGenProtos.XcodeprojBuildSetting.newBuilder()
                  .setName("IPHONEOS_DEPLOYMENT_TARGET")
                  .setValue(minimumOs)
                  .build())
          .addBuildSetting(
              XcodeGenProtos.XcodeprojBuildSetting.newBuilder()
                  .setName("DEBUG_INFORMATION_FORMAT")
                  .setValue(generateDebugSymbols ? "dwarf-with-dsym" : "dwarf")
                  .build())
          .build()
          .toByteString()
          .newInput();
    }
  }

  private ByteSource xcodegenControlFileBytes(XcodeProvider.Project project)
      throws InterruptedException {
    return new XcodegenControlFileBytes(
        ObjcRuleClasses.objcConfiguration(ruleContext),
        ruleContext.getFragment(AppleConfiguration.class),
        project,
        ruleContext.getImplicitOutputArtifact(XcodeSupport.PBXPROJ));
  }

  /**
   * Returns a list of default XCode build settings for Bazel-generated XCode projects.
   */
  @VisibleForTesting
  static Iterable<XcodeprojBuildSetting> defaultXcodeSettings() {
    // Do not use XCode headermap because Bazel-generated header search paths are sufficient for
    // resolving header imports.
    return ImmutableList.of(
        XcodeprojBuildSetting.newBuilder()
            .setName("USE_HEADERMAP")
            .setValue("NO")
            .build());
  }
}
