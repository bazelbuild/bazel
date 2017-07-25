// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.MULTI_ARCH_LINKED_BINARIES;
import static com.google.devtools.build.lib.syntax.Type.STRING;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfigurationMakeVariableContext;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.MakeVariableExpander;
import com.google.devtools.build.lib.analysis.MakeVariableExpander.ExpansionException;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.apple.ApplePlatform.PlatformType;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.build.lib.vfs.PathFragment;

/** Implementation for the "apple_stub_binary" rule. */
public class AppleStubBinary implements RuleConfiguredTargetFactory {

  /** Provides substitutions for the make variables that can be used in an xcenv_based_path */
  private static class XcenvBasedPathVariableContext extends ConfigurationMakeVariableContext {
    private final RuleContext ruleContext;

    /** The platform used to build $(PLATFORM_DIR). */
    private final ApplePlatform platform;

    /** The complete set of variables that may be used in paths. */
    public static final ImmutableList<String> DEFINED_VARS =
        ImmutableList.of("$(SDKROOT)", "$(PLATFORM_DIR)");

    public XcenvBasedPathVariableContext(RuleContext ruleContext, ApplePlatform platform) {
      super(
          ImmutableMap.<String, String>of(),
          ruleContext.getRule().getPackage(),
          ruleContext.getConfiguration());
      this.ruleContext = ruleContext;
      this.platform = platform;
    }

    /** Throws an exception if the given path is not rooted at a defined Make variable. */
    public void validatePathRoot(String path) throws RuleErrorException {
      for (String var : DEFINED_VARS) {
        if (path.startsWith(var)) {
          return;
        }
      }

      throw ruleContext.throwWithAttributeError(
          AppleStubBinaryRule.XCENV_BASED_PATH_ATTR,
          String.format(
              PATH_INCORRECTLY_ROOTED_ERROR_FORMAT,
              StringUtil.joinEnglishList(XcenvBasedPathVariableContext.DEFINED_VARS, "or")));
    }

    @Override
    public String lookupMakeVariable(String var) throws ExpansionException {
      if (var.equals("SDKROOT")) {
        return "__BAZEL_XCODE_SDKROOT__";
      }
      if (var.equals("PLATFORM_DIR")) {
        return AppleToolchain.platformDir(platform.getNameInPlist());
      }
      // Intentionally do not call super, because we only want to allow these specific variables and
      // discard any that might be inherited from toolchains and other contexts.
      throw new MakeVariableExpander.ExpansionException("$(" + var + ") not defined");
    }
  }

  @VisibleForTesting
  public static final String PATH_INCORRECTLY_ROOTED_ERROR_FORMAT =
      "The stub binary path must be rooted at %s";

  @VisibleForTesting
  public static final String PATH_NOT_NORMALIZED_ERROR =
      "The stub binary path must be normalized (i.e., not contain \".\" or \"..\")";

  @Override
  public final ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    MultiArchSplitTransitionProvider.validateMinimumOs(ruleContext);
    PlatformType platformType = MultiArchSplitTransitionProvider.getPlatformType(ruleContext);

    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);

    ApplePlatform platform = appleConfiguration.getMultiArchPlatform(platformType);
    ImmutableListMultimap<BuildConfiguration, ObjcProvider> configurationToDepsMap =
        ruleContext.getPrerequisitesByConfiguration(
            "deps", Mode.SPLIT, ObjcProvider.SKYLARK_CONSTRUCTOR);

    Artifact outputArtifact =
        ObjcRuleClasses.intermediateArtifacts(ruleContext).combinedArchitectureBinary();

    registerActions(ruleContext, appleConfiguration, platform, outputArtifact);

    NestedSetBuilder<Artifact> filesToBuild =
        NestedSetBuilder.<Artifact>stableOrder().add(outputArtifact);
    RuleConfiguredTargetBuilder targetBuilder =
        ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build());

    ObjcProvider.Builder objcProviderBuilder = new ObjcProvider.Builder();
    for (ObjcProvider depProvider : configurationToDepsMap.values()) {
      objcProviderBuilder.addTransitiveAndPropagate(depProvider);
    }
    objcProviderBuilder.add(MULTI_ARCH_LINKED_BINARIES, outputArtifact);

    ObjcProvider objcProvider = objcProviderBuilder.build();
    // TODO(cparsons): Stop propagating ObjcProvider directly from this rule.
    targetBuilder.addNativeDeclaredProvider(objcProvider);

    targetBuilder.addNativeDeclaredProvider(
        new AppleExecutableBinaryProvider(outputArtifact, objcProvider));

    return targetBuilder.build();
  }

  private static FilesToRunProvider xcrunwrapper(RuleContext ruleContext) {
    return ruleContext.getExecutablePrerequisite("$xcrunwrapper", Mode.HOST);
  }

  /** Registers the actions that copy the stub binary to the target's output. */
  private static void registerActions(
      RuleContext ruleContext,
      AppleConfiguration appleConfiguration,
      ApplePlatform platform,
      Artifact outputBinary)
      throws RuleErrorException {
    CustomCommandLine copyCommandLine =
        new CustomCommandLine.Builder()
            .add("/bin/cp")
            .add(resolveXcenvBasedPath(ruleContext, platform))
            .addExecPaths(ImmutableList.of(outputBinary))
            .build();

    ruleContext.registerAction(
        ObjcRuleClasses.spawnAppleEnvActionBuilder(appleConfiguration, platform)
            .setExecutable(xcrunwrapper(ruleContext))
            .setCommandLine(copyCommandLine)
            .setMnemonic("CopyStubExecutable")
            .addOutput(outputBinary)
            .disableSandboxing()
            .build(ruleContext));
  }

  /**
   * Returns the value of the xcenv_based_path attribute, emitting an error if its format is
   * invalid.
   *
   * @param ruleContext the rule context
   * @param platform the Apple platform
   * @return the path string, if it was valid
   * @throws RuleErrorException If the path string was invalid because it was not rooted at one of
   *     the allowed environment variables or it was not normalized
   */
  private static String resolveXcenvBasedPath(RuleContext ruleContext, ApplePlatform platform)
      throws RuleErrorException {
    String pathString =
        ruleContext.attributes().get(AppleStubBinaryRule.XCENV_BASED_PATH_ATTR, STRING);
    XcenvBasedPathVariableContext makeVariableContext =
        new XcenvBasedPathVariableContext(ruleContext, platform);

    makeVariableContext.validatePathRoot(pathString);

    PathFragment pathFragment = PathFragment.create(pathString);
    if (!pathFragment.isNormalized()) {
      throw ruleContext.throwWithAttributeError(
          AppleStubBinaryRule.XCENV_BASED_PATH_ATTR, PATH_NOT_NORMALIZED_ERROR);
    }

    return ruleContext.expandMakeVariables(
        AppleStubBinaryRule.XCENV_BASED_PATH_ATTR, pathString, makeVariableContext);
  }
}
