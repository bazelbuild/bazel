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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.apple.ApplePlatform.PlatformType;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.rules.apple.XcodeConfigInfo;
import com.google.devtools.build.lib.rules.apple.XcodeVersionProperties;
import com.google.devtools.build.lib.rules.cpp.CcModule;
import com.google.devtools.build.lib.rules.cpp.CppSemantics;
import com.google.devtools.build.lib.rules.objc.ObjcProvider.Flag;
import com.google.devtools.build.lib.starlarkbuildapi.SplitTransitionProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.apple.AppleCommonApi;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.syntax.Location;

/** A class that exposes apple rule implementation internals to Starlark. */
public class AppleStarlarkCommon
    implements AppleCommonApi<
        Artifact,
        ConstraintValueInfo,
        StarlarkRuleContext,
        ObjcProvider,
        XcodeConfigInfo,
        ApplePlatform> {

  @VisibleForTesting
  public static final String DEPRECATED_KEY_ERROR =
      "Key '%s' no longer supported in ObjcProvider (use CcInfo instead).";

  @VisibleForTesting
  public static final String BAD_KEY_ERROR =
      "Argument %s not a recognized key, 'strict_include', or 'providers'.";

  @VisibleForTesting
  public static final String BAD_PROVIDERS_ITER_ERROR =
      "Value for argument 'providers' must be a list of ObjcProvider instances, instead found %s.";

  @VisibleForTesting
  public static final String BAD_PROVIDERS_ELEM_ERROR =
      "Value for argument 'providers' must be a list of ObjcProvider instances, instead found "
          + "iterable with %s.";

  @VisibleForTesting
  public static final String NOT_SET_ERROR = "Value for key %s must be a set, instead found %s.";

  @Nullable private StructImpl platformType;
  @Nullable private StructImpl platform;

  private final CppSemantics cppSemantics;

  public AppleStarlarkCommon(CppSemantics cppSemantics) {
    this.cppSemantics = cppSemantics;
  }

  @Override
  public AppleToolchain getAppleToolchain() {
    return new AppleToolchain();
  }

  @Override
  public StructImpl getPlatformTypeStruct() {
    if (platformType == null) {
      platformType = PlatformType.getStarlarkStruct();
    }
    return platformType;
  }

  @Override
  public StructImpl getPlatformStruct() {
    if (platform == null) {
      platform = ApplePlatform.getStarlarkStruct();
    }
    return platform;
  }

  @Override
  public Provider getXcodeVersionPropertiesConstructor() {
    return XcodeVersionProperties.STARLARK_CONSTRUCTOR;
  }

  @Override
  public Provider getXcodeVersionConfigConstructor() {
    return XcodeConfigInfo.PROVIDER;
  }

  @Override
  public Provider getObjcProviderConstructor() {
    return ObjcProvider.STARLARK_CONSTRUCTOR;
  }

  @Override
  public Provider getAppleDynamicFrameworkConstructor() {
    return AppleDynamicFrameworkInfo.STARLARK_CONSTRUCTOR;
  }

  @Override
  public Provider getAppleDylibBinaryConstructor() {
    return AppleDylibBinaryInfo.STARLARK_CONSTRUCTOR;
  }

  @Override
  public Provider getAppleExecutableBinaryConstructor() {
    return AppleExecutableBinaryInfo.STARLARK_CONSTRUCTOR;
  }

  @Override
  public AppleStaticLibraryInfo.Provider getAppleStaticLibraryProvider() {
    return AppleStaticLibraryInfo.STARLARK_CONSTRUCTOR;
  }

  @Override
  public Provider getAppleDebugOutputsConstructor() {
    return AppleDebugOutputsInfo.STARLARK_CONSTRUCTOR;
  }

  @Override
  public Provider getAppleLoadableBundleBinaryConstructor() {
    return AppleLoadableBundleBinaryInfo.STARLARK_CONSTRUCTOR;
  }

  @Override
  public ImmutableMap<String, String> getAppleHostSystemEnv(XcodeConfigInfo xcodeConfig) {
    return AppleConfiguration.getXcodeVersionEnv(xcodeConfig.getXcodeVersion());
  }

  @Override
  public ImmutableMap<String, String> getTargetAppleEnvironment(
      XcodeConfigInfo xcodeConfigApi, ApplePlatform platformApi) {
    XcodeConfigInfo xcodeConfig = xcodeConfigApi;
    ApplePlatform platform = (ApplePlatform) platformApi;
    return AppleConfiguration.appleTargetPlatformEnv(
        platform, xcodeConfig.getSdkVersionForPlatform(platform));
  }

  @Override
  public SplitTransitionProviderApi getMultiArchSplitProvider() {
    return new MultiArchSplitTransitionProvider();
  }

  @Override
  // This method is registered statically for Starlark, and never called directly.
  public ObjcProvider newObjcProvider(Dict<String, Object> kwargs, StarlarkThread thread)
      throws EvalException {
    ObjcProvider.StarlarkBuilder resultBuilder =
        new ObjcProvider.StarlarkBuilder(thread.getSemantics());
    for (Map.Entry<String, Object> entry : kwargs.entrySet()) {
      ObjcProvider.Key<?> key = ObjcProvider.getStarlarkKeyForString(entry.getKey());
      if (key != null) {
        resultBuilder.addElementsFromStarlark(key, entry.getValue());
      } else {
        switch (entry.getKey()) {
          case "cc_library":
            CcModule.checkPrivateStarlarkificationAllowlist(thread);
            resultBuilder.uncheckedAddTransitive(
                ObjcProvider.CC_LIBRARY,
                ObjcProviderStarlarkConverters.convertToJava(
                    ObjcProvider.CC_LIBRARY, entry.getValue()));
            break;
          case "linkstamp":
            CcModule.checkPrivateStarlarkificationAllowlist(thread);
            resultBuilder.uncheckedAddTransitive(
                ObjcProvider.LINKSTAMP,
                ObjcProviderStarlarkConverters.convertToJava(
                    ObjcProvider.LINKSTAMP, entry.getValue()));
            break;
          case "flag":
            resultBuilder.add(ObjcProvider.FLAG, Flag.USES_CPP);
            break;
          case "strict_include":
            resultBuilder.addStrictIncludeFromStarlark(entry.getValue());
            break;
          case "providers":
            resultBuilder.addProvidersFromStarlark(entry.getValue());
            break;
          default:
            throw Starlark.errorf(BAD_KEY_ERROR, entry.getKey());
        }
      }
    }
    return resultBuilder.build();
  }

  @Override
  public AppleDynamicFrameworkInfo newDynamicFrameworkProvider(
      Object dylibBinary,
      ObjcProvider depsObjcProvider,
      Object dynamicFrameworkDirs,
      Object dynamicFrameworkFiles)
      throws EvalException {
    NestedSet<String> frameworkDirs =
        Depset.noneableCast(dynamicFrameworkDirs, String.class, "framework_dirs");
    NestedSet<Artifact> frameworkFiles =
        Depset.noneableCast(dynamicFrameworkFiles, Artifact.class, "framework_files");
    Artifact binary = (dylibBinary != Starlark.NONE) ? (Artifact) dylibBinary : null;

    return new AppleDynamicFrameworkInfo(binary, depsObjcProvider, frameworkDirs, frameworkFiles);
  }

  @Override
  public AppleExecutableBinaryInfo newExecutableBinaryProvider(
      Object executableBinary, ObjcProvider depsObjcProvider) throws EvalException {
    Artifact binary = (executableBinary != Starlark.NONE) ? (Artifact) executableBinary : null;
    return new AppleExecutableBinaryInfo(binary, depsObjcProvider);
  }

  @Override
  public StructImpl linkMultiArchBinary(
      StarlarkRuleContext starlarkRuleContext,
      Object avoidDeps,
      Sequence<?> extraLinkopts,
      Sequence<?> extraLinkInputs,
      StarlarkInt stamp,
      Boolean shouldLipo,
      StarlarkThread thread)
      throws EvalException, InterruptedException {
    try {
      RuleContext ruleContext = starlarkRuleContext.getRuleContext();
      ImmutableList<TransitiveInfoCollection> avoidDepsList =
          (avoidDeps != Starlark.NONE)
              ? ImmutableList.copyOf(
                  Sequence.cast(avoidDeps, TransitiveInfoCollection.class, "avoid_deps"))
              : ImmutableList.of();
      boolean isStampingEnabled =
          isStampingEnabled(stamp.toInt("stamp"), ruleContext.getConfiguration());
      AppleLinkingOutputs linkingOutputs =
          AppleBinary.linkMultiArchBinary(
              ruleContext,
              cppSemantics,
              avoidDepsList,
              ImmutableList.copyOf(Sequence.cast(extraLinkopts, String.class, "extra_linkopts")),
              Sequence.cast(extraLinkInputs, Artifact.class, "extra_link_inputs"),
              isStampingEnabled,
              shouldLipo);
      return createStarlarkLinkingOutputs(linkingOutputs, thread, shouldLipo);
    } catch (RuleErrorException | ActionConflictException exception) {
      throw new EvalException(exception);
    }
  }

  @Override
  public DottedVersion dottedVersion(String version) throws EvalException {
    try {
      return DottedVersion.fromString(version);
    } catch (DottedVersion.InvalidDottedVersionException e) {
      throw new EvalException(e.getMessage());
    }
  }

  /**
   * Returns the given value unless it is null, in which case the Starlark value {@code NONE} is
   * returned.
   */
  private Object valueOrNone(Object value) {
    if (value != null) {
      return value;
    }
    return Starlark.NONE;
  }

  /**
   * Creates a Starlark struct that contains the results of the {@code link_multi_arch_binary}
   * function.
   */
  private StructImpl createStarlarkLinkingOutputs(
      AppleLinkingOutputs linkingOutputs, StarlarkThread thread, boolean shouldLipo) {
    Provider linkingOutputConstructor =
        new BuiltinProvider<StructImpl>("apple_linking_output", StructImpl.class) {};
    ImmutableList.Builder<StarlarkInfo> outputStructs = ImmutableList.builder();

    for (AppleLinkingOutputs.LinkingOutput linkingOutput : linkingOutputs.getOutputs()) {
      outputStructs.add(
          StarlarkInfo.create(
              linkingOutputConstructor,
              ImmutableMap.<String, Object>builder()
                  .put("platform", linkingOutput.getPlatform())
                  .put("architecture", linkingOutput.getArchitecture())
                  .put("environment", linkingOutput.getEnvironment())
                  .put("binary", linkingOutput.getBinary())
                  .put("bitcode_symbols", valueOrNone(linkingOutput.getBitcodeSymbols()))
                  .put("dsym_binary", valueOrNone(linkingOutput.getDsymBinary()))
                  .put("linkmap", valueOrNone(linkingOutput.getLinkmap()))
                  .build(),
              Location.BUILTIN));
    }

    // We have to transform the output group dictionary into one that contains StarlarkValues
    // instead of plain NestedSets because the Starlark caller may want to return this directly from
    // their implementation function.
    Map<String, StarlarkValue> outputGroups =
        Maps.transformValues(linkingOutputs.getOutputGroups(), v -> Depset.of(Artifact.TYPE, v));

    ImmutableMap.Builder<String, Object> fields = ImmutableMap.builder();
    fields.put("objc", linkingOutputs.getDepsObjcProvider());
    fields.put("output_groups", Dict.copyOf(thread.mutability(), outputGroups));
    fields.put("outputs", StarlarkList.copyOf(thread.mutability(), outputStructs.build()));

    // TODO(b/110264170): Remove this field after clients have been migrated to use a provider
    // defined in Starlark and propagated by rules_apple instead.
    fields.put("debug_outputs_provider", linkingOutputs.getLegacyDebugOutputsProvider());

    if (shouldLipo) {
      fields.put("binary", linkingOutputs.getLegacyBinaryArtifact());
      fields.put("binary_provider", linkingOutputs.getLegacyBinaryInfoProvider());
    }

    Provider linkingOutputsConstructor =
        new BuiltinProvider<StructImpl>("apple_linking_outputs", StructImpl.class) {};
    return StarlarkInfo.create(linkingOutputsConstructor, fields.build(), Location.BUILTIN);
  }

  private static boolean isStampingEnabled(int stamp, BuildConfiguration config)
      throws EvalException {
    if (stamp == 0) {
      return false;
    }
    if (stamp == 1) {
      return true;
    }
    if (stamp == -1) {
      return config.stampBinaries();
    }
    throw Starlark.errorf(
        "stamp value %d is not supported; must be 0 (disabled), 1 (enabled), or -1 (default)",
        stamp);
  }
}
