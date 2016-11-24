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

import static com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode.TARGET;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.MULTI_ARCH_LINKED_BINARIES;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.DylibDependingRule.DYLIBS_ATTR_NAME;
import static com.google.devtools.build.lib.syntax.Type.STRING;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Functions;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.Platform;
import com.google.devtools.build.lib.rules.apple.Platform.PlatformType;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraLinkArgs;
import java.util.Map;
import java.util.Set;

/**
 * Implementation for the "apple_binary" rule.
 */
public class AppleBinary implements RuleConfiguredTargetFactory {

  // TODO(b/33077308): Expand into DYLIB when apple_dynamic_library is removed.
  enum BinaryType {
    EXECUTABLE, BUNDLE;

    @Override
    public String toString() {
      return name().toLowerCase();
    }

    /**
     * Returns the {@link BinaryType} with given name (case insensitive).
     *
     * @throws IllegalArgumentException if the name does not match a valid platform type.
     */
    public static BinaryType fromString(String name) {
      for (BinaryType binaryType : BinaryType.values()) {
        if (name.equalsIgnoreCase(binaryType.toString())) {
          return binaryType;
        }
      }
      throw new IllegalArgumentException(String.format("Unsupported binary type \"%s\"", name));
    }

    /** Returns the enum values as a list of strings for validation. */
    static Iterable<String> getValues() {
      return Iterables.transform(ImmutableList.copyOf(values()), Functions.toStringFunction());
    }
  }

  @VisibleForTesting
  static final String BUNDLE_LOADER_NOT_IN_BUNDLE_ERROR =
      "Can only use bundle_loader when binary_type is bundle.";

  @Override
  public final ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    PlatformType platformType = MultiArchSplitTransitionProvider.getPlatformType(ruleContext);
    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);

    Platform platform = appleConfiguration.getMultiArchPlatform(platformType);
    ImmutableListMultimap<BuildConfiguration, ObjcProvider> configurationToNonPropagatedObjcMap =
        ruleContext.getPrerequisitesByConfiguration("non_propagated_deps", Mode.SPLIT,
            ObjcProvider.class);
    ImmutableListMultimap<BuildConfiguration, TransitiveInfoCollection> configToDepsCollectionMap =
        ruleContext.getPrerequisitesByConfiguration("deps", Mode.SPLIT);
    Iterable<ObjcProvider> dylibProviders =
        ruleContext.getPrerequisites(DYLIBS_ATTR_NAME, Mode.TARGET, ObjcProvider.class);
    Set<BuildConfiguration> childConfigurations = getChildConfigurations(ruleContext);
    Artifact outputArtifact =
        ObjcRuleClasses.intermediateArtifacts(ruleContext).combinedArchitectureBinary();

    MultiArchBinarySupport multiArchBinarySupport = new MultiArchBinarySupport(ruleContext);
    Map<BuildConfiguration, ObjcCommon> objcCommonByDepConfiguration =
        multiArchBinarySupport.objcCommonByDepConfiguration(childConfigurations,
            configToDepsCollectionMap, configurationToNonPropagatedObjcMap, dylibProviders);

    multiArchBinarySupport.registerActions(
        platform,
        getExtraLinkArgs(ruleContext),
        getExtraLinkInputs(ruleContext),
        objcCommonByDepConfiguration,
        configToDepsCollectionMap,
        outputArtifact);

    NestedSetBuilder<Artifact> filesToBuild =
        NestedSetBuilder.<Artifact>stableOrder().add(outputArtifact);
    RuleConfiguredTargetBuilder targetBuilder =
        ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build());

    ObjcProvider.Builder objcProviderBuilder = new ObjcProvider.Builder();
    for (ObjcCommon objcCommon : objcCommonByDepConfiguration.values()) {
      objcProviderBuilder.addTransitiveAndPropagate(objcCommon.getObjcProvider());
    }
    objcProviderBuilder.add(MULTI_ARCH_LINKED_BINARIES, outputArtifact);

    targetBuilder.addProvider(ObjcProvider.class, objcProviderBuilder.build());
    return targetBuilder.build();
  }

  private ExtraLinkArgs getExtraLinkArgs(RuleContext ruleContext) throws RuleErrorException {
    String binaryTypeString = ruleContext
        .attributes()
        .get(AppleBinaryRule.BINARY_TYPE_ATTR, STRING);
    BinaryType binaryType = BinaryType.fromString(binaryTypeString);

    ImmutableList.Builder<String> extraLinkArgs = new ImmutableList.Builder<>();

    boolean didProvideBundleLoader =
        ruleContext
            .attributes()
            .isAttributeValueExplicitlySpecified(AppleBinaryRule.BUNDLE_LOADER_ATTR);

    if (didProvideBundleLoader && binaryType != BinaryType.BUNDLE) {
      ruleContext.throwWithRuleError(BUNDLE_LOADER_NOT_IN_BUNDLE_ERROR);
    }

    switch(binaryType) {
      case EXECUTABLE: break;
      case BUNDLE: {
        extraLinkArgs.add("-bundle");
        if (didProvideBundleLoader) {
          Artifact bundleLoader =
              ruleContext.getPrerequisiteArtifact(AppleBinaryRule.BUNDLE_LOADER_ATTR, TARGET);
          extraLinkArgs.add("-bundle_loader " + bundleLoader.getExecPathString());
        }
        break;
      }
    }

    return new ExtraLinkArgs(extraLinkArgs.build());
  }

  private Iterable<Artifact> getExtraLinkInputs(RuleContext ruleContext) {
    return Optional.fromNullable(
            ruleContext.getPrerequisiteArtifact(AppleBinaryRule.BUNDLE_LOADER_ATTR, TARGET))
        .asSet();
  }

  private Set<BuildConfiguration> getChildConfigurations(RuleContext ruleContext) {
    // This is currently a hack to obtain all child configurations regardless of the attribute
    // values of this rule -- this rule does not currently use the actual info provided by
    // this attribute. b/28403953 tracks cc toolchain usage.
    ImmutableListMultimap<BuildConfiguration, CcToolchainProvider> configToProvider =
        ruleContext.getPrerequisitesByConfiguration(":cc_toolchain", Mode.SPLIT,
            CcToolchainProvider.class);

    return configToProvider.keySet();
  }
}
