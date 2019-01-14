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

package com.google.devtools.build.lib.rules.cpp;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.cpp.CcCompilationHelper.CompilationInfo;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.HeadersCheckingMode;
import com.google.devtools.build.lib.rules.cpp.LibraryToLinkWrapper.CcLinkingContext;
import com.google.devtools.build.lib.syntax.Type;
import java.util.List;
import javax.annotation.Nullable;

/**
 * A ConfiguredTarget for <code>cc_import</code> rule.
 */
public abstract class CcImport implements RuleConfiguredTargetFactory {
  private final CppSemantics semantics;

  protected CcImport(CppSemantics semantics) {
    this.semantics = semantics;
  }

  /** Autovalue for output of private methods in this class. */
  @AutoValue
  public abstract static class NoPicAndPicStaticLibrary {
    private static NoPicAndPicStaticLibrary create(@Nullable Artifact staticLibrary) {
      Artifact noPicStaticLibrary = null;
      Artifact picStaticLibrary = null;
      if (staticLibrary != null) {
        if (staticLibrary.getExtension().equals(".pic.a")) {
          picStaticLibrary = staticLibrary;
        } else {
          noPicStaticLibrary = staticLibrary;
        }
      }
      return new AutoValue_CcImport_NoPicAndPicStaticLibrary(
          /* noPicStaticLibrary= */ noPicStaticLibrary, /* picStaticLibrary= */ picStaticLibrary);
    }

    @Nullable
    abstract Artifact noPicStaticLibrary();

    @Nullable
    abstract Artifact picStaticLibrary();
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {

    boolean systemProvided = ruleContext.attributes().get("system_provided", Type.BOOLEAN);
    CcToolchainProvider ccToolchain =
        CppHelper.getToolchainUsingDefaultCcToolchainAttribute(ruleContext);
    FeatureConfiguration featureConfiguration =
        CcCommon.configureFeaturesOrReportRuleError(ruleContext, ccToolchain);
    boolean targetWindows = featureConfiguration.isEnabled(CppRuleClasses.TARGETS_WINDOWS);

    Artifact staticLibrary = ruleContext.getPrerequisiteArtifact("static_library", Mode.TARGET);
    Artifact sharedLibrary = ruleContext.getPrerequisiteArtifact("shared_library", Mode.TARGET);
    Artifact interfaceLibrary =
        ruleContext.getPrerequisiteArtifact("interface_library", Mode.TARGET);
    performErrorChecks(ruleContext, systemProvided, sharedLibrary, interfaceLibrary, targetWindows);

    Artifact resolvedSymlinkDynamicLibrary = null;
    Artifact resolvedSymlinkInterfaceLibrary = null;
    if (!featureConfiguration.isEnabled(CppRuleClasses.TARGETS_WINDOWS)) {
      if (sharedLibrary != null) {
        resolvedSymlinkDynamicLibrary = sharedLibrary;
        sharedLibrary =
            SolibSymlinkAction.getDynamicLibrarySymlink(
                /* actionRegistry= */ ruleContext,
                /* actionConstructionContext= */ ruleContext,
                ccToolchain.getSolibDirectory(),
                sharedLibrary,
                /* preserveName= */ true,
                /* prefixConsumer= */ true,
                /* configuration= */ null);
      }
      if (interfaceLibrary != null) {
        resolvedSymlinkInterfaceLibrary = interfaceLibrary;
        interfaceLibrary =
            SolibSymlinkAction.getDynamicLibrarySymlink(
                /* actionRegistry= */ ruleContext,
                /* actionConstructionContext= */ ruleContext,
                ccToolchain.getSolibDirectory(),
                interfaceLibrary,
                /* preserveName= */ true,
                /* prefixConsumer= */ true,
                /* configuration= */ null);
      }
    }

    Artifact notNullArtifactToLink = null;
    if (staticLibrary != null) {
      notNullArtifactToLink = staticLibrary;
    } else if (sharedLibrary != null) {
      notNullArtifactToLink = sharedLibrary;
    } else if (interfaceLibrary != null) {
      notNullArtifactToLink = interfaceLibrary;
    }

    NoPicAndPicStaticLibrary noPicAndPicStaticLibrary =
        NoPicAndPicStaticLibrary.create(staticLibrary);

    CcLinkingContext ccLinkingContext = CcLinkingContext.EMPTY;

    if (notNullArtifactToLink != null) {
      LibraryToLinkWrapper libraryToLinkWrapper =
          LibraryToLinkWrapper.builder()
              .setStaticLibrary(noPicAndPicStaticLibrary.noPicStaticLibrary())
              .setPicStaticLibrary(noPicAndPicStaticLibrary.picStaticLibrary())
              .setDynamicLibrary(sharedLibrary)
              .setResolvedSymlinkDynamicLibrary(resolvedSymlinkDynamicLibrary)
              .setInterfaceLibrary(interfaceLibrary)
              .setResolvedSymlinkInterfaceLibrary(resolvedSymlinkInterfaceLibrary)
              .setAlwayslink(ruleContext.attributes().get("alwayslink", Type.BOOLEAN))
              .setLibraryIdentifier(CcLinkingOutputs.libraryIdentifierOf(notNullArtifactToLink))
              .build();
      ccLinkingContext =
          CcLinkingContext.builder()
              .addLibraries(
                  NestedSetBuilder.<LibraryToLinkWrapper>linkOrder()
                      .add(libraryToLinkWrapper)
                      .build())
              .build();
    }

    final CcCommon common = new CcCommon(ruleContext);
    CompilationInfo compilationInfo =
        new CcCompilationHelper(
                ruleContext,
                semantics,
                featureConfiguration,
                ccToolchain,
                ccToolchain.getFdoContext())
            .addPublicHeaders(common.getHeaders())
            .setHeadersCheckingMode(HeadersCheckingMode.STRICT)
            .compile();

    RuleConfiguredTargetBuilder result =
        new RuleConfiguredTargetBuilder(ruleContext)
            .addProvider(compilationInfo.getCppDebugFileProvider())
            .addNativeDeclaredProvider(
                CcInfo.builder()
                    .setCcCompilationContext(compilationInfo.getCcCompilationContext())
                    .setCcLinkingContext(ccLinkingContext)
                    .build())
            .addOutputGroups(compilationInfo.getOutputGroups())
            .addProvider(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY));

    CcSkylarkApiProvider.maybeAdd(ruleContext, result);
    return result.build();
  }

  private void performErrorChecks(
      RuleContext ruleContext,
      boolean systemProvided,
      Artifact sharedLibrary,
      Artifact interfaceLibrary,
      boolean targetsWindows) {
    // If the shared library will be provided by system during runtime, users are not supposed to
    // specify shared_library.
    if (systemProvided && sharedLibrary != null) {
      ruleContext.ruleError(
          "'shared_library' shouldn't be specified when 'system_provided' is true");
    }
    // If a shared library won't be provided by system during runtime and we are linking the shared
    // library through interface library, the shared library must be specified.
    if (!systemProvided && sharedLibrary == null && interfaceLibrary != null) {
      ruleContext.ruleError(
          "'shared_library' should be specified when 'system_provided' is false");
    }

    if (targetsWindows && sharedLibrary != null && interfaceLibrary == null) {
      ruleContext.ruleError(
          "'interface library' must be specified when using cc_import for shared library on"
              + " Windows");
    }
  }

  private <T extends Object> List<T> asList(Object object, Class<T> type) {
    if (object == null) {
      return ImmutableList.of();
    }
    return ImmutableList.of(type.cast(object));
  }
}
