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
import com.google.devtools.build.lib.rules.cpp.CcCompilationHelper.CompilationInfo;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.HeadersCheckingMode;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
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
  public abstract static class LibrariesToLinkAndRuntimeArtifact {
    private static LibrariesToLinkAndRuntimeArtifact create(
        @Nullable LibraryToLink staticLibrary,
        @Nullable LibraryToLink dynamicLibraryForLinking,
        @Nullable Artifact runtimeArtifact) {
      return new AutoValue_CcImport_LibrariesToLinkAndRuntimeArtifact(
          staticLibrary, dynamicLibraryForLinking, runtimeArtifact);
    }

    @Nullable
    abstract LibraryToLink staticLibrary();

    @Nullable
    abstract LibraryToLink dynamicLibraryForLinking();

    @Nullable
    abstract Artifact runtimeArtifact();
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

    final CcCommon common = new CcCommon(ruleContext);
    CompilationInfo compilationInfo =
        new CcCompilationHelper(
                ruleContext,
                semantics,
                featureConfiguration,
                ccToolchain,
                ccToolchain.getFdoProvider())
            .addPublicHeaders(common.getHeaders())
            .setHeadersCheckingMode(HeadersCheckingMode.STRICT)
            .compile();

    CcLinkingInfo ccLinkingInfo =
        buildCcLinkingInfo(
            buildLibrariesToLinkAndRuntimeArtifact(
                ruleContext,
                staticLibrary,
                sharedLibrary,
                interfaceLibrary,
                ccToolchain,
                targetWindows));

    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProvider(compilationInfo.getCppDebugFileProvider())
        .addNativeDeclaredProvider(
            CcInfo.builder()
                .setCcCompilationContext(compilationInfo.getCcCompilationContext())
                .setCcLinkingInfo(ccLinkingInfo)
                .build())
        .addSkylarkTransitiveInfo(CcSkylarkApiProvider.NAME, new CcSkylarkApiProvider())
        .addOutputGroups(compilationInfo.getOutputGroups())
        .addProvider(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY))
        .build();
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

  private LibraryToLink buildStaticLibraryToLink(
      RuleContext ruleContext, Artifact staticLibraryArtifact) {
    boolean alwayslink = ruleContext.attributes().get("alwayslink", Type.BOOLEAN);
    ArtifactCategory staticLibraryCategory =
        alwayslink ? ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY : ArtifactCategory.STATIC_LIBRARY;
    return LinkerInputs.opaqueLibraryToLink(
        staticLibraryArtifact,
        staticLibraryCategory,
        CcLinkingOutputs.libraryIdentifierOf(staticLibraryArtifact));
  }

  private LibraryToLink buildSharedLibraryToLink(
      RuleContext ruleContext,
      Artifact sharedLibraryArtifact,
      CcToolchainProvider ccToolchain,
      boolean targetsWindow) {
    if (targetsWindow) {
      return LinkerInputs.opaqueLibraryToLink(
          sharedLibraryArtifact,
          ArtifactCategory.DYNAMIC_LIBRARY,
          CcLinkingOutputs.libraryIdentifierOf(sharedLibraryArtifact));
    } else {
      Artifact dynamicLibrarySymlink =
          SolibSymlinkAction.getDynamicLibrarySymlink(
              /* actionRegistry= */ ruleContext,
              /* actionConstructionContext= */ ruleContext,
              ccToolchain.getSolibDirectory(),
              sharedLibraryArtifact,
              /* preserveName= */ true,
              /* prefixConsumer= */ true,
              ruleContext.getConfiguration());
      return LinkerInputs.solibLibraryToLink(
          dynamicLibrarySymlink,
          sharedLibraryArtifact,
          CcLinkingOutputs.libraryIdentifierOf(sharedLibraryArtifact));
    }
  }

  private LibraryToLink buildInterfaceLibraryToLInk(
      RuleContext ruleContext,
      Artifact interfaceLibraryArtifact,
      CcToolchainProvider ccToolchain,
      boolean targetsWindows) {
    if (targetsWindows) {
      return LinkerInputs.opaqueLibraryToLink(
          interfaceLibraryArtifact,
          ArtifactCategory.INTERFACE_LIBRARY,
          CcLinkingOutputs.libraryIdentifierOf(interfaceLibraryArtifact));
    } else {
      Artifact dynamicLibrarySymlink =
          SolibSymlinkAction.getDynamicLibrarySymlink(
              /* actionRegistry= */ ruleContext,
              /* actionConstructionContext= */ ruleContext,
              ccToolchain.getSolibDirectory(),
              interfaceLibraryArtifact,
              /* preserveName= */ true,
              /* prefixConsumer= */ true,
              ruleContext.getConfiguration());
      return LinkerInputs.solibLibraryToLink(
          dynamicLibrarySymlink,
          interfaceLibraryArtifact,
          CcLinkingOutputs.libraryIdentifierOf(interfaceLibraryArtifact));
    }
  }

  private LibrariesToLinkAndRuntimeArtifact buildLibrariesToLinkAndRuntimeArtifact(
      RuleContext ruleContext,
      Artifact staticLibraryArtifact,
      Artifact sharedLibraryArtifact,
      Artifact interfaceLibraryArtifact,
      CcToolchainProvider ccToolchain,
      boolean targetsWindows) {
    LibraryToLink staticLibrary = null;
    if (staticLibraryArtifact != null) {
      staticLibrary = buildStaticLibraryToLink(ruleContext, staticLibraryArtifact);
    }

    LibraryToLink sharedLibrary = null;
    Artifact runtimeArtifact = null;
    if (sharedLibraryArtifact != null) {
      sharedLibrary =
          buildSharedLibraryToLink(ruleContext, sharedLibraryArtifact, ccToolchain, targetsWindows);
      runtimeArtifact = sharedLibrary.getArtifact();
    }

    LibraryToLink interfaceLibrary = null;
    if (interfaceLibraryArtifact != null) {
      interfaceLibrary =
          buildInterfaceLibraryToLInk(
              ruleContext, interfaceLibraryArtifact, ccToolchain, targetsWindows);
    }

    LibraryToLink dynamicLibraryForLinking;
    if (interfaceLibrary != null) {
      dynamicLibraryForLinking = interfaceLibrary;
    } else {
      dynamicLibraryForLinking = sharedLibrary;
    }

    return LibrariesToLinkAndRuntimeArtifact.create(
        staticLibrary, dynamicLibraryForLinking, runtimeArtifact);
  }

  private <T extends Object> List<T> asList(Object object, Class<T> type) {
    if (object == null) {
      return ImmutableList.of();
    }
    return ImmutableList.of(type.cast(object));
  }

  private CcLinkingInfo buildCcLinkingInfo(
      LibrariesToLinkAndRuntimeArtifact librariesToLinkAndRuntimeArtifact) {
    LibraryToLink staticLibrary = librariesToLinkAndRuntimeArtifact.staticLibrary();
    LibraryToLink dynamicLibraryForLinking =
        librariesToLinkAndRuntimeArtifact.dynamicLibraryForLinking();
    Artifact runtimeArtifact = librariesToLinkAndRuntimeArtifact.runtimeArtifact();

    CcLinkParams.Builder staticModeParamsForExecutable = CcLinkParams.builder();
    CcLinkParams.Builder staticModeParamsForDynamicLibrary = CcLinkParams.builder();
    if (staticLibrary != null) {
      staticModeParamsForExecutable.addLibraries(asList(staticLibrary, LibraryToLink.class));
      staticModeParamsForDynamicLibrary.addLibraries(asList(staticLibrary, LibraryToLink.class));
    } else {
      staticModeParamsForExecutable
          .addLibraries(asList(dynamicLibraryForLinking, LibraryToLink.class))
          .addDynamicLibrariesForRuntime(asList(runtimeArtifact, Artifact.class));
      staticModeParamsForDynamicLibrary
          .addLibraries(asList(dynamicLibraryForLinking, LibraryToLink.class))
          .addDynamicLibrariesForRuntime(asList(runtimeArtifact, Artifact.class));
    }

    CcLinkParams.Builder dynamicModeParamsForExecutable = CcLinkParams.builder();
    CcLinkParams.Builder dynamicModeParamsForDynamicLibrary = CcLinkParams.builder();
    if (dynamicLibraryForLinking != null) {
      dynamicModeParamsForExecutable
          .addLibraries(asList(dynamicLibraryForLinking, LibraryToLink.class))
          .addDynamicLibrariesForRuntime(asList(runtimeArtifact, Artifact.class));
      dynamicModeParamsForDynamicLibrary
          .addLibraries(asList(dynamicLibraryForLinking, LibraryToLink.class))
          .addDynamicLibrariesForRuntime(asList(runtimeArtifact, Artifact.class));
    } else if (staticLibrary != null) {
      dynamicModeParamsForExecutable.addLibraries(asList(staticLibrary, LibraryToLink.class));
      dynamicModeParamsForDynamicLibrary.addLibraries(asList(staticLibrary, LibraryToLink.class));
    }

    return CcLinkingInfo.Builder.create()
        .setStaticModeParamsForExecutable(staticModeParamsForExecutable.build())
        .setStaticModeParamsForDynamicLibrary(staticModeParamsForDynamicLibrary.build())
        .setDynamicModeParamsForExecutable(dynamicModeParamsForExecutable.build())
        .setDynamicModeParamsForDynamicLibrary(dynamicModeParamsForDynamicLibrary.build())
        .build();
  }
}
