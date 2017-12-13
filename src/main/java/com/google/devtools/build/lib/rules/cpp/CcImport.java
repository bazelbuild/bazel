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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.HeadersCheckingMode;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * A ConfiguredTarget for <code>cc_import</code> rule.
 */
public abstract class CcImport implements RuleConfiguredTargetFactory {
  private final CppSemantics semantics;

  protected CcImport(CppSemantics semantics) {
    this.semantics = semantics;
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws RuleErrorException, InterruptedException {
    Artifact staticLibrary = ruleContext.getPrerequisiteArtifact("static_library", Mode.TARGET);
    Artifact sharedLibrary = ruleContext.getPrerequisiteArtifact("shared_library", Mode.TARGET);
    Artifact interfaceLibrary =
        ruleContext.getPrerequisiteArtifact("interface_library", Mode.TARGET);

    boolean systemProvided = ruleContext.attributes().get("system_provided", Type.BOOLEAN);
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

    // Create CcLibraryHelper
    CcToolchainProvider ccToolchain =
        CppHelper.getToolchainUsingDefaultCcToolchainAttribute(ruleContext);
    FeatureConfiguration featureConfiguration =
        CcCommon.configureFeatures(ruleContext, ccToolchain);
    FdoSupportProvider fdoSupport =
        CppHelper.getFdoSupportUsingDefaultCcToolchainAttribute(ruleContext);
    CcLibraryHelper helper =
        new CcLibraryHelper(ruleContext, semantics, featureConfiguration, ccToolchain, fdoSupport);

    // Add headers
    final CcCommon common = new CcCommon(ruleContext);
    helper.addPublicHeaders(common.getHeaders());
    helper.setHeadersCheckingMode(HeadersCheckingMode.STRICT);

    // Get alwayslink attribute
    boolean alwayslink = ruleContext.attributes().get("alwayslink", Type.BOOLEAN);
    ArtifactCategory staticLibraryCategory =
        alwayslink ? ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY : ArtifactCategory.STATIC_LIBRARY;

    PathFragment labelName = PathFragment.create(ruleContext.getLabel().getName());
    String libraryIdentifier =
        ruleContext
            .getPackageDirectory()
            .getRelative(labelName.replaceName("lib" + labelName.getBaseName()))
            .getPathString();

    if (staticLibrary != null) {
      if (CppFileTypes.PIC_ARCHIVE.matches(staticLibrary.getPath())) {
        helper.addPicStaticLibraries(
            ImmutableList.of(
                LinkerInputs.opaqueLibraryToLink(
                    staticLibrary, staticLibraryCategory, libraryIdentifier, alwayslink)));
      } else {
        helper.addStaticLibraries(
            ImmutableList.of(
                LinkerInputs.opaqueLibraryToLink(
                    staticLibrary, staticLibraryCategory, libraryIdentifier, alwayslink)));
      }
    }

    // Now we are going to have some platform dependent behaviors
    boolean targetWindows = featureConfiguration.isEnabled(CppRuleClasses.TARGETS_WINDOWS);

    Iterable<LibraryToLink> dynamicLibraryList = null;
    Iterable<LibraryToLink> executionDynamicLibraryList = null;
    if (sharedLibrary != null) {
      if (targetWindows) {
        executionDynamicLibraryList =
            ImmutableList.of(
                LinkerInputs.opaqueLibraryToLink(
                    sharedLibrary, ArtifactCategory.DYNAMIC_LIBRARY, libraryIdentifier));
      } else {
        executionDynamicLibraryList =
                ImmutableList.of(
                    LinkerInputs.solibLibraryToLink(
                        common.getDynamicLibrarySymlink(sharedLibrary, true),
                        sharedLibrary,
                        libraryIdentifier));
      }
      helper.addExecutionDynamicLibraries(executionDynamicLibraryList);
    }

    if (interfaceLibrary != null) {
      if (targetWindows) {
        dynamicLibraryList =
            ImmutableList.of(
                LinkerInputs.opaqueLibraryToLink(
                    interfaceLibrary, ArtifactCategory.INTERFACE_LIBRARY, libraryIdentifier));
      } else {
        dynamicLibraryList =
                ImmutableList.of(
                    LinkerInputs.solibLibraryToLink(
                        common.getDynamicLibrarySymlink(interfaceLibrary, true),
                        interfaceLibrary,
                        libraryIdentifier));
      }
    } else {
      // If interface_library is not specified and we are not building for Windows, then the dynamic
      // library required at linking time is the same as the one required at execution time.
      if (!targetWindows) {
        dynamicLibraryList = executionDynamicLibraryList;
      } else if (staticLibrary == null) {
        ruleContext.ruleError(
          "'interface library' must be specified when using cc_import for shared library on"
        + " Windows");
      }
    }

    if (dynamicLibraryList != null) {
      helper.addDynamicLibraries(dynamicLibraryList);
    }

    CcLibraryHelper.Info info = helper.build();

    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProviders(info.getProviders())
        .addSkylarkTransitiveInfo(CcSkylarkApiProvider.NAME, new CcSkylarkApiProvider())
        .addOutputGroups(info.getOutputGroups())
        .addProvider(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY))
        .build();
  }
}
