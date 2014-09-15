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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.devtools.build.lib.packages.Type.BOOLEAN;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.License;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.CompilationHelper;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.FileProvider;
import com.google.devtools.build.lib.view.LicensesProvider;
import com.google.devtools.build.lib.view.LicensesProvider.TargetLicense;
import com.google.devtools.build.lib.view.MiddlemanProvider;
import com.google.devtools.build.lib.view.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.view.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.Runfiles;
import com.google.devtools.build.lib.view.RunfilesProvider;
import com.google.devtools.build.lib.view.TransitiveInfoCollection;

import java.util.List;

/**
 * Implementation for the cc_toolchain rule.
 */
public class CcToolchain implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) {
    final Label label = ruleContext.getLabel();
    final NestedSet<Artifact> crosstool = ruleContext.getPrerequisite("all_files", Mode.HOST)
        .getProvider(FileProvider.class).getFilesToBuild();
    final NestedSet<Artifact> crosstoolMiddleman = getFiles(ruleContext, "all_files");
    final NestedSet<Artifact> compile = getFiles(ruleContext, "compiler_files");
    final NestedSet<Artifact> strip = getFiles(ruleContext, "strip_files");
    final NestedSet<Artifact> objcopy = getFiles(ruleContext, "objcopy_files");
    final NestedSet<Artifact> link = getFiles(ruleContext, "linker_files");
    final NestedSet<Artifact> dwp = getFiles(ruleContext, "dwp_files");
    String purposePrefix = Actions.escapeLabel(label) + "_";
    String runtimeSolibDirBase = "_solib_" + "_" + Actions.escapeLabel(label);
    final PathFragment runtimeSolibDir = ruleContext.getConfiguration()
        .getBinFragment().getRelative(runtimeSolibDirBase);
    TransitiveInfoCollection staticRuntimeLibDep = selectDep(ruleContext, "static_runtime_libs",
        ruleContext.getConfiguration().getFragment(CppConfiguration.class)
            .getStaticRuntimeLibsLabel());

    final NestedSet<Artifact> staticRuntimeLinkInputs = staticRuntimeLibDep
        .getProvider(FileProvider.class)
        .getFilesToBuild();

    NestedSet<Artifact> staticRuntimeLinkMiddlemanSet = CompilationHelper.getAggregatingMiddleman(
        ruleContext,
        purposePrefix + "static_runtime_link",
        staticRuntimeLibDep);

    TransitiveInfoCollection dynamicRuntimeLibDep = selectDep(ruleContext, "dynamic_runtime_libs",
        ruleContext.getConfiguration().getFragment(CppConfiguration.class)
            .getDynamicRuntimeLibsLabel());
    NestedSetBuilder<Artifact> dynamicRuntimeLinkInputsBuilder = NestedSetBuilder.stableOrder();
    for (Artifact artifact : dynamicRuntimeLibDep
        .getProvider(FileProvider.class).getFilesToBuild()) {
      if (CppHelper.SHARED_LIBRARY_FILETYPES.matches(artifact.getFilename())) {
        dynamicRuntimeLinkInputsBuilder.add(SolibSymlinkAction.getCppRuntimeSymlink(
            ruleContext, artifact, runtimeSolibDirBase,
            ruleContext.getConfiguration()).getArtifact());
      } else {
        dynamicRuntimeLinkInputsBuilder.add(artifact);
      }
    }

    final NestedSet<Artifact> dynamicRuntimeLinkInputs = dynamicRuntimeLinkInputsBuilder.build();

    List<Artifact> dynamicRuntimeLinkMiddlemanSet =
        CppHelper.getAggregatingMiddlemanForCppRuntimes(
            ruleContext,
            purposePrefix + "dynamic_runtime_link",
            dynamicRuntimeLibDep,
            runtimeSolibDirBase,
            ruleContext.getConfiguration());
    final Artifact staticRuntimeLinkMiddleman = staticRuntimeLinkMiddlemanSet.isEmpty()
        ? null : Iterables.getOnlyElement(staticRuntimeLinkMiddlemanSet);
    final Artifact dynamicRuntimeLinkMiddleman = dynamicRuntimeLinkMiddlemanSet.isEmpty()
        ? null : Iterables.getOnlyElement(dynamicRuntimeLinkMiddlemanSet);

    CppCompilationContext.Builder contextBuilder =
        new CppCompilationContext.Builder(ruleContext);
    CppModuleMap moduleMap = createCrosstoolModuleMap(ruleContext);
    if (moduleMap != null) {
      contextBuilder.setCppModuleMap(moduleMap);
    }
    final CppCompilationContext context = contextBuilder.build();
    boolean supportsParamFiles = ruleContext.attributes().get("supports_param_files", BOOLEAN);

    CcToolchainProvider provider = new CcToolchainProvider(
        crosstool,
        crosstoolMiddleman,
        compile,
        strip,
        objcopy,
        link,
        dwp,
        staticRuntimeLinkInputs,
        staticRuntimeLinkMiddleman,
        dynamicRuntimeLinkInputs,
        dynamicRuntimeLinkMiddleman,
        runtimeSolibDir,
        context,
        supportsParamFiles);
    RuleConfiguredTargetBuilder builder =
        new RuleConfiguredTargetBuilder(ruleContext)
            .add(CcToolchainProvider.class, provider)
            .setFilesToBuild(new NestedSetBuilder<Artifact>(Order.STABLE_ORDER).build())
            .add(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY));

    // If output_license is specified on the cc_toolchain rule, override the transitive licenses
    // with that one. This is necessary because cc_toolchain is used in the target configuration,
    // but it is sort-of-kind-of a tool, but various parts of it are linked into the output...
    // ...so we trust the judgment of the author of the cc_toolchain rule to figure out what
    // licenses should be propagated to C++ targets.
    License outputLicense = ruleContext.getRule().getToolOutputLicense(ruleContext.attributes());
    if (outputLicense != null && outputLicense != License.NO_LICENSE) {
      final NestedSet<TargetLicense> license = NestedSetBuilder.create(Order.STABLE_ORDER,
          new TargetLicense(ruleContext.getLabel(), outputLicense));
      LicensesProvider licensesProvider = new LicensesProvider() {
        @Override
        public NestedSet<TargetLicense> getTransitiveLicenses() {
          return license;
        }
      };

      builder.add(LicensesProvider.class, licensesProvider);
    }

    return builder.build();
  }

  private CppModuleMap createCrosstoolModuleMap(RuleContext ruleContext) {
    if (ruleContext.getPrerequisite("module_map", Mode.HOST) == null) {
      return null;
    }
    Artifact moduleMapArtifact = ruleContext.getPrerequisiteArtifact("module_map", Mode.HOST);
    if (moduleMapArtifact == null) {
      return null;
    }
    return new CppModuleMap(moduleMapArtifact, "crosstool");
  }

  private TransitiveInfoCollection selectDep(
      RuleContext ruleContext, String attribute, Label label) {
    for (TransitiveInfoCollection dep : ruleContext.getPrerequisites(attribute, Mode.TARGET)) {
      if (dep.getLabel().equals(label)) {
        return dep;
      }
    }

    return ruleContext.getPrerequisites(attribute, Mode.TARGET).get(0);
  }

  private NestedSet<Artifact> getFiles(RuleContext context, String attribute) {
    TransitiveInfoCollection dep = context.getPrerequisite(attribute, Mode.HOST);
    MiddlemanProvider middlemanProvider = dep.getProvider(MiddlemanProvider.class);
    // We use the middleman if we can (if the dep is a filegroup), otherwise, just the regular
    // filesToBuild (e.g. if it is a simple input file)
    return middlemanProvider != null
        ? middlemanProvider.getMiddlemanArtifact()
        : dep.getProvider(FileProvider.class).getFilesToBuild();
  }
}
