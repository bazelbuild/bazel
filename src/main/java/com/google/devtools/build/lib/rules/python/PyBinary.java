// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.python;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.cpp.CcCommon.CcFlagsSupplier;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.syntax.Type;
import java.util.ArrayList;
import java.util.List;

/**
 * An implementation for the {@code py_binary} rule.
 */
public abstract class PyBinary implements RuleConfiguredTargetFactory {
  /**
   * Create a {@link PythonSemantics} object that governs
   * the behavior of this rule.
   */
  protected abstract PythonSemantics createSemantics();

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    PyCommon common = new PyCommon(ruleContext);
    common.initCommon(common.getDefaultPythonVersion());

    RuleConfiguredTargetBuilder builder = init(ruleContext, createSemantics(), common);
    if (builder == null) {
      return null;
    }
    return builder.build();
  }

  static RuleConfiguredTargetBuilder init(
      RuleContext ruleContext, PythonSemantics semantics, PyCommon common)
      throws InterruptedException, RuleErrorException {
    ruleContext.initConfigurationMakeVariableContext(new CcFlagsSupplier(ruleContext));

    List<Artifact> srcs = common.validateSrcs();
    List<Artifact> allOutputs =
        new ArrayList<>(semantics.precompiledPythonFiles(ruleContext, srcs, common));
    if (ruleContext.hasErrors()) {
      return null;
    }

    common.initBinary(allOutputs);
    semantics.validate(ruleContext, common);
    if (ruleContext.hasErrors()) {
      return null;
    }

    NestedSet<String> imports = common.collectImports(ruleContext, semantics);
    if (ruleContext.hasErrors()) {
      return null;
    }

    CcInfo ccInfo =
        semantics.buildCcInfoProvider(ruleContext.getPrerequisites("deps", Mode.TARGET));

    Runfiles commonRunfiles = collectCommonRunfiles(ruleContext, common, semantics, ccInfo);

    Runfiles.Builder defaultRunfilesBuilder = new Runfiles.Builder(
        ruleContext.getWorkspaceName(), ruleContext.getConfiguration().legacyExternalRunfiles())
        .merge(commonRunfiles);
    semantics.collectDefaultRunfilesForBinary(ruleContext, defaultRunfilesBuilder);
    Runfiles defaultRunfiles = defaultRunfilesBuilder.build();

    RunfilesSupport runfilesSupport =
        RunfilesSupport.withExecutable(
            ruleContext,
            defaultRunfiles,
            common.getExecutable());

    if (ruleContext.hasErrors()) {
      return null;
    }

    Runfiles dataRunfiles;
    if (ruleContext.getFragment(PythonConfiguration.class).buildTransitiveRunfilesTrees()) {
      // Only include common runfiles and middleman. Default runfiles added by semantics are
      // excluded. The middleman is necessary to ensure the runfiles trees are generated for all
      // dependency binaries.
      dataRunfiles =
          new Runfiles.Builder(
                  ruleContext.getWorkspaceName(),
                  ruleContext.getConfiguration().legacyExternalRunfiles())
              .merge(commonRunfiles)
              .addLegacyExtraMiddleman(runfilesSupport.getRunfilesMiddleman())
              .build();
    } else {
      dataRunfiles = commonRunfiles;
    }

    RunfilesProvider runfilesProvider = RunfilesProvider.withData(defaultRunfiles, dataRunfiles);

    RuleConfiguredTargetBuilder builder =
        new RuleConfiguredTargetBuilder(ruleContext);
    common.addCommonTransitiveInfoProviders(builder, semantics, common.getFilesToBuild(), imports);

    semantics.postInitBinary(ruleContext, runfilesSupport, common);

    Artifact realExecutable = semantics.createExecutable(ruleContext, common, ccInfo, imports);

    return builder
        .setFilesToBuild(common.getFilesToBuild())
        .add(RunfilesProvider.class, runfilesProvider)
        .setRunfilesSupport(runfilesSupport, realExecutable)
        .addNativeDeclaredProvider(ccInfo)
        .add(PythonImportsProvider.class, new PythonImportsProvider(imports));
  }

  private static Runfiles collectCommonRunfiles(
      RuleContext ruleContext, PyCommon common, PythonSemantics semantics, CcInfo ccInfo)
      throws InterruptedException {
    Runfiles.Builder builder = new Runfiles.Builder(
        ruleContext.getWorkspaceName(), ruleContext.getConfiguration().legacyExternalRunfiles());
    builder.addArtifact(common.getExecutable());
    if (common.getConvertedFiles() != null) {
      builder.addSymlinks(common.getConvertedFiles());
    } else {
      builder.addTransitiveArtifacts(common.getFilesToBuild());
    }
    semantics.collectDefaultRunfiles(ruleContext, builder);
    builder.add(ruleContext, PythonRunfilesProvider.TO_RUNFILES);

    if (!ruleContext.attributes().has("legacy_create_init", Type.BOOLEAN)
        || ruleContext.attributes().get("legacy_create_init", Type.BOOLEAN)) {
      builder.setEmptyFilesSupplier(PythonUtils.GET_INIT_PY_FILES);
    }
    semantics.collectRunfilesForBinary(ruleContext, builder, common, ccInfo);
    return builder.build();
  }
}
