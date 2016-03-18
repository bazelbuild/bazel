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
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.cpp.CcLinkParams;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsProvider;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsStore;
import com.google.devtools.build.lib.vfs.PathFragment;

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
  public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    PyCommon common = new PyCommon(ruleContext);
    common.initCommon(common.getDefaultPythonVersion());

    RuleConfiguredTargetBuilder builder = init(ruleContext, createSemantics(), common);
    if (builder == null) {
      return null;
    }
    return builder.build();
  }

  static RuleConfiguredTargetBuilder init(RuleContext ruleContext, PythonSemantics semantics,
      PyCommon common) throws InterruptedException {
    CcLinkParamsStore ccLinkParamsStore = initializeCcLinkParamStore(ruleContext);

    List<Artifact> srcs = common.validateSrcs();
    List<Artifact> allOutputs =
        new ArrayList<>(semantics.precompiledPythonFiles(ruleContext, srcs, common));

    common.initBinary(allOutputs);
    semantics.validate(ruleContext, common);
    if (ruleContext.hasErrors()) {
      return null;
    }

    NestedSet<PathFragment> imports = common.collectImports(ruleContext, semantics);
    if (ruleContext.hasErrors()) {
      return null;
    }

    semantics.createExecutable(ruleContext, common, ccLinkParamsStore, imports);
    Runfiles commonRunfiles = collectCommonRunfiles(ruleContext, common, semantics);

    Runfiles.Builder defaultRunfilesBuilder = new Runfiles.Builder(ruleContext.getWorkspaceName())
        .merge(commonRunfiles);
    semantics.collectDefaultRunfilesForBinary(ruleContext, defaultRunfilesBuilder);
    Runfiles defaultRunfiles = defaultRunfilesBuilder.build();

    RunfilesSupport runfilesSupport = RunfilesSupport.withExecutable(ruleContext, defaultRunfiles,
        common.getExecutable(), ruleContext.shouldCreateRunfilesSymlinks());

    if (ruleContext.hasErrors()) {
      return null;
    }

    // Only include common runfiles and middleman. Default runfiles added by semantics are
    // excluded. The middleman is necessary to ensure the runfiles trees are generated for all
    // dependency binaries.
    Runfiles dataRunfiles = new Runfiles.Builder(ruleContext.getWorkspaceName())
        .merge(commonRunfiles)
        .addArtifact(runfilesSupport.getRunfilesMiddleman())
        .build();

    RunfilesProvider runfilesProvider = RunfilesProvider.withData(defaultRunfiles, dataRunfiles);

    RuleConfiguredTargetBuilder builder =
        new RuleConfiguredTargetBuilder(ruleContext);
    common.addCommonTransitiveInfoProviders(builder, semantics, common.getFilesToBuild());

    semantics.postInitBinary(ruleContext, runfilesSupport, common);
    return builder
        .setFilesToBuild(common.getFilesToBuild())
        .add(RunfilesProvider.class, runfilesProvider)
        .setRunfilesSupport(runfilesSupport, common.getExecutable())
        .add(CcLinkParamsProvider.class, new CcLinkParamsProvider(ccLinkParamsStore))
        .add(PythonImportsProvider.class, new PythonImportsProvider(imports));
  }

  private static Runfiles collectCommonRunfiles(RuleContext ruleContext, PyCommon common,
      PythonSemantics semantics) {
    Runfiles.Builder builder = new Runfiles.Builder(ruleContext.getWorkspaceName());
    builder.addArtifact(common.getExecutable());
    if (common.getConvertedFiles() != null) {
      builder.addSymlinks(common.getConvertedFiles());
    } else {
      builder.addTransitiveArtifacts(common.getFilesToBuild());
    }
    builder.addRunfiles(ruleContext, RunfilesProvider.DEFAULT_RUNFILES);
    builder.add(ruleContext, PythonRunfilesProvider.TO_RUNFILES);
    builder.setEmptyFilesSupplier(PythonUtils.GET_INIT_PY_FILES);
    semantics.collectRunfilesForBinary(ruleContext, builder, common);
    return builder.build();
  }

  private static CcLinkParamsStore initializeCcLinkParamStore(final RuleContext ruleContext) {
    return new CcLinkParamsStore() {
      @Override
      protected void collect(CcLinkParams.Builder builder, boolean linkingStatically,
                             boolean linkShared) {
        builder.addTransitiveTargets(ruleContext.getPrerequisites("deps", Mode.TARGET),
            PyCcLinkParamsProvider.TO_LINK_PARAMS,
            CcLinkParamsProvider.TO_LINK_PARAMS);
      }
    };
  }
}
