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
package com.google.devtools.build.lib.rules.python;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.cpp.CcLinkParams;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsProvider;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsStore;

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
  public ConfiguredTarget create(RuleContext ruleContext) {
    PyCommon common = new PyCommon(ruleContext);
    common.initCommon(common.getDefaultPythonVersion());

    RuleConfiguredTargetBuilder builder = init(ruleContext, createSemantics(), common);
    if (builder == null) {
      return null;
    }
    return builder.build();
  }

  static RuleConfiguredTargetBuilder init(
      RuleContext ruleContext, PythonSemantics semantics, PyCommon common) {
    List<Artifact> srcs = common.validateSrcs();
    CcLinkParamsStore ccLinkParamsStore = initializeCcLinkParamStore(ruleContext);

    common.initBinary(srcs);
    semantics.validate(ruleContext, common);
    if (ruleContext.hasErrors()) {
      return null;
    }

    semantics.createExecutable(ruleContext, common, ccLinkParamsStore);
    Runfiles.Builder runfilesBuilder = collectCommonRunfiles(ruleContext, common);
    semantics.collectRunfilesForBinary(ruleContext, runfilesBuilder, common);
    Runfiles dataRunfiles = runfilesBuilder.build();
    semantics.collectDefaultRunfilesForBinary(ruleContext, runfilesBuilder);
    Runfiles defaultRunfiles = runfilesBuilder.build();

    RunfilesSupport runfilesSupport = RunfilesSupport.withExecutable(ruleContext, defaultRunfiles,
        common.getExecutable(), ruleContext.shouldCreateRunfilesSymlinks());

    if (ruleContext.hasErrors()) {
      return null;
    }

    RunfilesProvider runfilesProvider = RunfilesProvider.withData(defaultRunfiles, dataRunfiles);

    RuleConfiguredTargetBuilder builder =
        new RuleConfiguredTargetBuilder(ruleContext);
    common.addCommonTransitiveInfoProviders(builder, semantics, common.getFilesToBuild());

    semantics.postInitBinary(ruleContext, runfilesSupport, common);
    return builder
        .setFilesToBuild(common.getFilesToBuild())
        .add(RunfilesProvider.class, runfilesProvider)
        .setRunfilesSupport(runfilesSupport, common.getExecutable())
        .add(CcLinkParamsProvider.class, new CcLinkParamsProvider(ccLinkParamsStore));
  }

  private static Runfiles.Builder collectCommonRunfiles(RuleContext ruleContext, PyCommon common) {
    Runfiles.Builder builder = new Runfiles.Builder();
    builder.addArtifact(common.getExecutable());
    if (common.getConvertedFiles() != null) {
      builder.addSymlinks(common.getConvertedFiles());
    } else {
      builder.addTransitiveArtifacts(common.getFilesToBuild());
    }
    builder.addRunfiles(ruleContext, RunfilesProvider.DEFAULT_RUNFILES);
    builder.add(ruleContext, PythonRunfilesProvider.TO_RUNFILES);
    builder.setEmptyFilesSupplier(PythonUtils.GET_INIT_PY_FILES);
    return builder;
  }

  private static CcLinkParamsStore initializeCcLinkParamStore(final RuleContext ruleContext) {
    return new CcLinkParamsStore() {
      @Override
      protected void collect(CcLinkParams.Builder builder, boolean linkingStatically,
                             boolean linkShared) {
        Iterable<? extends TransitiveInfoCollection> deps =
            ruleContext.getPrerequisites("deps", Mode.TARGET);
        builder.addTransitiveTargets(deps);
        builder.addTransitiveLangTargets(deps, PyCcLinkParamsProvider.TO_LINK_PARAMS);
      }
    };
  }
}

