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
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.cpp.CcLinkParams;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsProvider;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsStore;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.List;

/**
 * An implementation for the {@code py_library} rule.
 */
public abstract class PyLibrary implements RuleConfiguredTargetFactory {

  /**
   * Create a {@link PythonSemantics} object that governs
   * the behavior of this rule.
   */
  protected abstract PythonSemantics createSemantics();

  @Override
  public ConfiguredTarget create(final RuleContext ruleContext) {
    PythonSemantics semantics = createSemantics();
    PyCommon common = new PyCommon(ruleContext);
    common.initCommon(common.getDefaultPythonVersion());
    common.validatePackageName();
    semantics.validate(ruleContext, common);

    List<Artifact> srcs = common.validateSrcs();
    List<Artifact> allOutputs =
        new ArrayList<>(semantics.precompiledPythonFiles(ruleContext, srcs, common));

    NestedSet<Artifact> filesToBuild =
        NestedSetBuilder.wrap(Order.STABLE_ORDER, allOutputs);
    common.addPyExtraActionPseudoAction();

    CcLinkParamsStore ccLinkParamsStore = new CcLinkParamsStore() {
      @Override
      protected void collect(CcLinkParams.Builder builder, boolean linkingStatically,
                             boolean linkShared) {
        builder.addTransitiveTargets(ruleContext.getPrerequisites("deps", Mode.TARGET),
            PyCcLinkParamsProvider.TO_LINK_PARAMS,
            CcLinkParamsProvider.TO_LINK_PARAMS);
      }
    };

    NestedSet<PathFragment> imports = common.collectImports(ruleContext, semantics);
    if (ruleContext.hasErrors()) {
      return null;
    }

    Runfiles.Builder runfilesBuilder = new Runfiles.Builder(ruleContext.getWorkspaceName());
    if (common.getConvertedFiles() != null) {
      runfilesBuilder.addSymlinks(common.getConvertedFiles());
    } else {
      runfilesBuilder.addTransitiveArtifacts(filesToBuild);
    }
    runfilesBuilder.setEmptyFilesSupplier(PythonUtils.GET_INIT_PY_FILES);
    runfilesBuilder.add(ruleContext, PythonRunfilesProvider.TO_RUNFILES);
    runfilesBuilder.addRunfiles(ruleContext, RunfilesProvider.DEFAULT_RUNFILES);

    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext);
    common.addCommonTransitiveInfoProviders(builder, semantics, filesToBuild);
    return builder
        .setFilesToBuild(filesToBuild)
        .add(RunfilesProvider.class, RunfilesProvider.simple(runfilesBuilder.build()))
        .add(CcLinkParamsProvider.class, new CcLinkParamsProvider(ccLinkParamsStore))
        .add(PythonImportsProvider.class, new PythonImportsProvider(imports))
        .build();
  }
}
