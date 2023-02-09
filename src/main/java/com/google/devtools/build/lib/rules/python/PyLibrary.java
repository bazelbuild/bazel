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
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/** Base implementation of {@code py_library}. */
public abstract class PyLibrary implements RuleConfiguredTargetFactory {

  /**
   * Create a {@link PythonSemantics} object that governs
   * the behavior of this rule.
   */
  protected abstract PythonSemantics createSemantics();

  @Override
  @Nullable
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    PythonSemantics semantics = createSemantics();
    PyCommon common =
        new PyCommon(
            ruleContext, semantics, /*validateSources=*/ true, /*requiresMainFile=*/ false);
    semantics.validate(ruleContext, common);

    List<Artifact> srcs = common.getPythonSources();
    List<Artifact> allOutputs =
        new ArrayList<>(semantics.precompiledPythonFiles(ruleContext, srcs, common));
    if (ruleContext.hasErrors()) {
      return null;
    }

    NestedSet<Artifact> filesToBuild =
        NestedSetBuilder.wrap(Order.STABLE_ORDER, allOutputs);
    common.addPyExtraActionPseudoAction();

    Runfiles.Builder runfilesBuilder = new Runfiles.Builder(
        ruleContext.getWorkspaceName(), ruleContext.getConfiguration().legacyExternalRunfiles());
    runfilesBuilder.addTransitiveArtifacts(filesToBuild);
    runfilesBuilder.addRunfiles(ruleContext, RunfilesProvider.DEFAULT_RUNFILES);

    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext);
    common.addCommonTransitiveInfoProviders(builder);

    return builder
        .setFilesToBuild(filesToBuild)
        .addNativeDeclaredProvider(
            new PyCcLinkParamsProvider(
                semantics.buildCcInfoProvider(ruleContext, ruleContext.getPrerequisites("deps"))))
        .add(RunfilesProvider.class, RunfilesProvider.simple(runfilesBuilder.build()))
        .build();
  }
}
