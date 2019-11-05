// Copyright 2018 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.rules.cpp.CcCommon.CcFlagsSupplier;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import java.util.ArrayList;
import java.util.List;

/** Common implementation logic for {@code py_binary} and {@code py_test}. */
public abstract class PyExecutable implements RuleConfiguredTargetFactory {

  /**
   * Creates a pluggable semantics object to be used for the analysis of a target of this rule type.
   */
  protected abstract PythonSemantics createSemantics();

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    // Init the make variable context first. Otherwise it may be incorrectly initialized by default
    // inside semantics/common via {@link RuleContext#getExpander}.
    ruleContext.initConfigurationMakeVariableContext(new CcFlagsSupplier(ruleContext));

    PythonSemantics semantics = createSemantics();
    PyCommon common = new PyCommon(ruleContext, semantics);

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

    CcInfo ccInfo =
        semantics.buildCcInfoProvider(ruleContext.getPrerequisites("deps", Mode.TARGET));

    Runfiles commonRunfiles = collectCommonRunfiles(ruleContext, common, semantics, ccInfo);

    Runfiles.Builder defaultRunfilesBuilder = new Runfiles.Builder(
        ruleContext.getWorkspaceName(), ruleContext.getConfiguration().legacyExternalRunfiles())
        .merge(commonRunfiles);
    semantics.collectDefaultRunfilesForBinary(ruleContext, common, defaultRunfilesBuilder);

    common.createExecutable(ccInfo, defaultRunfilesBuilder);

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
    common.addCommonTransitiveInfoProviders(builder, common.getFilesToBuild());

    semantics.postInitExecutable(ruleContext, runfilesSupport, common, builder);

    return builder
        .setFilesToBuild(common.getFilesToBuild())
        .add(RunfilesProvider.class, runfilesProvider)
        .setRunfilesSupport(runfilesSupport, common.getExecutable())
        .addNativeDeclaredProvider(new PyCcLinkParamsProvider(ccInfo))
        .build();
  }

  /**
   * If requested, creates empty __init__.py files for each manifest file.
   *
   * <p>We do this if the rule defines {@code legacy_create_init} and its value is true. Auto is
   * treated as false iff {@code --incompatible_default_to_explicit_init_py} is given.
   *
   * <p>See {@link PythonUtils#getInitPyFiles} for details about how the files are created.
   */
  private static void maybeCreateInitFiles(RuleContext ruleContext, Runfiles.Builder builder) {
    boolean createFiles;
    if (!ruleContext.attributes().has("legacy_create_init", BuildType.TRISTATE)) {
      createFiles = true;
    } else {
      TriState legacy = ruleContext.attributes().get("legacy_create_init", BuildType.TRISTATE);
      if (legacy == TriState.AUTO) {
        createFiles = !ruleContext.getFragment(PythonConfiguration.class).defaultToExplicitInitPy();
      } else {
        createFiles = legacy != TriState.NO;
      }
    }
    if (createFiles) {
      builder.setEmptyFilesSupplier(PythonUtils.GET_INIT_PY_FILES);
    }
  }

  private static Runfiles collectCommonRunfiles(
      RuleContext ruleContext, PyCommon common, PythonSemantics semantics, CcInfo ccInfo)
      throws InterruptedException, RuleErrorException {
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

    maybeCreateInitFiles(ruleContext, builder);

    semantics.collectRunfilesForBinary(ruleContext, builder, common, ccInfo);
    return builder.build();
  }
}
