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

package com.google.devtools.build.lib.bazel.rules.python;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles.Builder;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Substitution;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Template;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsStore;
import com.google.devtools.build.lib.rules.python.PyCommon;
import com.google.devtools.build.lib.rules.python.PythonSemantics;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesCollector.InstrumentationSpec;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Functionality specific to the Python rules in Bazel.
 */
public class BazelPythonSemantics implements PythonSemantics {
  private static final Template STUB_TEMPLATE =
      Template.forResource(BazelPythonSemantics.class, "stub_template.txt");
  public static final InstrumentationSpec PYTHON_COLLECTION_SPEC = new InstrumentationSpec(
      FileTypeSet.of(BazelPyRuleClasses.PYTHON_SOURCE),
      "srcs", "deps", "data");

  @Override
  public void validate(RuleContext ruleContext, PyCommon common) {
  }

  @Override
  public void collectRunfilesForBinary(RuleContext ruleContext, Builder builder, PyCommon common) {
  }

  @Override
  public void collectDefaultRunfilesForBinary(RuleContext ruleContext, Builder builder) {
  }

  @Override
  public InstrumentationSpec getCoverageInstrumentationSpec() {
    return PYTHON_COLLECTION_SPEC;
  }

  @Override
  public Collection<Artifact> precompiledPythonFiles(
      RuleContext ruleContext, Collection<Artifact> sources, PyCommon common) {
    return ImmutableList.copyOf(sources);
  }

  @Override
  public List<PathFragment> getImports(RuleContext ruleContext) {
    List<PathFragment> result = new ArrayList<>();
    PathFragment packageFragment = ruleContext.getLabel().getPackageIdentifier().getPathFragment();
    for (String importsAttr : ruleContext.attributes().get("imports", Type.STRING_LIST)) {
      importsAttr = ruleContext.expandMakeVariables("includes", importsAttr);
      if (importsAttr.startsWith("/")) {
        ruleContext.attributeWarning("imports",
            "ignoring invalid absolute path '" + importsAttr + "'");
        continue;
      }
      PathFragment importsPath = packageFragment.getRelative(importsAttr).normalize();
      if (!importsPath.isNormalized()) {
        ruleContext.attributeError("imports",
            "Path references a path above the execution root.");
      }
      result.add(importsPath);
    }
    return result;
  }

  @Override
  public void createExecutable(RuleContext ruleContext, PyCommon common,
      CcLinkParamsStore ccLinkParamsStore, NestedSet<PathFragment> imports) {
    String main = common.determineMainExecutableSource();
    BazelPythonConfiguration config = ruleContext.getFragment(BazelPythonConfiguration.class);
    String pythonBinary;

    switch (common.getVersion()) {
      case PY2: pythonBinary = config.getPython2Path(); break;
      case PY3: pythonBinary = config.getPython3Path(); break;
      default: throw new IllegalStateException();
    }

    ruleContext.registerAction(new TemplateExpansionAction(
        ruleContext.getActionOwner(),
        common.getExecutable(),
        STUB_TEMPLATE,
        ImmutableList.of(
            Substitution.of("%main%", main),
            Substitution.of("%python_binary%", pythonBinary),
            Substitution.of("%imports%", Joiner.on(":").join(imports))),
        true));
  }

  @Override
  public void postInitBinary(RuleContext ruleContext, RunfilesSupport runfilesSupport,
      PyCommon common) {
  }
}
