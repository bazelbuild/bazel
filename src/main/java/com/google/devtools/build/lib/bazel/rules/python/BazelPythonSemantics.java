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


import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.python.PyCommon;
import com.google.devtools.build.lib.rules.python.PythonSemantics;
import com.google.devtools.build.lib.rules.python.PythonUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.Serializable;
import java.util.Collection;
import java.util.List;
import java.util.function.Predicate;

/** Functionality specific to the Python rules in Bazel. */
public class BazelPythonSemantics implements PythonSemantics {

  public static final Runfiles.EmptyFilesSupplier GET_INIT_PY_FILES =
      new PythonUtils.GetInitPyFiles((Predicate<PathFragment> & Serializable) source -> false);

  @Override
  public Runfiles.EmptyFilesSupplier getEmptyRunfilesSupplier() {
    return GET_INIT_PY_FILES;
  }

  @Override
  public String getSrcsVersionDocURL() {
    throw new UnsupportedOperationException("Should not be called");
  }

  @Override
  public void validate(RuleContext ruleContext, PyCommon common) {
    throw new UnsupportedOperationException("Should not be called");
  }

  @Override
  public boolean prohibitHyphensInPackagePaths() {
    throw new UnsupportedOperationException("Should not be called");
  }

  @Override
  public void collectRunfilesForBinary(
      RuleContext ruleContext, Runfiles.Builder builder, PyCommon common, CcInfo ccInfo) {
    throw new UnsupportedOperationException("Should not be called");
  }

  @Override
  public void collectDefaultRunfilesForBinary(
      RuleContext ruleContext, PyCommon common, Runfiles.Builder builder) {
    throw new UnsupportedOperationException("Should not be called");
  }

  @Override
  public Collection<Artifact> precompiledPythonFiles(
      RuleContext ruleContext, Collection<Artifact> sources, PyCommon common) {
    throw new UnsupportedOperationException("Should not be called");
  }

  @Override
  public List<String> getImports(RuleContext ruleContext) {
    throw new UnsupportedOperationException("Should not be called");
  }

  @Override
  public void createExecutable(
      RuleContext ruleContext, PyCommon common, CcInfo ccInfo, Runfiles.Builder runfilesBuilder) {
    throw new UnsupportedOperationException("Should not be called");
  }

  @Override
  public void postInitExecutable(
      RuleContext ruleContext,
      RunfilesSupport runfilesSupport,
      PyCommon common,
      RuleConfiguredTargetBuilder builder) {
    throw new UnsupportedOperationException("Should not be called");
  }

  @Override
  public CcInfo buildCcInfoProvider(
      RuleContext ruleContext, Iterable<? extends TransitiveInfoCollection> deps) {
    throw new UnsupportedOperationException("Should not be called");
  }
}
