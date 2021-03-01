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
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector.InstrumentationSpec;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import java.util.Collection;
import java.util.List;

/**
 * Pluggable semantics for Python rules.
 *
 * <p>A new instance of this class is created for each configured target, therefore, it is allowed
 * to keep state.
 */
public interface PythonSemantics {

  /** Returns the URL where documentation for the srcs_version attr lives. */
  String getSrcsVersionDocURL();

  /**
   * Called at the beginning of the analysis of {@code py_binary}, {@code py_test}, and {@code
   * py_library} targets to validate their attributes.
   */
  void validate(RuleContext ruleContext, PyCommon common);

  /**
   * Returns whether we are prohibiting hyphen ('-') characters in the package paths of Python
   * targets and source files.
   */
  boolean prohibitHyphensInPackagePaths();

  /**
   * Extends for the default and data runfiles of {@code py_binary} and {@code py_test} rules with
   * custom elements.
   */
  void collectRunfilesForBinary(
      RuleContext ruleContext, Runfiles.Builder builder, PyCommon common, CcInfo ccInfo)
      throws InterruptedException, RuleErrorException;

  /**
   * Extends the default runfiles of {@code py_binary} and {@code py_test} rules with custom
   * elements.
   */
  void collectDefaultRunfilesForBinary(
      RuleContext ruleContext, PyCommon common, Runfiles.Builder builder)
      throws InterruptedException, RuleErrorException;

  /** Collects a rule's default runfiles. */
  void collectDefaultRunfiles(RuleContext ruleContext, Runfiles.Builder builder);

  /** Returns the coverage instrumentation specification to be used in Python rules. */
  InstrumentationSpec getCoverageInstrumentationSpec();

  /** Utility function to compile multiple .py files to .pyc files, if required. */
  Collection<Artifact> precompiledPythonFiles(
      RuleContext ruleContext, Collection<Artifact> sources, PyCommon common);

  /** Returns a list of PathFragments for the import paths specified in the imports attribute. */
  List<String> getImports(RuleContext ruleContext);

  /** Create a generating action for {@code common.getExecutable()}. */
  void createExecutable(
      RuleContext ruleContext, PyCommon common, CcInfo ccInfo, Runfiles.Builder runfilesBuilder)
      throws InterruptedException, RuleErrorException;

  /**
   * Called at the end of the analysis of {@code py_binary} and {@code py_test} targets.
   *
   * @throws InterruptedException
   */
  void postInitExecutable(
      RuleContext ruleContext,
      RunfilesSupport runfilesSupport,
      PyCommon common,
      RuleConfiguredTargetBuilder builder)
      throws InterruptedException, RuleErrorException;

  CcInfo buildCcInfoProvider(Iterable<? extends TransitiveInfoCollection> deps);

  /**
   * Called when building executables or packages to fill in missing empty __init__.py files if the
   * --incompatible_default_to_explicit_init_py has not yet been enabled. This usually returns a
   * public static final reference, code is free to use that directly on specific implementations
   * instead of making this call.
   */
  Runfiles.EmptyFilesSupplier getEmptyRunfilesSupplier();
}
