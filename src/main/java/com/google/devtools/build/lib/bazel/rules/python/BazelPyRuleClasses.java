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

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.TRISTATE;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;
import static com.google.devtools.build.lib.syntax.Type.STRING;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCppRuleClasses.CcToolchainRequiringRule;
import com.google.devtools.build.lib.packages.Attribute.AllowedValueSet;
import com.google.devtools.build.lib.packages.Attribute.LabelLateBoundDefault;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.rules.python.PyCommon;
import com.google.devtools.build.lib.rules.python.PyRuleClasses;
import com.google.devtools.build.lib.rules.python.PythonVersion;
import com.google.devtools.build.lib.util.FileType;

/**
 * Bazel-specific rule definitions for Python rules.
 */
public final class BazelPyRuleClasses {
  public static final FileType PYTHON_SOURCE = FileType.of(".py");

  public static final LabelLateBoundDefault<?> PY_INTERPRETER =
      LabelLateBoundDefault.fromTargetConfiguration(
          BazelPythonConfiguration.class,
          null,
          (rule, attributes, bazelPythonConfig) -> bazelPythonConfig.getPythonTop());

  /**
   * Base class for Python rule definitions.
   */
  public static final class PyBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          /* <!-- #BLAZE_RULE($base_py).ATTRIBUTE(deps) -->
          The list of other libraries to be linked in to the binary target.
          See general comments about <code>deps</code> at
          <a href="${link common-definitions#common-attributes}">
          Attributes common to all build rules</a>.
          These can be
          <a href="${link py_binary}"><code>py_binary</code></a> rules,
          <a href="${link py_library}"><code>py_library</code></a> rules.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .override(
              builder
                  .copy("deps")
                  .legacyMandatoryProviders(PyCommon.PYTHON_SKYLARK_PROVIDER_NAME)
                  .allowedFileTypes())
          /* <!-- #BLAZE_RULE($base_py).ATTRIBUTE(imports) -->
          List of import directories to be added to the <code>PYTHONPATH</code>.
          <p>
          Subject to <a href="${link make-variables}">"Make variable"</a> substitution. These import
          directories will be added for this rule and all rules that depend on it (note: not the
          rules this rule depends on. Each directory will be added to <code>PYTHONPATH</code> by
          <a href="${link py_binary}"><code>py_binary</code></a> rules that depend on this rule.
          </p>
          <p>
          Absolute paths (paths that start with <code>/</code>) and paths that references a path
          above the execution root are not allowed and will result in an error.
          </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("imports", STRING_LIST).value(ImmutableList.<String>of()))
          /* <!-- #BLAZE_RULE($base_py).ATTRIBUTE(srcs_version) -->
          The value set here is for documentation purpose, and it will NOT determine which version
          of python interpreter to use. Starting with 0.5.3 this attribute has been deprecated and
          no longer has effect.
          A string specifying the Python major version(s) that the <code>.py</code> source
          files listed in the <code>srcs</code> of this rule are compatible with.
          Please reference to <a href="${link py_runtime}"><code>py_runtime</code></a> rules for
          determining the python version.
          Valid values are:<br/>
          <code>"PY2ONLY"</code> -
            Python 2 code that is <b>not</b> suitable for <code>2to3</code> conversion.<br/>
          <code>"PY2"</code> -
            Python 2 code that is expected to work when run through <code>2to3</code>.<br/>
          <code>"PY2AND3"</code> -
            Code that is compatible with both Python 2 and 3 without
            <code>2to3</code> conversion.<br/>
          <code>"PY3ONLY"</code> -
            Python 3 code that will not run on Python 2.<br/>
          <code>"PY3"</code> -
            A synonym for PY3ONLY.<br/>
          <br/>
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(
              attr("srcs_version", STRING)
                  .value(PythonVersion.DEFAULT_SRCS_VALUE.toString())
                  .allowedValues(new AllowedValueSet(PythonVersion.ALL_STRINGS)))
          // TODO(brandjon): Consider adding to py_interpreter a .mandatoryNativeProviders() of
          // BazelPyRuntimeProvider. (Add a test case to PythonConfigurationTest for violations
          // of this requirement.)
          .add(attr(":py_interpreter", LABEL).value(PY_INTERPRETER))
          // do not depend on lib2to3:2to3 rule, because it creates circular dependencies
          // 2to3 is itself written in Python and depends on many libraries.
          .add(
              attr("$python2to3", LABEL)
                  .cfg(HostTransition.INSTANCE)
                  .exec()
                  .value(env.getToolsLabel("//tools/python:2to3")))
          .setPreferredDependencyPredicate(PyRuleClasses.PYTHON_SOURCE)
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$base_py")
          .type(RuleClassType.ABSTRACT)
          .ancestors(BaseRuleClasses.RuleBase.class)
          .build();
    }
  }

  /**
   * Base class for Python rule definitions that produce binaries.
   */
  public static final class PyBinaryBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, final RuleDefinitionEnvironment env) {
      return builder
          /* <!-- #BLAZE_RULE($base_py_binary).ATTRIBUTE(data) -->
          The list of files needed by this binary at runtime.
          See general comments about <code>data</code> at
          <a href="${link common-definitions#common-attributes}">
          Attributes common to all build rules</a>.
          Also see the <a href="${link py_library.data}"><code>data</code></a> argument of
          the <a href="${link py_library}"><code>py_library</code></a> rule for details.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */

          /* <!-- #BLAZE_RULE($base_py_binary).ATTRIBUTE(main) -->
          The name of the source file that is the main entry point of the application.
          This file must also be listed in <code>srcs</code>. If left unspecified,
          <code>name</code> is used instead (see above). If <code>name</code> does not
          match any filename in <code>srcs</code>, <code>main</code> must be specified.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("main", LABEL).allowedFileTypes(PYTHON_SOURCE))
          /* <!-- #BLAZE_RULE($base_py_binary).ATTRIBUTE(default_python_version) -->
          A string specifying the default Python major version to use when building this binary and
          all of its <code>deps</code>.
          Valid values are <code>"PY2"</code> (default) or <code>"PY3"</code>.
          Python 3 support is experimental.
          <p>If both this attribute and <code>python_version</code> are supplied,
          <code>default_python_version</code> will be ignored.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(
              attr(PyCommon.DEFAULT_PYTHON_VERSION_ATTRIBUTE, STRING)
                  .value(PythonVersion.DEFAULT_TARGET_VALUE.toString())
                  .allowedValues(new AllowedValueSet(PythonVersion.TARGET_STRINGS))
                  .nonconfigurable(
                      "read by PyRuleClasses.PYTHON_VERSION_TRANSITION, which doesn't have access"
                          + " to the configuration"))
          /* <!-- #BLAZE_RULE($base_py_binary).ATTRIBUTE(python_version) -->
          An experimental replacement for <code>default_python_version</code>. Only available when
          <code>--experimental_better_python_version_mixing</code> is enabled. If both this and
          <code>default_python_version</code> are supplied, the latter will be ignored.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(
              attr(PyCommon.PYTHON_VERSION_ATTRIBUTE, STRING)
                  .value(PythonVersion.DEFAULT_TARGET_VALUE.toString())
                  .allowedValues(new AllowedValueSet(PythonVersion.TARGET_STRINGS))
                  .nonconfigurable(
                      "read by PyRuleClasses.PYTHON_VERSION_TRANSITION, which doesn't have access"
                          + " to the configuration"))
          /* <!-- #BLAZE_RULE($base_py_binary).ATTRIBUTE(srcs) -->
          The list of source files that are processed to create the target.
          This includes all your checked-in code and any
          generated source files.  The line between <code>srcs</code> and
          <code>deps</code> is loose. The <code>.py</code> files
          probably belong in <code>srcs</code> and library targets probably belong
          in <code>deps</code>, but don't worry about it too much.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(
              attr("srcs", LABEL_LIST)
                  .mandatory()
                  .allowedFileTypes(PYTHON_SOURCE)
                  .direct_compile_time_input()
                  .allowedFileTypes(BazelPyRuleClasses.PYTHON_SOURCE))
          /* <!-- #BLAZE_RULE($base_py_binary).ATTRIBUTE(legacy_create_init) -->
          Whether to implicitly create empty __init__.py files in the runfiles tree.
          These are created in every directory containing Python source code or
          shared libraries, and every parent directory of those directories.
          Default is true for backward compatibility.  If false, the user is responsible
          for creating __init__.py files (empty or not) and adding them to `srcs` or `deps`
          of Python targets as required.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("legacy_create_init", BOOLEAN).value(true))
          /* <!-- #BLAZE_RULE($base_py_binary).ATTRIBUTE(stamp) -->
          Enable link stamping.
          Whether to encode build information into the binary. Possible values:
          <ul>
            <li><code>stamp = 1</code>: Stamp the build information into the
              binary. Stamped binaries are only rebuilt when their dependencies
              change. Use this if there are tests that depend on the build
              information.</li>
            <li><code>stamp = 0</code>: Always replace build information by constant
              values. This gives good build result caching.</li>
            <li><code>stamp = -1</code>: Embedding of build information is controlled
              by the <a href="../user-manual.html#flag--stamp">--[no]stamp</a> Blaze
              flag.</li>
          </ul>
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("stamp", TRISTATE).value(TriState.AUTO))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$base_py_binary")
          .type(RuleClassType.ABSTRACT)
          .ancestors(PyBaseRule.class, CcToolchainRequiringRule.class)
          .build();
    }
  }
}
