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

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.TRISTATE;
import static com.google.devtools.build.lib.syntax.Type.STRING;

import com.google.devtools.build.lib.Constants;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCppRuleClasses;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.rules.python.PyRuleClasses;
import com.google.devtools.build.lib.rules.python.PythonVersion;
import com.google.devtools.build.lib.util.FileType;

/**
 * Bazel-specific rule definitions for Python rules.
 */
public final class BazelPyRuleClasses {
  public static final FileType PYTHON_SOURCE = FileType.of(".py");

  public static final String[] ALLOWED_RULES_IN_DEPS = new String[] {
      "py_binary",
      "py_library",
  };

  /**
   * Base class for Python rule definitions.
   */
  public static final class PyBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          /* <!-- #BLAZE_RULE($base_py).ATTRIBUTE(deps) -->
          The list of other libraries to be linked in to the binary target.
          ${SYNOPSIS}
          See general comments about <code>deps</code> at
          <a href="common-definitions.html#common-attributes">
          Attributes common to all build rules</a>.
          These can be
          <a href="#py_binary"><code>py_binary</code></a> rules,
          <a href="#py_library"><code>py_library</code></a> rules or
          <a href="c-cpp.html#cc_library"><code>cc_library</code></a> rules,
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .override(builder.copy("deps")
              .allowedRuleClasses(ALLOWED_RULES_IN_DEPS)
              .allowedFileTypes())
          /* <!-- #BLAZE_RULE($base_py).ATTRIBUTE(srcs_version) -->
          A string specifying the Python major version(s) that the <code>.py</code> source
          files listed in the <code>srcs</code> of this rule are compatible with.
          ${SYNOPSIS}
          Valid values are:<br/>
          <code>"PY2ONLY"</code> -
            Python 2 code that is <b>not</b> suitable for <code>2to3</code> conversion.<br/>
          <code>"PY2"</code> -
            Python 2 code that is expected to work when run through <code>2to3</code>.<br/>
          <code>"PY2AND3"</code> -
            Code that is compatible with both Python 2 and 3 without
            <code>2to3</code> conversion.<br/>
          <code>"PY3"</code> -
            Python 3 code that will not run on Python 2.<br/>
          <br/>
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("srcs_version", STRING)
              .value(PythonVersion.defaultValue().toString()))
          // do not depend on lib2to3:2to3 rule, because it creates circular dependencies
          // 2to3 is itself written in Python and depends on many libraries.
          .add(attr("$python2to3", LABEL).cfg(HOST).exec()
              .value(env.getLabel(Constants.TOOLS_REPOSITORY + "//tools/python:2to3")))
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
         ${SYNOPSIS}
         See general comments about <code>data</code> at
         <a href="common-definitions.html#common-attributes">
         Attributes common to all build rules</a>.
         Also see the <a href="#py_library.data"><code>data</code></a> argument of
         the <a href="#py_library"><code>py_library</code></a> rule for details.
         <!-- #END_BLAZE_RULE.ATTRIBUTE --> */

          /* <!-- #BLAZE_RULE($base_py_binary).ATTRIBUTE(main) -->
          The name of the source file that is the main entry point of the application.
          ${SYNOPSIS}
          This file must also be listed in <code>srcs</code>. If left unspecified,
          <code>name</code> is used instead (see above). If <code>name</code> does not
          match any filename in <code>srcs</code>, <code>main</code> must be specified.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("main", LABEL).allowedFileTypes(PYTHON_SOURCE))
          /* <!-- #BLAZE_RULE($base_py_binary).ATTRIBUTE(default_python_version) -->
          A string specifying the default Python major version to use when building this binary and
          all of its <code>deps</code>.
          ${SYNOPSIS}
          Valid values are <code>"PY2"</code> (default) or <code>"PY3"</code>.
          Python 3 support is experimental.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("default_python_version", STRING)
               .value(PythonVersion.defaultValue().toString())
               .nonconfigurable("read by PythonUtils.getNewPythonVersion, which doesn't have access"
                   + " to configuration keys"))
          /* <!-- #BLAZE_RULE($base_py_binary).ATTRIBUTE(srcs) -->
          The list of source files that are processed to create the target.
          ${SYNOPSIS}
          This includes all your checked-in code and any
          generated source files.  The line between <code>srcs</code> and
          <code>deps</code> is loose. The <code>.py</code> files
          probably belong in <code>srcs</code> and library targets probably belong
          in <code>deps</code>, but don't worry about it too much.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("srcs", LABEL_LIST)
              .mandatory()
              .allowedFileTypes(PYTHON_SOURCE)
              .direct_compile_time_input()
              .allowedFileTypes(BazelPyRuleClasses.PYTHON_SOURCE))
          /* <!-- #BLAZE_RULE($base_py_binary).ATTRIBUTE(stamp) -->
          Enable link stamping.
          ${SYNOPSIS}
          Whether to encode build information into the binary. Possible values:
          <ul>
            <li><code>stamp = 1</code>: Stamp the build information into the
              binary. Stamped binaries are only rebuilt when their dependencies
              change. Use this if there are tests that depend on the build
              information.</li>
            <li><code>stamp = 0</code>: Always replace build information by constant
              values. This gives good build result caching.</li>
            <li><code>stamp = -1</code>: Embedding of build information is controlled
              by the <a href="../blaze-user-manual.html#flag--stamp">--[no]stamp</a> Blaze
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
          .ancestors(PyBaseRule.class, BazelCppRuleClasses.CcLinkingRule.class)
          .build();
    }
  }
}
