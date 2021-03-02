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
import static com.google.devtools.build.lib.packages.BuildType.NODEP_LABEL;
import static com.google.devtools.build.lib.packages.BuildType.TRISTATE;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;

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
import com.google.devtools.build.lib.packages.RuleClass.ToolchainTransitionMode;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.rules.python.PyCommon;
import com.google.devtools.build.lib.rules.python.PyInfo;
import com.google.devtools.build.lib.rules.python.PyRuleClasses;
import com.google.devtools.build.lib.rules.python.PyStructUtils;
import com.google.devtools.build.lib.rules.python.PythonVersion;
import com.google.devtools.build.lib.util.FileType;

/**
 * Bazel-specific rule definitions for Python rules.
 */
public final class BazelPyRuleClasses {
  public static final FileType PYTHON_SOURCE = FileType.of(".py", ".py3");

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
          These are generally
          <a href="${link py_library}"><code>py_library</code></a> rules.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .override(
              builder
                  .copy("deps")
                  .mandatoryProvidersList(
                      ImmutableList.of(
                          // Legacy provider.
                          // TODO(b/153363654): Remove this legacy set.
                          ImmutableList.of(
                              StarlarkProviderIdentifier.forLegacy(PyStructUtils.PROVIDER_NAME)),
                          // Modern provider.
                          ImmutableList.of(PyInfo.PROVIDER.id())))
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
          This attribute declares the target's <code>srcs</code> to be compatible with either Python
          2, Python 3, or both. To actually set the Python runtime version, use the
          <a href="${link py_binary.python_version}"><code>python_version</code></a> attribute of an
          executable Python rule (<code>py_binary</code> or <code>py_test</code>).

          <p>Allowed values are: <code>"PY2AND3"</code>, <code>"PY2"</code>, and <code>"PY3"</code>.
          The values <code>"PY2ONLY"</code> and <code>"PY3ONLY"</code> are also allowed for historic
          reasons, but they are essentially the same as <code>"PY2"</code> and <code>"PY3"</code>
          and should be avoided.

          <p>Note that only the executable rules (<code>py_binary</code> and <code>py_library
          </code>) actually verify the current Python version against the value of this attribute.
          (This is a feature; since <code>py_library</code> does not change the current Python
          version, if it did the validation, it'd be impossible to build both <code>PY2ONLY</code>
          and <code>PY3ONLY</code> libraries in the same invocation.) Furthermore, if there is a
          version mismatch, the error is only reported in the execution phase. In particular, the
          error will not appear in a <code>bazel build --nobuild</code> invocation.)

          <p>To get diagnostic information about which dependencies introduce version requirements,
          you can run the <code>find_requirements</code> aspect on your target:
          <pre>
          bazel build &lt;your target&gt; \
              --aspects=@rules_python//python:defs.bzl%find_requirements \
              --output_groups=pyversioninfo
          </pre>
          This will build a file with the suffix <code>-pyversioninfo.txt</code> giving information
          about why your target requires one Python version or another. Note that it works even if
          the given target failed to build due to a version conflict.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(
              attr("srcs_version", STRING)
                  .value(PythonVersion.DEFAULT_SRCS_VALUE.toString())
                  .allowedValues(new AllowedValueSet(PythonVersion.SRCS_STRINGS)))
          // do not depend on lib2to3:2to3 rule, because it creates circular dependencies
          // 2to3 is itself written in Python and depends on many libraries.
          .add(
              attr("$python2to3", LABEL)
                  .cfg(HostTransition.createFactory())
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
          .ancestors(BaseRuleClasses.NativeActionCreatingRule.class)
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
          /* <!-- #BLAZE_RULE($base_py_binary).ATTRIBUTE(main) -->
          The name of the source file that is the main entry point of the application.
          This file must also be listed in <code>srcs</code>. If left unspecified,
          <code>name</code> is used instead (see above). If <code>name</code> does not
          match any filename in <code>srcs</code>, <code>main</code> must be specified.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("main", LABEL).allowedFileTypes(PYTHON_SOURCE))
          /* <!-- #BLAZE_RULE($base_py_binary).ATTRIBUTE(python_version) -->
          Whether to build this target (and its transitive <code>deps</code>) for Python 2 or Python
          3. Valid values are <code>"PY2"</code> and <code>"PY3"</code> (the default).

          <p>The Python version is always reset (possibly by default) to whatever version is
          specified by this attribute, regardless of the version specified on the command line or by
          other higher targets that depend on this one.

          <p>If you want to <code>select()</code> on the current Python version, you can inspect the
          value of <code>@rules_python//python:python_version</code>. See
          <a href="https://github.com/bazelbuild/rules_python/blob/120590e2f2b66e5590bf4dc8ebef9c5338984775/python/BUILD#L43">here</a>
          for more information.

          <p><b>Bug warning:</b> This attribute sets the version for which Bazel builds your target,
          but due to <a href="https://github.com/bazelbuild/bazel/issues/4815">#4815</a>, the
          resulting stub script may still invoke the wrong interpreter version at runtime. See
          <a href="https://github.com/bazelbuild/bazel/issues/4815#issuecomment-460777113">this
          workaround</a>, which involves defining a <code>py_runtime</code> target that points to
          either Python version as needed, and activating this <code>py_runtime</code> by setting
          <code>--python_top</code>.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(
              attr(PyCommon.PYTHON_VERSION_ATTRIBUTE, STRING)
                  .value(PythonVersion._INTERNAL_SENTINEL.toString())
                  .allowedValues(PyRuleClasses.TARGET_PYTHON_ATTR_VALUE_SET)
                  .nonconfigurable(
                      "read by PyRuleClasses.PYTHON_VERSION_TRANSITION, which doesn't have access"
                          + " to the configuration"))
          /* <!-- #BLAZE_RULE($base_py_binary).ATTRIBUTE(srcs) -->
          The list of source (<code>.py</code>) files that are processed to create the target.
          This includes all your checked-in code and any generated source files. Library targets
          belong in <code>deps</code> instead, while other binary files needed at runtime belong in
          <code>data</code>.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(
              attr("srcs", LABEL_LIST)
                  .mandatory()
                  .allowedFileTypes(PYTHON_SOURCE)
                  .direct_compile_time_input())
          /* <!-- #BLAZE_RULE($base_py_binary).ATTRIBUTE(legacy_create_init) -->
          Whether to implicitly create empty __init__.py files in the runfiles tree.
          These are created in every directory containing Python source code or
          shared libraries, and every parent directory of those directories, excluding the repo root
          directory. The default, auto, means true unless
          <code>--incompatible_default_to_explicit_init_py</code> is used. If false, the user is
          responsible for creating (possibly empty) __init__.py files and adding them to the
          <code>srcs</code> of Python targets as required.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("legacy_create_init", TRISTATE).value(TriState.AUTO))
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
          // TODO(brandjon): Consider adding to py_interpreter a .mandatoryBuiltinProviders() of
          // PyRuntimeInfoProvider. (Add a test case to PythonConfigurationTest for violations of
          // this requirement.) Probably moot now that this is going to be replaced by toolchains.
          .add(attr(":py_interpreter", LABEL).value(PY_INTERPRETER))
          .add(
              attr("$py_toolchain_type", NODEP_LABEL)
                  .value(env.getToolsLabel("//tools/python:toolchain_type")))
          .addRequiredToolchains(env.getToolsLabel("//tools/python:toolchain_type"))
          .useToolchainTransition(ToolchainTransitionMode.ENABLED)
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
