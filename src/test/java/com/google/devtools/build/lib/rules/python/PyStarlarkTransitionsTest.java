// Copyright 2019 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PyStarlarkTransitions}. */
@RunWith(JUnit4.class)
public final class PyStarlarkTransitionsTest extends BuildViewTestCase {

  @Before
  public void setUp() throws Exception {
    scratch.file("myinfo/myinfo.bzl", "MyInfo = provider()");
    scratch.file("myinfo/BUILD");

    scratch.file(
        "my_package/my_rule.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def impl(ctx):",
        "    return MyInfo(",
        "        wrapped = ctx.attr.wrapped,",
        "    )",
        "my_rule = rule(",
        "    implementation = impl,",
        "    attrs = {",
        "        'wrapped': attr.label(cfg = py_transitions.cfg),",
        "        'python_version': attr.string(),",
        "    },",
        ")",
        "missing_attr_rule = rule(",
        "    implementation = impl,",
        "    attrs = {",
        "        'wrapped': attr.label(cfg = py_transitions.cfg),",
        "    },",
        ")");

    scratch.file(
        "my_package/BUILD",
        "load('//my_package:my_rule.bzl', 'my_rule', 'missing_attr_rule')",
        "cc_binary(name = 'wrapped', srcs = ['wrapped.cc'])",
        "my_rule(name = 'py2', wrapped = ':wrapped', python_version = 'PY2')",
        "my_rule(name = 'py3', wrapped = ':wrapped', python_version = 'PY3')",
        "my_rule(name = 'default', wrapped = ':wrapped', python_version = 'DEFAULT')",
        "my_rule(name = 'invalid', wrapped = ':wrapped', python_version = 'invalid')",
        "missing_attr_rule(name = 'missing_attr', wrapped = ':wrapped')");
    setStarlarkSemanticsOptions("--experimental_google_legacy_api");
  }

  @Test
  public void testTransitionToPY2() throws Exception {
    useConfiguration("--python_version=PY3");
    verifyVersion("//my_package:py2", PythonVersion.PY2);
  }

  @Test
  public void testTransitionToPY3() throws Exception {
    useConfiguration("--python_version=PY2");
    verifyVersion("//my_package:py3", PythonVersion.PY3);
  }

  @Test
  public void testTransitionToDefault() throws Exception {
    useConfiguration("--python_version=PY2", "--incompatible_py3_is_default=true");
    verifyVersion("//my_package:default", PythonVersion.PY3);
  }

  @Test
  public void testNoPythonVersionAttribute() throws Exception {
    useConfiguration("--python_version=PY3", "--incompatible_py3_is_default=false");
    // Make sure no errors occur for invalid values, and no transition is applied.
    verifyVersion("//my_package:missing_attr", PythonVersion.PY3);
  }

  @Test
  public void testInvalidPythonVersionAttribute() throws Exception {
    useConfiguration("--python_version=PY2", "--incompatible_py3_is_default=true");
    // Make sure no errors occur for missing python_version attribute, and no transition is applied.
    verifyVersion("//my_package:invalid", PythonVersion.PY2);
  }

  private void verifyVersion(String target, PythonVersion version) throws Exception {
    ConfiguredTarget configuredTarget = getConfiguredTarget(target);
    Provider.Key key =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//myinfo:myinfo.bzl", ImmutableMap.of()), "MyInfo");
    StructImpl myInfo = (StructImpl) configuredTarget.get(key);
    ConfiguredTarget wrapped = (ConfiguredTarget) myInfo.getValue("wrapped");
    PythonOptions wrappedPythonOptions =
        getConfiguration(wrapped).getOptions().get(PythonOptions.class);
    assertThat(wrappedPythonOptions.pythonVersion).isEqualTo(version);
  }
}
