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

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.syntax.Type;

/**
 * An implementation for {@code py_test} rules.
 */
public abstract class PyTest implements RuleConfiguredTargetFactory {
  /**
   * Create a {@link PythonSemantics} object that governs
   * the behavior of this rule.
   */
  protected abstract PythonSemantics createSemantics();

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    PythonSemantics semantics = createSemantics();
    PyCommon common = new PyCommon(ruleContext);
    common.initCommon(getDefaultPythonVersion(ruleContext));

    RuleConfiguredTargetBuilder builder = PyBinary.init(ruleContext, semantics, common);
    if (builder == null) {
      return null;
    }
    return builder.build();
  }

  private PythonVersion getDefaultPythonVersion(RuleContext ruleContext) {
    return ruleContext.getRule().isAttrDefined("default_python_version", Type.STRING)
        ? PyCommon.getPythonVersionAttr(ruleContext, "default_python_version", PythonVersion.PY2,
            PythonVersion.PY3, PythonVersion.PY2AND3)
        : null;
  }
}

