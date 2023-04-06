// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.Rule;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkThread;

/**
 * Provider that signals that rules that use launchers can use this target as the launcher.
 *
 * @deprecated This is google internal provider and it will be replaced with a more generally useful
 *     provider in Bazel. Do not use to implement support for launchers in new rules. It's only
 *     supported to be used in existing rules (PyBinary, JavaBinary, JavaTest).
 */
@Deprecated()
@StarlarkBuiltin(
    name = "CcLauncherInfo",
    documented = false,
    doc =
        "Provider that signals that rules that use launchers can use this target as "
            + "the launcher.")
public class CcLauncherInfo extends NativeInfo {
  private static final String RESTRICTION_ERROR_MESSAGE =
      "This provider is restricted to native.java_binary, native.py_binary and native.java_test. "
          + "This is a ";
  public static final String PROVIDER_NAME = "CcLauncherInfo";
  public static final Provider PROVIDER = new Provider();

  private final CcCompilationOutputs ccCompilationOutputs;
  private final CcInfo ccInfo;

  public CcLauncherInfo(CcInfo ccInfo, CcCompilationOutputs ccCompilationOutputs) {
    this.ccInfo = ccInfo;
    this.ccCompilationOutputs = ccCompilationOutputs;
  }

  @Override
  public Provider getProvider() {
    return PROVIDER;
  }

  public CcCompilationOutputs getCcCompilationOutputs(RuleContext ruleContext) {
    checkRestrictedUsage(ruleContext);
    return ccCompilationOutputs;
  }

  @VisibleForTesting
  public CcCompilationOutputs getCcCompilationOutputsForTesting() {
    return ccCompilationOutputs;
  }

  @StarlarkMethod(name = "compilation_outputs", documented = false, useStarlarkThread = true)
  public CcCompilationOutputs getCcCompilationOutputsStarlark(StarlarkThread thread)
      throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return ccCompilationOutputs;
  }

  public CcInfo getCcInfo(RuleContext ruleContext) {
    checkRestrictedUsage(ruleContext);
    return ccInfo;
  }

  @StarlarkMethod(name = "cc_info", documented = false, useStarlarkThread = true)
  public CcInfo getCcInfoForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return ccInfo;
  }

  private void checkRestrictedUsage(RuleContext ruleContext) {
    Rule rule = ruleContext.getRule();
    if (rule.getRuleClassObject().isStarlark()
        || (!rule.getRuleClass().equals("java_binary")
            && !rule.getRuleClass().equals("java_test")
            && !rule.getRuleClass().equals("py_binary")
            && !rule.getRuleClass().equals("py_test"))) {
      throw new IllegalStateException(RESTRICTION_ERROR_MESSAGE + rule.getRuleClass());
    }
  }

  /** Provider class for {@link CcLauncherInfo} objects. */
  public static class Provider extends BuiltinProvider<CcLauncherInfo> {
    private Provider() {
      super(PROVIDER_NAME, CcLauncherInfo.class);
    }
  }
}
