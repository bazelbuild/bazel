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

import static com.google.devtools.build.lib.skyframe.BzlLoadValue.keyForBuild;
import static com.google.devtools.build.lib.skyframe.BzlLoadValue.keyForBuiltins;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StarlarkProviderWrapper;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.skyframe.BzlLoadValue;
import net.starlark.java.eval.EvalException;

/** A target that provides C++ libraries to be linked into Python targets. */
@VisibleForTesting
@Immutable
public final class PyCcLinkParamsProvider {
  private static final BuiltinProvider BUILTIN_PROVIDER = new BuiltinProvider();
  private static final RulesPythonProvider RULES_PYTHON_PROVIDER = new RulesPythonProvider();

  private final CcInfo ccInfo;

  public PyCcLinkParamsProvider(StarlarkInfo info) throws EvalException {
    this.ccInfo = info.getValue("cc_info", CcInfo.class);
  }

  public static PyCcLinkParamsProvider fromTarget(ConfiguredTarget target)
      throws RuleErrorException {
    PyCcLinkParamsProvider provider = target.get(RULES_PYTHON_PROVIDER);
    if (provider != null) {
      return provider;
    }
    provider = target.get(BUILTIN_PROVIDER);
    if (provider != null) {
      return provider;
    }
    throw new IllegalStateException(
        String.format("Unable to find PyCcLinkParamsProvider provider in %s", target));
  }

  public CcInfo getCcInfo() {
    return ccInfo;
  }

  private static class BaseProvider extends StarlarkProviderWrapper<PyCcLinkParamsProvider> {
    private BaseProvider(BzlLoadValue.Key loadKey) {
      super(loadKey, "PyCcLinkParamsProvider");
    }

    @Override
    public PyCcLinkParamsProvider wrap(Info value) throws RuleErrorException {
      try {
        return new PyCcLinkParamsProvider((StarlarkInfo) value);
      } catch (EvalException e) {
        throw new RuleErrorException(e.getMessageWithStack());
      }
    }
  }

  /** Provider class for builtin PyWrapCcLinkParamsProvider. */
  public static class BuiltinProvider extends BaseProvider {
    private BuiltinProvider() {
      super(
          keyForBuiltins(
              Label.parseCanonicalUnchecked("@_builtins//:common/python/providers.bzl")));
    }
  }

  /** Provider class for rules_python PyWrapCcLinkParamsProvider. */
  public static class RulesPythonProvider extends BaseProvider {
    private RulesPythonProvider() {
      super(
          keyForBuild(
              Label.parseCanonicalUnchecked(
                  "//third_party/bazel_rules/rules_python/python/private/common:providers.bzl")));
    }
  }
}
