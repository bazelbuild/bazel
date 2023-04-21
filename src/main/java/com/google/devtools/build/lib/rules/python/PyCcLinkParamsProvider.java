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

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StarlarkProviderWrapper;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import net.starlark.java.eval.EvalException;

/** A target that provides C++ libraries to be linked into Python targets. */
@VisibleForTesting
@Immutable
public final class PyCcLinkParamsProvider {
  public static final Provider PROVIDER = new Provider();

  private final CcInfo ccInfo;

  public PyCcLinkParamsProvider(StarlarkInfo info) throws EvalException {
    this.ccInfo = info.getValue("cc_info", CcInfo.class);
  }

  public Provider getProvider() {
    return PROVIDER;
  }

  public CcInfo getCcInfo() {
    return ccInfo;
  }

  /** Provider class for {@link PyCcLinkParamsProvider} objects. */
  public static class Provider extends StarlarkProviderWrapper<PyCcLinkParamsProvider> {
    private Provider() {
      super(
          Label.parseCanonicalUnchecked("@_builtins//:common/python/providers.bzl"),
          "PyCcLinkParamsProvider");
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
}
