// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.starlarkbuildapi.objc.AppleCommonApi;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/** A class that exposes apple rule implementation internals to Starlark. */
public class AppleStarlarkCommon
    implements AppleCommonApi<ConstraintValueInfo, StarlarkRuleContext> {

  @VisibleForTesting
  public static final String BAD_KEY_ERROR =
      "Argument %s not a recognized key, 'strict_include', or 'providers'.";

  @VisibleForTesting
  public static final String NOT_SET_ERROR = "Value for key %s must be a set, instead found %s.";

  @Nullable private StructImpl platform;

  public AppleStarlarkCommon() {}

  @Override
  public StructImpl getPlatformStruct() {
    if (platform == null) {
      platform = ApplePlatform.getStarlarkStruct();
    }
    return platform;
  }

  @Override
  public DottedVersion dottedVersion(String version) throws EvalException {
    try {
      return DottedVersion.fromString(version);
    } catch (DottedVersion.InvalidDottedVersionException e) {
      throw new EvalException(e.getMessage());
    }
  }
}
