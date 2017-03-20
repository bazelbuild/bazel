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

package com.google.devtools.build.lib.ideinfo;

import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspect.Builder;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.NativeAspectClass;

/** Generates ide-build information for Android Studio. */
public class AndroidStudioInfoAspect extends NativeAspectClass implements ConfiguredAspectFactory {
  public static final String NAME = "AndroidStudioInfoAspect";

  @Override
  public String getName() {
    return NAME;
  }

  @Override
  public AspectDefinition getDefinition(AspectParameters aspectParameters) {
    return new AspectDefinition.Builder(this).build();
  }

  @Override
  public ConfiguredAspect create(
      ConfiguredTarget base, RuleContext ruleContext, AspectParameters parameters) {
    // Deprecated for bazel > 0.45
    // Can be completely removed after a version or two
    ruleContext.ruleError(
        "AndroidStudioInfoAspect is deprecated. "
            + "If you are using an IntelliJ bazel plugin, "
            + "please update to the latest plugin version from the Jetbrains plugin repository.");
    ConfiguredAspect.Builder builder = new Builder(this, parameters, ruleContext);
    return builder.build();
  }
}
