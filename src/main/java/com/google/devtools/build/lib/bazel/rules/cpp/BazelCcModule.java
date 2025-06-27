// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.cpp;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.starlark.StarlarkActionFactory;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.rules.cpp.CcCommon.Language;
import com.google.devtools.build.lib.rules.cpp.CcCompilationContext;
import com.google.devtools.build.lib.rules.cpp.CcCompilationOutputs;
import com.google.devtools.build.lib.rules.cpp.CcDebugInfoContext;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext.LinkerInput;
import com.google.devtools.build.lib.rules.cpp.CcModule;
import com.google.devtools.build.lib.rules.cpp.CcToolchainConfigInfo;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables;
import com.google.devtools.build.lib.rules.cpp.CppModuleMap;
import com.google.devtools.build.lib.rules.cpp.CppSemantics;
import com.google.devtools.build.lib.rules.cpp.FeatureConfigurationForStarlark;
import com.google.devtools.build.lib.rules.cpp.LtoBackendArtifacts;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.BazelCcModuleApi;

/** A module that contains Starlark utilities for C++ support. */
public class BazelCcModule extends CcModule
    implements BazelCcModuleApi<
        StarlarkActionFactory,
        Artifact,
        ConstraintValueInfo,
        StarlarkRuleContext,
        FeatureConfigurationForStarlark,
        CcCompilationContext,
        CcCompilationOutputs,
        LtoBackendArtifacts,
        LinkerInput,
        CcLinkingContext,
        CcToolchainVariables,
        CcToolchainConfigInfo,
        CcDebugInfoContext,
        CppModuleMap> {

  @Override
  public CppSemantics getSemantics() {
    return BazelCppSemantics.CPP;
  }

  @Override
  public CppSemantics getSemantics(Language language) {
    return (language == Language.CPP) ? BazelCppSemantics.CPP : BazelCppSemantics.OBJC;
  }
}
