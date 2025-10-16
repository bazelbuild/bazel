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

package com.google.devtools.build.lib.bazel.rules.cpp;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.rules.cpp.CcCommon.Language;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppSemantics;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;

/** C++ compilation semantics. */
public class BazelCppSemantics implements CppSemantics {
  @SerializationConstant
  public static final BazelCppSemantics CPP = new BazelCppSemantics(Language.CPP);

  @SerializationConstant
  public static final BazelCppSemantics OBJC = new BazelCppSemantics(Language.OBJC);

  private final Language language;

  private BazelCppSemantics(Language language) {
    this.language = language;
  }

  private static final Label CPP_TOOLCHAIN_TYPE =
      Label.parseCanonicalUnchecked("@bazel_tools//tools/cpp:toolchain_type");

  @Override
  public Label getCppToolchainType() {
    return CPP_TOOLCHAIN_TYPE;
  }

  @Override
  public Language language() {
    return language;
  }

  @Override
  public boolean needsIncludeValidation() {
    return language != Language.OBJC;
  }

  @Override
  public void validateLayeringCheckFeatures(
      RuleContext ruleContext,
      AspectDescriptor aspectDescriptor,
      CcToolchainProvider ccToolchain,
      ImmutableSet<String> unsupportedFeatures) {}
}
