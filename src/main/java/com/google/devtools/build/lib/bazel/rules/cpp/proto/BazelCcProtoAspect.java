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

package com.google.devtools.build.lib.bazel.rules.cpp.proto;

import static com.google.devtools.build.lib.rules.cpp.CppRuleClasses.ccToolchainAttribute;

import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.rules.cpp.CppSemantics;
import com.google.devtools.build.lib.rules.cpp.proto.CcProtoAspect;

/**
 * Part of the implementation of cc_proto_library.
 *
 * <p>This class is used to inject Bazel-specific constants into CcProtoAspect.
 */
public class BazelCcProtoAspect extends CcProtoAspect {
  public BazelCcProtoAspect(CppSemantics cppSemantics, RuleDefinitionEnvironment env) {
    super(cppSemantics, ccToolchainAttribute(env));
  }
}
