// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.packages.Attribute.LabelLateBoundDefault;
import com.google.devtools.build.lib.rules.LateBoundAlias.CommonAliasRule;

/** Implementation of the {@code cc_libc_top_alias} rule. */
public class CcLibcTopAlias extends CommonAliasRule {

  public CcLibcTopAlias() {
    super(
        "cc_libc_top_alias",
        env -> CcLibcTopAlias.getSysrootAttribute(),
        CppConfiguration.class);
  }

  private static LabelLateBoundDefault<CppConfiguration> getSysrootAttribute() {
    return LabelLateBoundDefault.fromTargetConfiguration(
        CppConfiguration.class,
        null,
        (rules, attributes, cppConfig) -> cppConfig.getLibcTopLabel());
  }
}
