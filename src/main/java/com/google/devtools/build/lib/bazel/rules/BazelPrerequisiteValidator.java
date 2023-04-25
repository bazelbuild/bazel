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

package com.google.devtools.build.lib.bazel.rules;

import com.google.devtools.build.lib.analysis.CommonPrerequisiteValidator;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.bazel.BazelConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;

/** Ensures that a target's prerequisites are visible to it and match its testonly status. */
public class BazelPrerequisiteValidator extends CommonPrerequisiteValidator {
  @Override
  public boolean isSameLogicalPackage(
      PackageIdentifier thisPackage, PackageIdentifier prerequisitePackage) {
    return thisPackage.equals(prerequisitePackage);
  }

  @Override
  protected boolean packageUnderExperimental(PackageIdentifier packageIdentifier) {
    return false;
  }

  @Override
  protected boolean checkVisibilityForExperimental(RuleContext.Builder context) {
    // It does not matter whether we return true or false here if packageUnderExperimental always
    // returns false.
    return true;
  }

  @Override
  protected boolean checkVisibilityForToolchains(RuleContext.Builder context, Label prerequisite) {
    return context
        .getConfiguration()
        .getFragment(BazelConfiguration.class)
        .checkVisibilityForToolchains();
  }

  @Override
  protected boolean allowExperimentalDeps(RuleContext.Builder context) {
    // It does not matter whether we return true or false here if packageUnderExperimental always
    // returns false.
    return false;
  }
}
