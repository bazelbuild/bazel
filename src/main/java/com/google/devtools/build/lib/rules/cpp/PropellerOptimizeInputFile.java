// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.FileType.HasFileType;
import java.util.Objects;

/** Value object reused by propeller configurations that has two artifacts. */
@Immutable
public final class PropellerOptimizeInputFile implements HasFileType {

  private final Artifact ccArtifact;
  private final Artifact ldArtifact;

  private PropellerOptimizeInputFile(Artifact ccArtifact, Artifact ldArtifact) {
    Preconditions.checkArgument((ccArtifact != null) || (ldArtifact != null));
    this.ccArtifact = ccArtifact;
    this.ldArtifact = ldArtifact;
  }

  public Artifact getCcArtifact() {
    return ccArtifact;
  }

  public Artifact getLdArtifact() {
    return ldArtifact;
  }

  public String getBasename() {
    return ccArtifact != null ? ccArtifact.getFilename() : ldArtifact.getFilename();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }

    if (!(o instanceof PropellerOptimizeInputFile)) {
      return false;
    }

    PropellerOptimizeInputFile that = (PropellerOptimizeInputFile) o;
    return Objects.equals(this.ccArtifact, that.ccArtifact)
        && Objects.equals(this.ldArtifact, that.ldArtifact);
  }

  @Override
  public int hashCode() {
    return Objects.hash(ccArtifact, ldArtifact);
  }

  public static PropellerOptimizeInputFile fromProfileRule(RuleContext ruleContext) {

    boolean isCcProfile =
        ruleContext.attributes().isAttributeValueExplicitlySpecified("cc_profile");
    boolean isLdProfile =
        ruleContext.attributes().isAttributeValueExplicitlySpecified("ld_profile");

    if (!isCcProfile && !isLdProfile) {
      return null;
    }

    Artifact ccArtifact = null;
    if (isCcProfile) {
      ccArtifact = ruleContext.getPrerequisiteArtifact("cc_profile");
      if (!ccArtifact.isSourceArtifact()) {
        ruleContext.attributeError("cc_profile", "the target is not an input file");
      }
    }

    Artifact ldArtifact = null;
    if (isLdProfile) {
      ldArtifact = ruleContext.getPrerequisiteArtifact("ld_profile");
      if (!ldArtifact.isSourceArtifact()) {
        ruleContext.attributeError("ld_profile", "the target is not an input file");
      }
    }
    return new PropellerOptimizeInputFile(ccArtifact, ldArtifact);
  }

  @Override
  public String filePathForFileTypeMatcher() {
    String s = "";
    if (ccArtifact != null) {
      s += ccArtifact.filePathForFileTypeMatcher();
    }
    if (ldArtifact != null) {
      s += ldArtifact.filePathForFileTypeMatcher();
    }
    return s;
  }
}
