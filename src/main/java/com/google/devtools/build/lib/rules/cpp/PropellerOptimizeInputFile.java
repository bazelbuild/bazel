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
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.FileType.HasFileType;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Objects;

/** Value object reused by propeller configurations that has two artifacts. */
@Immutable
public final class PropellerOptimizeInputFile implements HasFileType {

  private final Artifact ccArtifact;
  private final Artifact ldArtifact;

  public PropellerOptimizeInputFile(Artifact ccArtifact, Artifact ldArtifact) {
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

  public static Artifact createAbsoluteArtifact(
      RuleContext ruleContext, PathFragment absolutePath) {
    if (!ruleContext.getFragment(CppConfiguration.class).isFdoAbsolutePathEnabled()) {
      ruleContext.ruleError(
          "absolute paths cannot be used when --enable_fdo_profile_absolute_path is false");
      return null;
    }
    Artifact artifact =
        ruleContext.getUniqueDirectoryArtifact(
            "fdo", absolutePath.getBaseName(), ruleContext.getBinOrGenfilesDirectory());
    ruleContext.registerAction(
        SymlinkAction.toAbsolutePath(
            ruleContext.getActionOwner(),
            PathFragment.create(absolutePath.getPathString()),
            artifact,
            "Symlinking LLVM Propeller Profile " + absolutePath.getPathString()));
    return artifact;
  }

  public static Artifact getAbsolutePathArtifact(RuleContext ruleContext, String attributeName) {
    String pathString = ruleContext.getExpander().expand(attributeName);
    PathFragment absolutePath = PathFragment.create(pathString);
    if (!absolutePath.isAbsolute()) {
      ruleContext.attributeError(
          attributeName, String.format("%s is not an absolute path", absolutePath.getPathString()));
      return null;
    }
    return createAbsoluteArtifact(ruleContext, absolutePath);
  }

  public static PropellerOptimizeInputFile fromProfileRule(RuleContext ruleContext) {

    boolean isCcProfile =
        ruleContext.attributes().isAttributeValueExplicitlySpecified("cc_profile");
    boolean isAbsCcProfile =
        ruleContext.attributes().isAttributeValueExplicitlySpecified("absolute_cc_profile");
    boolean isLdProfile =
        ruleContext.attributes().isAttributeValueExplicitlySpecified("ld_profile");
    boolean isAbsLdProfile =
        ruleContext.attributes().isAttributeValueExplicitlySpecified("absolute_ld_profile");

    if (!isCcProfile && !isLdProfile && !isAbsCcProfile && !isAbsLdProfile) {
      return null;
    }

    if (isCcProfile && isAbsCcProfile) {
      ruleContext.attributeError("cc_profile", "Both relative and absolute profiles are provided.");
    }

    if (isLdProfile && isAbsLdProfile) {
      ruleContext.attributeError("ld_profile", "Both relative and absolute profiles are provided.");
    }

    Artifact ccArtifact = null;
    if (isCcProfile) {
      ccArtifact = ruleContext.getPrerequisiteArtifact("cc_profile");
      if (!ccArtifact.isSourceArtifact()) {
        ruleContext.attributeError("cc_profile", "the target is not an input file");
      }
    } else if (isAbsCcProfile) {
      ccArtifact = getAbsolutePathArtifact(ruleContext, "absolute_cc_profile");
    }

    Artifact ldArtifact = null;
    if (isLdProfile) {
      ldArtifact = ruleContext.getPrerequisiteArtifact("ld_profile");
      if (!ldArtifact.isSourceArtifact()) {
        ruleContext.attributeError("ld_profile", "the target is not an input file");
      }
    } else if (isAbsLdProfile) {
      ldArtifact = getAbsolutePathArtifact(ruleContext, "absolute_ld_profile");
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
