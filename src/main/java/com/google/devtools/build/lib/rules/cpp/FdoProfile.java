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
package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.vfs.PathFragment;

/** Implementation for the {@code fdo_profile} rule. */
@Immutable
public final class FdoProfile implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws RuleErrorException, ActionConflictException {

    FdoProfileProvider provider;

    boolean isLabel = ruleContext.attributes().isAttributeValueExplicitlySpecified("profile");
    boolean isAbsolutePath =
        ruleContext.attributes().isAttributeValueExplicitlySpecified("absolute_path_profile");

    if (isLabel == isAbsolutePath) {
      ruleContext.ruleError("exactly one of profile and absolute_path_profile should be specified");
      return null;
    }

    if (isLabel) {
      Artifact artifact = ruleContext.getPrerequisiteArtifact("profile", Mode.TARGET);
      if (!artifact.isSourceArtifact()) {
        ruleContext.attributeError("profile", " the target is not an input file");
      }
      provider = FdoProfileProvider.fromArtifact(artifact);
    } else {
      if (!ruleContext.getFragment(CppConfiguration.class).isFdoAbsolutePathEnabled()) {
        ruleContext.attributeError(
            "absolute_path_profile",
            "this attribute cannot be used when --enable_fdo_profile_absolute_path is false");
        return null;
      }
      String path = ruleContext.getExpander().expand("absolute_path_profile");
      PathFragment fdoPath = PathFragment.create(path);
      if (!fdoPath.isAbsolute()) {
        ruleContext.attributeError(
            "absolute_path_profile",
            String.format("%s is not an absolute path", fdoPath.getPathString()));
      }
      provider = FdoProfileProvider.fromAbsolutePath(fdoPath);
    }

    return new RuleConfiguredTargetBuilder(ruleContext)
        .addNativeDeclaredProvider(provider)
        .addProvider(RunfilesProvider.simple(Runfiles.EMPTY))
        .build();
  }
}
