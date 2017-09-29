// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.rules.java;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.CompilationHelper;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.MakeVariableInfo;
import com.google.devtools.build.lib.analysis.MiddlemanProvider;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.vfs.PathFragment;

/** Implementation for the {@code java_runtime} rule. */
public class JavaRuntime implements RuleConfiguredTargetFactory {
  // TODO(lberki): This is incorrect but that what the Jvm configuration fragment did. We'd have the
  // the ability to do better if we knew what OS the BuildConfiguration refers to.
  private static final String BIN_JAVA = "bin/java" + OsUtils.executableExtension();

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    NestedSet<Artifact> filesToBuild =
        PrerequisiteArtifacts.nestedSet(ruleContext, "srcs", Mode.TARGET);
    PathFragment javaHome = defaultJavaHome(ruleContext.getLabel());
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("java_home")) {
      PathFragment javaHomeAttribute =
          PathFragment.create(ruleContext.getExpander().expand("java_home"));
      if (!filesToBuild.isEmpty() && javaHomeAttribute.isAbsolute()) {
        ruleContext.ruleError("'java_home' with an absolute path requires 'srcs' to be empty.");
      }

      if (ruleContext.hasErrors()) {
        return null;
      }

      javaHome = javaHome.getRelative(javaHomeAttribute);
    }

    PathFragment javaBinaryExecPath = javaHome.getRelative(BIN_JAVA);
    PathFragment javaBinaryRunfilesPath = getRunfilesJavaExecutable(
        javaHome, ruleContext.getLabel());

    NestedSet<Artifact> middleman =
        CompilationHelper.getAggregatingMiddleman(
            ruleContext, Actions.escapeLabel(ruleContext.getLabel()), filesToBuild);
    // TODO(cushon): clean up uses of java_runtime in data deps and remove this
    Runfiles runfiles =
        new Runfiles.Builder(ruleContext.getWorkspaceName())
            .addTransitiveArtifacts(filesToBuild)
            .build();

    JavaRuntimeInfo javaRuntime = new JavaRuntimeInfo(
        filesToBuild, javaHome, javaBinaryExecPath, javaBinaryRunfilesPath);

    MakeVariableInfo makeVariableInfo = new MakeVariableInfo(ImmutableMap.of(
        "JAVA", javaBinaryExecPath.getPathString(),
        "JAVABASE", javaHome.getPathString()));

    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProvider(RunfilesProvider.class, RunfilesProvider.simple(runfiles))
        .setFilesToBuild(filesToBuild)
        .addNativeDeclaredProvider(javaRuntime)
        .addProvider(MiddlemanProvider.class, new MiddlemanProvider(middleman))
        .addNativeDeclaredProvider(makeVariableInfo)
        .build();
  }

  static PathFragment defaultJavaHome(Label javabase) {
    if (javabase.getPackageIdentifier().getRepository().isDefault()) {
      return javabase.getPackageFragment();
    }
    return javabase.getPackageIdentifier().getSourceRoot();
  }

  private static PathFragment getRunfilesJavaExecutable(PathFragment javaHome, Label javabase) {
    if (javaHome.isAbsolute() || javabase.getPackageIdentifier().getRepository().isMain()) {
      return javaHome.getRelative(BIN_JAVA);
    } else {
      return javabase.getPackageIdentifier().getRepository().getRunfilesPath()
          .getRelative(BIN_JAVA);
    }
  }
}
