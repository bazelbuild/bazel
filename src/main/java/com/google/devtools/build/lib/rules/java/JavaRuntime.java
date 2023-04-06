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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.packages.Type.INTEGER;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TemplateVariableInfo;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/** Implementation for the {@code java_runtime} rule. */
public class JavaRuntime implements RuleConfiguredTargetFactory {
  // TODO(lberki): This is incorrect but that what the Jvm configuration fragment did. We'd have the
  // the ability to do better if we knew what OS the BuildConfigurationValue refers to.
  private static final String BIN_JAVA = "bin/java" + OsUtils.executableExtension();

  @Override
  @Nullable
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    NestedSetBuilder<Artifact> filesBuilder = NestedSetBuilder.stableOrder();
    BuildConfigurationValue configuration = checkNotNull(ruleContext.getConfiguration());
    filesBuilder.addTransitive(PrerequisiteArtifacts.nestedSet(ruleContext, "srcs"));
    boolean siblingRepositoryLayout = configuration.isSiblingRepositoryLayout();
    PathFragment javaHome = defaultJavaHome(ruleContext.getLabel(), siblingRepositoryLayout);
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("java_home")) {
      PathFragment javaHomeAttribute =
          PathFragment.create(ruleContext.getExpander().expand("java_home"));
      if (!filesBuilder.isEmpty() && javaHomeAttribute.isAbsolute()) {
        ruleContext.ruleError("'java_home' with an absolute path requires 'srcs' to be empty.");
      }

      if (ruleContext.hasErrors()) {
        return null;
      }

      javaHome = javaHome.getRelative(javaHomeAttribute);
    }

    PathFragment javaBinaryExecPath = javaHome.getRelative(BIN_JAVA);
    PathFragment javaBinaryRunfilesPath =
        getRunfilesJavaExecutable(javaHome, ruleContext.getLabel());

    Artifact java = ruleContext.getPrerequisiteArtifact("java");
    if (java != null) {
      if (javaHome.isAbsolute()) {
        ruleContext.ruleError("'java_home' with an absolute path requires 'java' to be empty.");
      }
      javaBinaryExecPath = java.getExecPath();
      javaBinaryRunfilesPath = java.getRunfilesPath();
      if (!isJavaBinary(javaBinaryExecPath)) {
        ruleContext.ruleError("the path to 'java' must end in 'bin/java'.");
      }
      if (ruleContext.hasErrors()) {
        return null;
      }
      javaHome = javaBinaryExecPath.getParentDirectory().getParentDirectory();
      filesBuilder.add(java);
    }
    PathFragment javaHomeRunfilesPath =
        javaBinaryRunfilesPath.getParentDirectory().getParentDirectory();

    NestedSet<Artifact> hermeticInputs =
        PrerequisiteArtifacts.nestedSet(ruleContext, "hermetic_srcs");
    filesBuilder.addTransitive(hermeticInputs);

    Artifact libModules = ruleContext.getPrerequisiteArtifact("lib_modules");

    ImmutableList<CcInfo> hermeticStaticLibs =
        ImmutableList.copyOf(ruleContext.getPrerequisites("hermetic_static_libs", CcInfo.PROVIDER));

    // If a runtime does not set default_cds in hermetic mode, it is not fatal.
    // We can skip the default CDS in the check below.
    Artifact defaultCDS = ruleContext.getPrerequisiteArtifact("default_cds");

    if ((!hermeticInputs.isEmpty() || libModules != null || !hermeticStaticLibs.isEmpty())
        && (hermeticInputs.isEmpty() || libModules == null || hermeticStaticLibs.isEmpty())) {
      ruleContext.attributeError(
          "hermetic",
          "hermetic specified, all of java_runtime.lib_modules, java_runtime.hermetic_srcs and"
              + " java_runtime.hermetic_static_libs must be specified");
    }

    NestedSet<Artifact> filesToBuild = filesBuilder.build();

    // TODO(cushon): clean up uses of java_runtime in data deps and remove this
    Runfiles runfiles =
        new Runfiles.Builder(ruleContext.getWorkspaceName())
            .addTransitiveArtifacts(filesToBuild)
            .build();

    JavaRuntimeInfo javaRuntime =
        JavaRuntimeInfo.create(
            filesToBuild,
            javaHome,
            javaBinaryExecPath,
            javaHomeRunfilesPath,
            javaBinaryRunfilesPath,
            hermeticInputs,
            libModules,
            defaultCDS,
            hermeticStaticLibs,
            ruleContext.attributes().get("version", INTEGER).toIntUnchecked());

    TemplateVariableInfo templateVariableInfo =
        new TemplateVariableInfo(
            ImmutableMap.of(
                "JAVA", javaBinaryExecPath.getPathString(),
                "JAVABASE", javaHome.getPathString()),
            ruleContext.getRule().getLocation());

    ToolchainInfo toolchainInfo =
        new ToolchainInfo(
            ImmutableMap.<String, Object>builder().put("java_runtime", javaRuntime).buildOrThrow());
    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProvider(RunfilesProvider.class, RunfilesProvider.simple(runfiles))
        .setFilesToBuild(filesToBuild)
        .addNativeDeclaredProvider(javaRuntime)
        .addNativeDeclaredProvider(templateVariableInfo)
        .addNativeDeclaredProvider(toolchainInfo)
        .build();
  }

  private static boolean isJavaBinary(PathFragment path) {
    return path.endsWith(PathFragment.create("bin/java"))
        || path.endsWith(PathFragment.create("bin/java.exe"));
  }

  static PathFragment defaultJavaHome(Label javabase, boolean siblingRepositoryLayout) {
    if (javabase.getRepository().isMain()) {
      return javabase.getPackageFragment();
    }
    return javabase.getPackageIdentifier().getExecPath(siblingRepositoryLayout);
  }

  private static PathFragment getRunfilesJavaExecutable(PathFragment javaHome, Label javabase) {
    if (javaHome.isAbsolute() || javabase.getRepository().isMain()) {
      return javaHome.getRelative(BIN_JAVA);
    } else {
      return javabase
          .getRepository()
          .getRunfilesPath()
          .getRelative(BIN_JAVA);
    }
  }
}
