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

package com.google.devtools.build.lib.analysis;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;

import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.ToolchainContext.ResolvedToolchainProviders;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.RuleClass;
//import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
//import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.function.Function;

/** Class to work with the shell toolchain, e.g. get the shell interpreter's path. */
public final class ShToolchain {

  private static final String TOOLCHAIN_TYPE_ATTR = "$sh_toolchain_type";

  public static Attribute.Builder createAttribute(Label toolchainType) {
    return attr(TOOLCHAIN_TYPE_ATTR, LABEL).value(toolchainType);
  }

  public static String getToolchainTypeLabel() {
    return "//tools/sh:toolchain_type";
  }

  public static RuleClass.Builder addDependency(
      RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return addDependency(builder, env.getToolsLabel(getToolchainTypeLabel()));
  }

  public static RuleClass.Builder addDependency(
      RuleClass.Builder builder, Function<String, Label> getToolsLabel) {
    return addDependency(builder, getToolsLabel.apply(getToolchainTypeLabel()));
  }

  private static RuleClass.Builder addDependency(RuleClass.Builder builder, Label toolchainType) {
    return builder
        //.requiresConfigurationFragments(
         //   PlatformConfiguration.class, CppConfiguration.class, JavaConfiguration.class)
        .addRequiredToolchains(toolchainType)
        .add(createAttribute(toolchainType));
  }

  /**
   * Returns the shell executable's path, or an empty path if not set.
   *
   * <p>This method checks the configuration's {@link ShellConfiguration} fragment.
   */
  public static PathFragment getPath(BuildConfiguration config) {
    PathFragment result = PathFragment.EMPTY_FRAGMENT;

    ShellConfiguration configFragment =
      (ShellConfiguration) config.getFragment(ShellConfiguration.class);
    if (configFragment != null) {
      PathFragment path = configFragment.getShellExecutable();
      if (path != null) {
        result = path;
      }
    }

    return result;
  }

  /**
   * Returns the shell executable's path, or reports a rule error if the path is empty.
   *
   * <p>This method checks the rule's configuration's {@link ShellConfiguration} fragment for the
   * shell executable's path. If null or empty but the rule depends on the shell toolchain, this
   * method gets the path from the selected shell toolchain. If the path is still null or empty, the
   * method reports an error against the rule.
   */
  public static PathFragment getPathOrError(RuleContext ctx) {
    PathFragment result = getPath(ctx.getConfiguration());
    // PathFragment result = PathFragment.EMPTY_FRAGMENT;

    if (result.isEmpty() && ctx.attributes().has(TOOLCHAIN_TYPE_ATTR, LABEL)) {
      if (ctx == null) {
        System.out.printf("DEBUG: 1 ShToolchain.getPathOrError, ctx=(%s)%n", ctx);
      }
      if (ctx.getToolchainContext() == null) {
        System.out.printf("DEBUG: 2 ShToolchain.getPathOrError, ctx=(%s)%n", ctx);
      }
      if (ctx.getToolchainContext().getResolvedToolchainProviders() == null) {
        System.out.printf("DEBUG: 3 ShToolchain.getPathOrError, ctx=(%s)%n", ctx);
      }
      ResolvedToolchainProviders toolchains =
          (ResolvedToolchainProviders) ctx.getToolchainContext().getResolvedToolchainProviders();

      ToolchainInfo activeToolchain =
          toolchains.getForToolchainType(ctx.attributes().get(TOOLCHAIN_TYPE_ATTR, LABEL));

      if (activeToolchain != null) {
        String path = null;
        try {
          path = (String) activeToolchain.getValue("path");
        } catch (EvalException e) {
          throw new IllegalStateException(e);
        }

        if (path != null && !path.isEmpty()) {
          result = PathFragment.create(path);
        }
      }
    }

    if (result.isEmpty()) {
      Thread.dumpStack();
      ctx.ruleError(
          "This rule needs a shell interpreter. Use the --shell_executable=<path> flag to specify"
          + " the interpreter's path, e.g. --shell_executable=/usr/local/bin/bash");
    }

    return result;
  }

  private ShToolchain() {}

  public static boolean isToolchainAttribute(Attribute a) {
    return a.getName().equals(TOOLCHAIN_TYPE_ATTR);
  }

  public static boolean isToolchainLabel(String l) {
    return l.endsWith("//tools/sh:toolchain_type");
  }
}
