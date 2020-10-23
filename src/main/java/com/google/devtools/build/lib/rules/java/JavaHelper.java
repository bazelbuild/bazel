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
package com.google.devtools.build.lib.rules.java;

import static com.google.devtools.build.lib.packages.BuildType.NODEP_LABEL_LIST;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.List;

/** Utility methods for use by Java-related parts of Bazel. */
// TODO(bazel-team): Merge with JavaUtil.
public abstract class JavaHelper {

  private JavaHelper() {}

  /**
   * Returns the java launcher implementation for the given target, if any. A null return value
   * means "use the JDK launcher".
   */
  public static TransitiveInfoCollection launcherForTarget(
      JavaSemantics semantics, RuleContext ruleContext) {
    String launcher = filterLauncherForTarget(ruleContext);
    return (launcher == null) ? null : ruleContext.getPrerequisite(launcher);
  }

  /**
   * Returns the java launcher artifact for the given target, if any. A null return value means "use
   * the JDK launcher".
   */
  public static Artifact launcherArtifactForTarget(
      JavaSemantics semantics, RuleContext ruleContext) {
    String launcher = filterLauncherForTarget(ruleContext);
    return (launcher == null) ? null : ruleContext.getPrerequisiteArtifact(launcher);
  }

  /**
   * Control structure abstraction for safely extracting a prereq from the launcher attribute or
   * --java_launcher flag.
   */
  private static String filterLauncherForTarget(RuleContext ruleContext) {
    // create_executable=0 disables the launcher
    if (ruleContext.getRule().isAttrDefined("create_executable", Type.BOOLEAN)
        && !ruleContext.attributes().get("create_executable", Type.BOOLEAN)) {
      return null;
    }
    // BUILD rule "launcher" attribute
    if (ruleContext.getRule().isAttrDefined("launcher", BuildType.LABEL)
        && ruleContext.attributes().get("launcher", BuildType.LABEL) != null) {
      if (isJdkLauncher(ruleContext, ruleContext.attributes().get("launcher", BuildType.LABEL))) {
        return null;
      }
      return "launcher";
    }
    // Blaze flag --java_launcher
    JavaConfiguration javaConfig = ruleContext.getFragment(JavaConfiguration.class);
    if (ruleContext.getRule().isAttrDefined(":java_launcher", BuildType.LABEL)
        && javaConfig.getJavaLauncherLabel() != null
        && !isJdkLauncher(ruleContext, javaConfig.getJavaLauncherLabel())) {
      return ":java_launcher";
    }
    return null;
  }

  /**
   * Javac options require special processing - People use them and expect the options to be
   * tokenized.
   */
  public static List<String> tokenizeJavaOptions(Iterable<String> inOpts) {
    // Ideally, this would be in the options parser. Unfortunately,
    // the options parser can't handle a converter that expands
    // from a value X into a List<X> and allow-multiple at the
    // same time.
    List<String> result = new ArrayList<>();
    for (String current : inOpts) {
      try {
        ShellUtils.tokenize(result, current);
      } catch (ShellUtils.TokenizationException ex) {
        // Tokenization failed; this likely means that the user
        // did not want tokenization to happen on their argument.
        // (Any tokenization where we should produce an error
        // has already been done by the shell that invoked
        // blaze). Therefore, pass the argument through to
        // the tool, so that we can see the original error.
        result.add(current);
      }
    }
    return result;
  }

  public static PathFragment getJavaResourcePath(
      JavaSemantics semantics, RuleContext ruleContext, Artifact resource) {
    boolean siblingRepositoryLayout = ruleContext.getConfiguration().isSiblingRepositoryLayout();
    PathFragment resourcePath = resource.getOutputDirRelativePath(siblingRepositoryLayout);
    PathFragment repoExecPath =
        ruleContext
            .getLabel()
            .getRepository()
            .getExecPath(siblingRepositoryLayout);
    if (!repoExecPath.isEmpty() && resourcePath.startsWith(repoExecPath)) {
      resourcePath = resourcePath.relativeTo(repoExecPath);
    }

    if (!ruleContext.attributes().has("resource_strip_prefix", Type.STRING)
        || !ruleContext.attributes().isAttributeValueExplicitlySpecified("resource_strip_prefix")) {
      return semantics.getDefaultJavaResourcePath(resourcePath);
    }

    PathFragment prefix =
        PathFragment.create(ruleContext.attributes().get("resource_strip_prefix", Type.STRING));

    if (!resourcePath.startsWith(prefix)) {
      ruleContext.attributeError(
          "resource_strip_prefix",
          String.format(
              "Resource file '%s' is not under the specified prefix to strip", resourcePath));
      return resourcePath;
    }

    return resourcePath.relativeTo(prefix);
  }

  /**
   * Returns true if the given Label is of the pseudo-cc_binary that tells Bazel a Java target's
   * JAVABIN is never to be replaced by the contents of --java_launcher; only the JDK's launcher
   * will ever be used.
   */
  public static boolean isJdkLauncher(RuleContext ruleContext, Label label) {
    if (!ruleContext.attributes().has("$no_launcher")) {
      return false;
    }
    List<Label> noLauncherAttribute =
        ruleContext.attributes().get("$no_launcher", NODEP_LABEL_LIST);
    return noLauncherAttribute != null && noLauncherAttribute.contains(label);
  }
}
