// Copyright 2018 The Bazel Authors. All rights reserved.
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
import static com.google.common.base.Preconditions.checkState;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.ImportDepsCheckingLevel;
import java.util.stream.Collectors;

/** Utility for generating a call to the import_deps_checker. */
public final class ImportDepsCheckActionBuilder {

  public static ImportDepsCheckActionBuilder newBuilder() {
    return new ImportDepsCheckActionBuilder();
  }

  private Artifact jdepsArtifact;
  private Label ruleLabel;
  private NestedSet<Artifact> jarsToCheck;
  private NestedSet<Artifact> bootclasspath;
  private NestedSet<Artifact> declaredDeps;
  private NestedSet<Artifact> transitiveDeps;
  private ImportDepsCheckingLevel importDepsCheckingLevel;

  private ImportDepsCheckActionBuilder() {}

  public ImportDepsCheckActionBuilder checkJars(NestedSet<Artifact> jarsToCheck) {
    checkState(this.jarsToCheck == null);
    this.jarsToCheck = checkNotNull(jarsToCheck);
    return this;
  }

  public ImportDepsCheckActionBuilder ruleLabel(Label ruleLabel) {
    checkState(this.ruleLabel == null);
    this.ruleLabel = checkNotNull(ruleLabel);
    return this;
  }

  public ImportDepsCheckActionBuilder importDepsCheckingLevel(
      ImportDepsCheckingLevel importDepsCheckingLevel) {
    checkState(this.importDepsCheckingLevel == null);
    this.importDepsCheckingLevel = checkNotNull(importDepsCheckingLevel);
    return this;
  }

  public ImportDepsCheckActionBuilder bootclasspath(NestedSet<Artifact> bootclasspath) {
    checkState(this.bootclasspath == null);
    this.bootclasspath = checkNotNull(bootclasspath);
    return this;
  }

  public ImportDepsCheckActionBuilder jdepsOutputArtifact(Artifact jdepsArtifact) {
    checkState(this.jdepsArtifact == null);
    this.jdepsArtifact = checkNotNull(jdepsArtifact);
    return this;
  }

  public ImportDepsCheckActionBuilder declareDeps(NestedSet<Artifact> declaredDeps) {
    checkState(this.declaredDeps == null);
    this.declaredDeps = checkNotNull(declaredDeps);
    return this;
  }

  public ImportDepsCheckActionBuilder transitiveDeps(NestedSet<Artifact> transitiveDeps) {
    checkState(this.transitiveDeps == null);
    this.transitiveDeps = checkNotNull(transitiveDeps);
    return this;
  }

  public void buildAndRegister(RuleContext ruleContext) {
    checkNotNull(jarsToCheck);
    checkNotNull(bootclasspath);
    checkNotNull(declaredDeps);
    checkNotNull(transitiveDeps);
    checkNotNull(importDepsCheckingLevel);
    checkNotNull(jdepsArtifact);
    checkNotNull(ruleLabel);

    ruleContext.registerAction(
        new SpawnAction.Builder()
            .useDefaultShellEnvironment()
            .setExecutable(ruleContext.getExecutablePrerequisite("$import_deps_checker", Mode.HOST))
            .addTransitiveInputs(jarsToCheck)
            .addTransitiveInputs(declaredDeps)
            .addTransitiveInputs(transitiveDeps)
            .addTransitiveInputs(bootclasspath)
            .addOutput(jdepsArtifact)
            .setMnemonic("ImportDepsChecker")
            .setProgressMessage(
                "Checking the completeness of the deps for %s",
                jarsToCheck.toList().stream()
                    .map(Artifact::prettyPrint)
                    .collect(Collectors.joining(", ")))
            .addCommandLine(buildCommandLine())
            .build(ruleContext));
  }

  private CustomCommandLine buildCommandLine() {
    return CustomCommandLine.builder()
        .addExecPaths(VectorArg.addBefore("--input").each(jarsToCheck))
        .addExecPaths(VectorArg.addBefore("--directdep").each(declaredDeps))
        .addExecPaths(VectorArg.addBefore("--classpath_entry").each(transitiveDeps))
        .addExecPaths(VectorArg.addBefore("--bootclasspath_entry").each(bootclasspath))
        .addDynamicString(convertErrorFlag(importDepsCheckingLevel))
        .addExecPath("--jdeps_output", jdepsArtifact)
        .add("--rule_label", ruleLabel.toString())
        .build();
  }

  private static String convertErrorFlag(ImportDepsCheckingLevel level) {
    switch (level) {
      case ERROR:
        return "--checking_mode=error";
      case WARNING:
        return "--checking_mode=warning";
      case OFF:
        return "--checking_mode=silence";
      default:
        throw new RuntimeException("Unhandled deps checking level: " + level);
    }
  }
}
