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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.ImportDepsCheckingLevel;
import java.util.stream.Collectors;

/** Utility for generating a call to the import_deps_checker. */
public final class ImportDepsCheckActionBuilder {

  public static ImportDepsCheckActionBuilder newBuilder() {
    return new ImportDepsCheckActionBuilder();
  }

  private Artifact outputArtifact;
  private NestedSet<Artifact> jarsToCheck;
  private NestedSet<Artifact> bootclasspath;
  private NestedSet<Artifact> declaredDeps;
  private ImportDepsCheckingLevel importDepsCheckingLevel;

  private ImportDepsCheckActionBuilder() {}

  public ImportDepsCheckActionBuilder checkJars(NestedSet<Artifact> jarsToCheck) {
    checkState(this.jarsToCheck == null);
    this.jarsToCheck = checkNotNull(jarsToCheck);
    return this;
  }

  public ImportDepsCheckActionBuilder outputArtifiact(Artifact outputArtifact) {
    checkState(this.outputArtifact == null);
    this.outputArtifact = checkNotNull(outputArtifact);
    return this;
  }

  public ImportDepsCheckActionBuilder importDepsCheckingLevel(
      ImportDepsCheckingLevel importDepsCheckingLevel) {
    checkState(this.importDepsCheckingLevel == null);
    checkArgument(importDepsCheckingLevel != ImportDepsCheckingLevel.OFF);
    this.importDepsCheckingLevel = checkNotNull(importDepsCheckingLevel);
    return this;
  }

  public ImportDepsCheckActionBuilder bootcalsspath(NestedSet<Artifact> bootclasspath) {
    checkState(this.bootclasspath == null);
    this.bootclasspath = checkNotNull(bootclasspath);
    return this;
  }

  public ImportDepsCheckActionBuilder declareDeps(NestedSet<Artifact> declaredDeps) {
    checkState(this.declaredDeps == null);
    this.declaredDeps = checkNotNull(declaredDeps);
    return this;
  }

  public void buildAndRegister(RuleContext ruleContext) {
    checkNotNull(outputArtifact);
    checkNotNull(jarsToCheck);
    checkNotNull(bootclasspath);
    checkNotNull(declaredDeps);
    checkState(
        importDepsCheckingLevel == ImportDepsCheckingLevel.ERROR
            || importDepsCheckingLevel == ImportDepsCheckingLevel.WARNING,
        "%s",
        importDepsCheckingLevel);

    CustomCommandLine args =
        CustomCommandLine.builder()
            .addExecPath("--output", outputArtifact)
            .addExecPaths(VectorArg.addBefore("--input").each(jarsToCheck))
            .addExecPaths(VectorArg.addBefore("--classpath_entry").each(declaredDeps))
            .addExecPaths(VectorArg.addBefore("--bootclasspath_entry").each(bootclasspath))
            .add(
                importDepsCheckingLevel == ImportDepsCheckingLevel.ERROR
                    ? "--fail_on_errors"
                    : "--nofail_on_errors")
            .build();
    ruleContext.registerAction(
        new SpawnAction.Builder()
            .useDefaultShellEnvironment()
            .setExecutable(ruleContext.getExecutablePrerequisite("$import_deps_checker", Mode.HOST))
            .addTransitiveInputs(jarsToCheck)
            .addTransitiveInputs(declaredDeps)
            .addTransitiveInputs(bootclasspath)
            .addOutput(outputArtifact)
            .setMnemonic("ImportDepsChecker")
            .setProgressMessage(
                "Checking the completeness of the deps for %s",
                jarsToCheck
                    .toList()
                    .stream()
                    .map(Artifact::prettyPrint)
                    .collect(Collectors.joining(", ")))
            .addCommandLine(args)
            .build(ruleContext));
  }
}
