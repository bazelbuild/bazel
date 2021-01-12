// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuildType;
import java.util.Collection;

/**
 * Helpers for implementing rules which export Proguard specs.
 *
 * <p>This is not a ConfiguredTargetFactory; $proguard_library, which this class implements, is an
 * abstract rule class, and simply contributes this functionality to other rules.
 */
public final class ProguardLibrary {

  private static final String LOCAL_SPEC_ATTRIBUTE = "proguard_specs";
  private static final ImmutableSet<String> DEPENDENCY_ATTRIBUTES =
      ImmutableSet.of("deps", "exports", "runtime_deps", "plugins", "exported_plugins");

  private final RuleContext ruleContext;

  /** Creates a new ProguardLibrary wrapping the given RuleContext. */
  public ProguardLibrary(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
  }

  /** Collects the validated proguard specs exported by this rule and its dependencies. */
  public NestedSet<Artifact> collectProguardSpecs() {
    return collectProguardSpecs(DEPENDENCY_ATTRIBUTES);
  }

  /**
   * Collects the validated proguard specs exported by this rule and its dependencies through the
   * given attributes.
   */
  public NestedSet<Artifact> collectProguardSpecs(Iterable<String> attributes) {
    NestedSetBuilder<Artifact> specsBuilder = NestedSetBuilder.naiveLinkOrder();

    for (String attribute : attributes) {
      specsBuilder.addTransitive(collectProguardSpecsFromAttribute(attribute));
    }

    Collection<Artifact> localSpecs = collectLocalProguardSpecs();
    if (!localSpecs.isEmpty()) {
      // Pass our local proguard configs through the validator, which checks an allowlist.
      FilesToRunProvider proguardAllowlister =
          JavaToolchainProvider.from(ruleContext).getProguardAllowlister();
      if (proguardAllowlister == null) {
        ruleContext.ruleError(
            "java_toolchain.proguard_allowlister is required to use proguard_specs");
        return specsBuilder.build();
      }
      for (Artifact specToValidate : localSpecs) {
        specsBuilder.add(validateProguardSpec(ruleContext, proguardAllowlister, specToValidate));
      }
    }

    return specsBuilder.build();
  }

  /** Collects the unvalidated proguard specs exported by this rule. */
  public ImmutableList<Artifact> collectLocalProguardSpecs() {
    if (!ruleContext.attributes().has(LOCAL_SPEC_ATTRIBUTE, BuildType.LABEL_LIST)) {
      return ImmutableList.of();
    }
    return ruleContext.getPrerequisiteArtifacts(LOCAL_SPEC_ATTRIBUTE).list();
  }

  /**
   * Collects the proguard specs exported by dependencies on the given LABEL_LIST/LABEL attribute.
   */
  private NestedSet<Artifact> collectProguardSpecsFromAttribute(String attributeName) {
    if (!ruleContext.attributes().has(attributeName, BuildType.LABEL_LIST)
        && !ruleContext.attributes().has(attributeName, BuildType.LABEL)) {
      return NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    }
    NestedSetBuilder<Artifact> dependencySpecsBuilder = NestedSetBuilder.naiveLinkOrder();
    for (ProguardSpecProvider provider :
        ruleContext.getPrerequisites(attributeName, ProguardSpecProvider.PROVIDER)) {
      dependencySpecsBuilder.addTransitive(provider.getTransitiveProguardSpecs());
    }
    return dependencySpecsBuilder.build();
  }

  /**
   * Creates an action to run the Proguard allowlister over the given Proguard spec and returns the
   * validated Proguard spec, ready to be exported.
   */
  private Artifact validateProguardSpec(
      RuleContext ruleContext, FilesToRunProvider proguardAllowlister, Artifact specToValidate) {
    // If we're validating j/a/b/testapp/proguard.cfg, the output will be:
    // j/a/b/testapp/proguard.cfg_valid
    Artifact output =
        ruleContext.getUniqueDirectoryArtifact(
            "validated_proguard",
            specToValidate
                .getRootRelativePath()
                .replaceName(specToValidate.getFilename() + "_valid"),
            ruleContext.getBinOrGenfilesDirectory());
    SpawnAction.Builder builder =
        new SpawnAction.Builder().addInput(specToValidate).addOutput(output);
    if (proguardAllowlister.getExecutable().getExtension().equals("jar")) {
      builder
          .setJarExecutable(
              JavaCommon.getHostJavaExecutable(ruleContext),
              proguardAllowlister.getExecutable(),
              JavaToolchainProvider.from(ruleContext).getJvmOptions())
          .addTransitiveInputs(JavaRuntimeInfo.forHost(ruleContext).javaBaseInputsMiddleman());
    } else {
      // TODO(b/170769708): remove this branch and require java_toolchain.proguard_allowlister to
      // always be a _deploy.jar
      builder.setExecutable(proguardAllowlister);
    }
    builder
        .setProgressMessage("Validating proguard configuration")
        .setMnemonic("ValidateProguard")
        .addCommandLine(
            CustomCommandLine.builder()
                .addExecPath("--path", specToValidate)
                .addExecPath("--output", output)
                .build());
    ruleContext.registerAction(builder.build(ruleContext));
    return output;
  }
}
