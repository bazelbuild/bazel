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
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuildType;

import java.util.Collection;
import java.util.Map.Entry;

/**
 * Helpers for implementing rules which export Proguard specs.
 *
 * <p>This is not a ConfiguredTargetFactory; $proguard_library, which this class implements, is an
 * abstract rule class, and simply contributes this functionality to other rules.
 */
public final class ProguardLibrary {

  private static final String LOCAL_SPEC_ATTRIBUTE = "proguard_specs";
  private static final ImmutableMultimap<Mode, String> DEPENDENCY_ATTRIBUTES =
      ImmutableMultimap.<Mode, String>builder()
          .putAll(Mode.TARGET, "deps", "exports", "runtime_deps")
          .putAll(Mode.HOST, "plugins", "exported_plugins")
          .build();

  private final RuleContext ruleContext;

  /**
   * Creates a new ProguardLibrary wrapping the given RuleContext.
   */
  public ProguardLibrary(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
  }

  /**
   * Collects the validated proguard specs exported by this rule and its dependencies.
   */
  public NestedSet<Artifact> collectProguardSpecs() {
    return collectProguardSpecs(DEPENDENCY_ATTRIBUTES);
  }

  /**
   * Collects the validated proguard specs exported by this rule and its dependencies through the
   * given attributes.
   */
  public NestedSet<Artifact> collectProguardSpecs(Multimap<Mode, String> attributes) {
    NestedSetBuilder<Artifact> specsBuilder = NestedSetBuilder.naiveLinkOrder();

    for (Entry<Mode, String> attribute : attributes.entries()) {
      specsBuilder.addTransitive(
          collectProguardSpecsFromAttribute(attribute.getValue(), attribute.getKey()));
    }

    Collection<Artifact> localSpecs = collectLocalProguardSpecs();
    if (!localSpecs.isEmpty()) {
      // Pass our local proguard configs through the validator, which checks a whitelist.
      FilesToRunProvider proguardWhitelister =
          ruleContext.getExecutablePrerequisite("$proguard_whitelister", Mode.HOST);
      for (Artifact specToValidate : localSpecs) {
        specsBuilder.add(validateProguardSpec(proguardWhitelister, specToValidate));
      }
    }

    return specsBuilder.build();
  }

  /**
   * Collects the unvalidated proguard specs exported by this rule.
   */
  private Collection<Artifact> collectLocalProguardSpecs() {
    if (!ruleContext.getRule().isAttrDefined(LOCAL_SPEC_ATTRIBUTE, BuildType.LABEL_LIST)) {
      return ImmutableList.of();
    }
    return ruleContext.getPrerequisiteArtifacts(LOCAL_SPEC_ATTRIBUTE, Mode.TARGET).list();
  }

  /**
   * Collects the proguard specs exported by dependencies on the given LABEL_LIST/LABEL attribute.
   */
  private NestedSet<Artifact> collectProguardSpecsFromAttribute(String attribute, Mode mode) {
    if (!(ruleContext.getRule().isAttrDefined(attribute, BuildType.LABEL_LIST)
        || ruleContext.getRule().isAttrDefined(attribute, BuildType.LABEL))) {
      return NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    }
    NestedSetBuilder<Artifact> dependencySpecsBuilder = NestedSetBuilder.naiveLinkOrder();
    for (ProguardSpecProvider provider :
        ruleContext.getPrerequisites(attribute, mode, ProguardSpecProvider.class)) {
      dependencySpecsBuilder.addTransitive(provider.getTransitiveProguardSpecs());
    }
    return dependencySpecsBuilder.build();
  }

  /**
   * Creates an action to run the Proguard whitelister over the given Proguard spec and returns the
   * validated Proguard spec, ready to be exported.
   */
  private Artifact validateProguardSpec(
      FilesToRunProvider proguardWhitelister, Artifact specToValidate) {
    // If we're validating j/a/b/testapp/proguard.cfg, the output will be:
    // j/a/b/testapp/proguard.cfg_valid
    Artifact output =
        ruleContext.getUniqueDirectoryArtifact(
            "validated_proguard",
            specToValidate
                .getRootRelativePath()
                .replaceName(specToValidate.getFilename() + "_valid"),
            ruleContext.getBinOrGenfilesDirectory());
    ruleContext.registerAction(
        new SpawnAction.Builder()
            .addInput(specToValidate)
            .setExecutable(proguardWhitelister)
            .setProgressMessage("Validating proguard configuration")
            .setMnemonic("ValidateProguard")
            .addArgument("--path")
            .addArgument(specToValidate.getExecPathString())
            .addArgument("--output")
            .addArgument(output.getExecPathString())
            .addOutput(output)
            .build(ruleContext));
    return output;
  }
}
