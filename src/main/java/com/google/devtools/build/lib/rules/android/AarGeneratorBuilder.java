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
package com.google.devtools.build.lib.rules.android;

import com.google.common.base.Functions;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.rules.android.AndroidResourcesProvider.ResourceContainer;
import com.google.devtools.build.lib.rules.android.AndroidResourcesProvider.ResourceType;

import java.util.ArrayList;
import java.util.List;

/**
 * Builder for creating aar generator action.
 */
public class AarGeneratorBuilder {

  private ResourceContainer primary;
  private Artifact manifest;
  private Artifact rTxt;
  private Artifact classes;

  private Artifact aarOut;

  private final RuleContext ruleContext;
  private final SpawnAction.Builder builder;

  /**
   * Creates an {@link AarGeneratorBuilder}.
   *
   * @param ruleContext The {@link RuleContext} that is used to register the {@link Action}.
   */
  public AarGeneratorBuilder(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
    this.builder = new SpawnAction.Builder();
  }

  public AarGeneratorBuilder withPrimary(ResourceContainer primary) {
    this.primary = primary;
    return this;
  }

  public AarGeneratorBuilder withManifest(Artifact manifest) {
    this.manifest = manifest;
    return this;
  }

  public AarGeneratorBuilder withRtxt(Artifact rTxt) {
    this.rTxt = rTxt;
    return this;
  }

  public AarGeneratorBuilder withClasses(Artifact classes) {
    this.classes = classes;
    return this;
  }

  public AarGeneratorBuilder setAAROut(Artifact aarOut) {
    this.aarOut = aarOut;
    return this;
  }

  public void build(ActionConstructionContext context) {
    List<Artifact> outs = new ArrayList<>();
    List<Artifact> ins = new ArrayList<>();
    List<String> args = new ArrayList<>();

    args.add("--mainData");
    addPrimaryResourceContainer(ins, args, primary);

    if (manifest != null) {
      args.add("--manifest");
      args.add(manifest.getExecPathString());
      ins.add(manifest);
    }

    if (rTxt != null) {
      args.add("--rtxt");
      args.add(rTxt.getExecPathString());
      ins.add(rTxt);
    }

    if (classes != null) {
      args.add("--classes");
      args.add(classes.getExecPathString());
      ins.add(classes);
    }

    args.add("--aarOutput");
    args.add(aarOut.getExecPathString());
    outs.add(aarOut);

    ruleContext.registerAction(this.builder
        .addInputs(ImmutableList.<Artifact>copyOf(ins))
        .addOutputs(ImmutableList.<Artifact>copyOf(outs))
        .setCommandLine(CommandLine.of(args, false))
        .setExecutable(
            ruleContext.getExecutablePrerequisite("$android_aar_generator", Mode.HOST))
        .setProgressMessage("Building AAR package for " + ruleContext.getLabel())
        .setMnemonic("AARGenerator")
        .build(context));
  }

  private void addPrimaryResourceContainer(List<Artifact> inputs, List<String> args,
      ResourceContainer container) {
    Iterables.addAll(inputs, container.getArtifacts());
    inputs.add(container.getManifest());

    // no R.txt, because it will be generated from this action.
    args.add(String.format("%s:%s:%s",
        convertRoots(container, ResourceType.RESOURCES),
        convertRoots(container, ResourceType.ASSETS),
        container.getManifest().getExecPathString()
    ));
  }

  private static String convertRoots(ResourceContainer container, ResourceType resourceType) {
    return Joiner.on("#").join(
        Iterators.transform(
            container.getRoots(resourceType).iterator(), Functions.toStringFunction()));
  }
}
