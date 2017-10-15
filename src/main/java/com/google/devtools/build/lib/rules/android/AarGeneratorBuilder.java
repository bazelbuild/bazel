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
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.rules.android.ResourceContainer.ResourceType;
import com.google.devtools.build.lib.util.OS;
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
  private boolean throwOnResourceConflict;

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

  public AarGeneratorBuilder setThrowOnResourceConflict(boolean throwOnResourceConflict) {
    this.throwOnResourceConflict = throwOnResourceConflict;
    return this;
  }

  public void build(ActionConstructionContext context) {
    List<Artifact> outs = new ArrayList<>();
    List<Artifact> ins = new ArrayList<>();
    List<String> args = new ArrayList<>();

    // Set the busybox tool
    args.add("--tool");
    args.add("GENERATE_AAR");
    // Deliminate between the tool and the tool arguments.
    args.add("--");

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

    if (throwOnResourceConflict) {
      args.add("--throwOnResourceConflict");
    }

    if (OS.getCurrent() == OS.WINDOWS) {
      // Some flags (e.g. --mainData) may specify lists (or lists of lists) separated by special
      // characters (colon, semicolon, hashmark, ampersand) that don't work on Windows, and quoting
      // semantics are very complicated (more so than in Bash), so let's just always use a parameter
      // file.
      // TODO(laszlocsomor), TODO(corysmith): restructure the Android BusyBux's flags by deprecating
      // list-type and list-of-list-type flags that use such problematic separators in favor of
      // multi-value flags (to remove one level of listing) and by changing all list separators to a
      // platform-safe character (= comma).
      builder.alwaysUseParameterFile(ParameterFileType.UNQUOTED);
    }

    ruleContext.registerAction(
        this.builder
            .useDefaultShellEnvironment()
            .addInputs(ImmutableList.<Artifact>copyOf(ins))
            .addOutputs(ImmutableList.<Artifact>copyOf(outs))
            .setCommandLine(CommandLine.of(args))
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$android_resources_busybox", Mode.HOST))
            .setProgressMessage("Building AAR package for %s", ruleContext.getLabel())
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
