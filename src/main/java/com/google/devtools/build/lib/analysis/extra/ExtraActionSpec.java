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
package com.google.devtools.build.lib.analysis.extra;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.analysis.BashCommandConstructor;
import com.google.devtools.build.lib.analysis.CommandHelper;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

/**
 * The specification for a particular extra action type.
 */
@Immutable
public final class ExtraActionSpec implements TransitiveInfoProvider {
  private final PathFragment shExecutable;
  private final NestedSet<Artifact> resolvedTools;
  private final RunfilesSupplier runfilesSupplier;
  private final ImmutableList<Artifact> resolvedData;
  private final ImmutableList<String> outputTemplates;
  private final ImmutableMap<String, String> executionInfo;
  private final String command;
  private final boolean requiresActionOutput;
  private final Label label;

  public ExtraActionSpec(
      PathFragment shExecutable,
      NestedSet<Artifact> resolvedTools,
      RunfilesSupplier runfilesSupplier,
      List<Artifact> resolvedData,
      List<String> outputTemplates,
      String command,
      Label label,
      Map<String, String> executionInfo,
      boolean requiresActionOutput) {
    this.shExecutable = shExecutable;
    this.resolvedTools = resolvedTools;
    this.runfilesSupplier = runfilesSupplier;
    this.resolvedData = ImmutableList.copyOf(resolvedData);
    this.outputTemplates = ImmutableList.copyOf(outputTemplates);
    this.command = command;
    this.label = label;
    this.executionInfo = ImmutableMap.copyOf(executionInfo);
    this.requiresActionOutput = requiresActionOutput;
  }

  public Label getLabel() {
    return label;
  }

  /** Adds an extra_action to the action graph based on the action to shadow. */
  public Collection<Artifact.DerivedArtifact> addExtraAction(
      RuleContext owningRule, Action actionToShadow) throws InterruptedException {
    Collection<Artifact.DerivedArtifact> extraActionOutputs = new LinkedHashSet<>();
    Collection<Artifact.DerivedArtifact> protoOutputs = new ArrayList<>();
    NestedSetBuilder<Artifact> extraActionInputs = NestedSetBuilder.stableOrder();

    Label ownerLabel = owningRule.getLabel();
    if (requiresActionOutput) {
      extraActionInputs.addAll(actionToShadow.getOutputs());
    }

    extraActionInputs.addTransitive(resolvedTools);
    extraActionInputs.addAll(resolvedData);

    boolean createDummyOutput = false;

    for (String outputTemplate : outputTemplates) {
      // We create output for the extra_action based on the 'out_template' attribute.
      // See {link #getExtraActionOutputArtifact} for supported variables.
      extraActionOutputs.add(getExtraActionOutputArtifact(
          owningRule, actionToShadow, outputTemplate));
    }
    // extra_action has no output, we need to create some dummy output to keep the build up-to-date.
    if (extraActionOutputs.isEmpty()) {
      createDummyOutput = true;
      extraActionOutputs.add(getExtraActionOutputArtifact(
          owningRule, actionToShadow, "$(ACTION_ID).dummy"));
    }

    // We generate a file containing a protocol buffer describing the action that is being shadowed.
    // It is up to each action being shadowed to decide what contents to store here.
    Artifact.DerivedArtifact extraActionInfoFile =
        getExtraActionOutputArtifact(owningRule, actionToShadow, "$(ACTION_ID).xa");
    owningRule.registerAction(new ExtraActionInfoFileWriteAction(
        actionToShadow.getOwner(), extraActionInfoFile, actionToShadow));
    extraActionInputs.add(extraActionInfoFile);
    protoOutputs.add(extraActionInfoFile);

    // Expand extra_action specific variables from the provided command-line.
    // See {@link #createExpandedCommand} for list of supported variables.
    String command = createExpandedCommand(owningRule, actionToShadow, extraActionInfoFile);

    CommandHelper commandHelper = CommandHelper.builder(owningRule).build();

    // Multiple actions in the same configured target need to have different names for the artifact
    // that might be created here, so we append something that should be unique for each action.
    String actionUniquifier =
        actionToShadow.getPrimaryOutput().getExecPath().getBaseName()
            + "."
            + actionToShadow.getKey(owningRule.getActionKeyContext(), /*artifactExpander=*/ null);
    BashCommandConstructor constructor =
        CommandHelper.buildBashCommandConstructor(
            executionInfo, shExecutable, "." + actionUniquifier + ".extra_action_script.sh");
    List<String> argv = commandHelper.buildCommandLine(command, extraActionInputs, constructor);

    String commandMessage = String.format("Executing extra_action %s on %s", label, ownerLabel);
    owningRule.registerAction(
        new ExtraAction(
            extraActionInputs.build(),
            runfilesSupplier,
            extraActionOutputs,
            actionToShadow,
            createDummyOutput,
            CommandLine.of(argv),
            owningRule.getConfiguration().getActionEnvironment(),
            owningRule.getConfiguration().modifiedExecutionInfo(executionInfo, label.getName()),
            commandMessage,
            label.getName()));

    return ImmutableSet.<Artifact.DerivedArtifact>builder()
        .addAll(extraActionOutputs)
        .addAll(protoOutputs)
        .build();
  }

  /**
   * Expand extra_action specific variables: $(EXTRA_ACTION_FILE): expands to a path of the file
   * containing a protocol buffer describing the action being shadowed. $(output <out_template>):
   * expands the output template to the execPath of the file. e.g. $(output $(ACTION_ID).out) ->
   * <build_path>/extra_actions/bar/baz/devtools/build/test_A41234.out
   */
  private String createExpandedCommand(
      RuleContext owningRule, Action action, Artifact extraActionInfoFile)
      throws InterruptedException {
    String realCommand = command.replace(
        "$(EXTRA_ACTION_FILE)", extraActionInfoFile.getExecPathString());

    for (String outputTemplate : outputTemplates) {
      String outFile = getExtraActionOutputArtifact(owningRule, action, outputTemplate)
        .getExecPathString();
      realCommand = realCommand.replace("$(output " + outputTemplate + ")", outFile);
    }
    return realCommand;
  }

  /**
   * Creates an output artifact for the extra_action based on the output_template. The path will be
   * in the following form: <output
   * dir>/<target-configuration-specific-path>/extra_actions/<extra_action_label>/ +
   * <configured_target_label>/<expanded_template>
   *
   * <p>The template can use the following variables: $(ACTION_ID): a unique id for the
   * extra_action.
   *
   * <p>Sample: extra_action: foo/bar:extra template: $(ACTION_ID).analysis target: foo/bar:main
   * expands to: output/configuration/extra_actions/\
   * foo/bar/extra/foo/bar/4683026f7ac1dd1a873ccc8c3d764132.analysis
   */
  private Artifact.DerivedArtifact getExtraActionOutputArtifact(
      RuleContext ruleContext, Action action, String template) throws InterruptedException {
    String actionId =
        getActionId(ruleContext.getActionKeyContext(), ruleContext.getActionOwner(), action);

    template = template.replace("$(ACTION_ID)", actionId);
    template = template.replace("$(OWNER_LABEL_DIGEST)", getOwnerDigest(ruleContext));

    return getRootRelativePath(template, ruleContext);
  }

  private Artifact.DerivedArtifact getRootRelativePath(String template, RuleContext ruleContext) {
    PathFragment extraActionPackageFragment = label.getPackageIdentifier().getSourceRoot();
    PathFragment extraActionPrefix = extraActionPackageFragment.getRelative(label.getName());
    PathFragment rootRelativePath = PathFragment.create("extra_actions")
        .getRelative(extraActionPrefix)
        .getRelative(ruleContext.getPackageDirectory())
        .getRelative(template);
    // We need to use getDerivedArtifact here because extra actions are at
    // <EXTRA ACTION LABEL> / <RULE LABEL> instead of <RULE LABEL> / <EXTRA ACTION LABEL>. Bummer.
    return ruleContext.getAnalysisEnvironment().getDerivedArtifact(rootRelativePath,
        ruleContext.getConfiguration().getOutputDirectory(ruleContext.getRule().getRepository()));
  }

  /**
   * Calculates a digest representing the rule context.  We use the digest instead of the
   * original value as the original value might lead to a filename that is too long.
   * By using a digest, tools can deterministically find all extra_action outputs for a given
   * target, without having to open every file in the package.
   */
  private static String getOwnerDigest(RuleContext ruleContext) {
    Fingerprint f = new Fingerprint();
    f.addString(ruleContext.getLabel().toString());
    return f.hexDigestAndReset();
  }

  /**
   * Creates a unique id for the action shadowed by this extra_action.
   *
   * <p>We need to have a unique id for the extra_action to use. We build this from the owner's
   * label and the shadowed action id (which is only guaranteed to be unique per target). Together
   * with the subfolder matching the original target's package name, we believe this is enough of a
   * uniqueness guarantee.
   */
  @VisibleForTesting
  public static String getActionId(
      ActionKeyContext actionKeyContext, ActionOwner owner, Action action)
      throws InterruptedException {
    Fingerprint f = new Fingerprint();
    f.addString(owner.getLabel().toString());
    ImmutableList<AspectDescriptor> aspectDescriptors = owner.getAspectDescriptors();
    f.addInt(aspectDescriptors.size());
    for (AspectDescriptor aspectDescriptor : aspectDescriptors) {
      f.addString(aspectDescriptor.getDescription());
    }
    f.addString(action.getKey(actionKeyContext, /*artifactExpander=*/ null));
    return f.hexDigestAndReset();
  }
}
