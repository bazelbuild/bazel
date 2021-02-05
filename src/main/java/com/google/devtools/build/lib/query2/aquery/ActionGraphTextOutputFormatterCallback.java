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
package com.google.devtools.build.lib.query2.aquery;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.analysis.AspectValue;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetValue;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.util.CommandDescriptionForm;
import com.google.devtools.build.lib.util.CommandFailureUtils;
import com.google.devtools.build.lib.util.ShellEscaper;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.stream.Collectors;

/** Output callback for aquery, prints human readable output. */
class ActionGraphTextOutputFormatterCallback extends AqueryThreadsafeCallback {

  private final ActionKeyContext actionKeyContext = new ActionKeyContext();
  private final AqueryActionFilter actionFilters;
  private Map<String, String> paramFileNameToContentMap;

  ActionGraphTextOutputFormatterCallback(
      ExtendedEventHandler eventHandler,
      AqueryOptions options,
      OutputStream out,
      SkyframeExecutor skyframeExecutor,
      TargetAccessor<ConfiguredTargetValue> accessor,
      AqueryActionFilter actionFilters) {
    super(eventHandler, options, out, skyframeExecutor, accessor);
    this.actionFilters = actionFilters;
  }

  @Override
  public String getName() {
    return "text";
  }

  @Override
  public void processOutput(Iterable<ConfiguredTargetValue> partialResult)
      throws IOException, InterruptedException {
    try {
      // Enabling includeParamFiles should enable includeCommandline by default.
      options.includeCommandline |= options.includeParamFiles;

      for (ConfiguredTargetValue configuredTargetValue : partialResult) {
        for (ActionAnalysisMetadata action : configuredTargetValue.getActions()) {
          writeAction(action, printStream);
        }
        if (options.useAspects) {
          if (configuredTargetValue.getConfiguredTarget() instanceof RuleConfiguredTarget) {
            for (AspectValue aspectValue : accessor.getAspectValues(configuredTargetValue)) {
              for (ActionAnalysisMetadata action : aspectValue.getActions()) {
                writeAction(action, printStream);
              }
            }
          }
        }
      }
    } catch (CommandLineExpansionException e) {
      throw new IOException(e.getMessage());
    }
  }

  private void writeAction(ActionAnalysisMetadata action, PrintStream printStream)
      throws IOException, CommandLineExpansionException, InterruptedException {
    if (options.includeParamFiles && action instanceof ParameterFileWriteAction) {
      ParameterFileWriteAction parameterFileWriteAction = (ParameterFileWriteAction) action;

      String fileContent = String.join(" \\\n    ", parameterFileWriteAction.getArguments());
      String paramFileName = action.getPrimaryOutput().getExecPathString();

      getParamFileNameToContentMap().put(paramFileName, fileContent);
    }

    if (!AqueryUtils.matchesAqueryFilters(action, actionFilters)) {
      return;
    }

    ActionOwner actionOwner = action.getOwner();
    StringBuilder stringBuilder = new StringBuilder();
    stringBuilder
        .append(action.prettyPrint())
        .append('\n')
        .append("  Mnemonic: ")
        .append(action.getMnemonic())
        .append('\n');

    if (actionOwner != null) {
      BuildEvent configuration = actionOwner.getConfiguration();
      BuildEventStreamProtos.Configuration configProto =
          configuration.asStreamProto(/*context=*/ null).getConfiguration();

      stringBuilder
          .append("  Target: ")
          .append(actionOwner.getLabel())
          .append('\n')
          .append("  Configuration: ")
          .append(configProto.getMnemonic())
          .append('\n');

      // In the case of aspect-on-aspect, AspectDescriptors are listed in
      // topological order of the dependency graph.
      // e.g. [A -> B] would imply that aspect A is applied on top of aspect B.
      ImmutableList<AspectDescriptor> aspectDescriptors =
          actionOwner.getAspectDescriptors().reverse();
      if (!aspectDescriptors.isEmpty()) {
        stringBuilder
            .append("  AspectDescriptors: [")
            .append(
                aspectDescriptors.stream()
                    .map(
                        aspectDescriptor -> {
                          StringBuilder aspectDescription = new StringBuilder();
                          aspectDescription
                              .append(aspectDescriptor.getAspectClass().getName())
                              .append('(')
                              .append(
                                  aspectDescriptor
                                      .getParameters()
                                      .getAttributes()
                                      .entries()
                                      .stream()
                                      .map(
                                          parameter ->
                                              parameter.getKey()
                                                  + "='"
                                                  + parameter.getValue()
                                                  + "'")
                                      .collect(Collectors.joining(", ")))
                              .append(')');
                          return aspectDescription.toString();
                        })
                    .collect(Collectors.joining("\n    -> ")))
            .append("]\n");
      }
    }

    if (action instanceof ActionExecutionMetadata) {
      ActionExecutionMetadata actionExecutionMetadata = (ActionExecutionMetadata) action;
      stringBuilder
          .append("  ActionKey: ")
          .append(actionExecutionMetadata.getKey(actionKeyContext, /*artifactExpander=*/ null))
          .append('\n');
    }

    if (options.includeArtifacts) {
      stringBuilder
          .append("  Inputs: [")
          .append(
              action.getInputs().toList().stream()
                  .map(input -> input.getExecPathString())
                  .sorted()
                  .collect(Collectors.joining(", ")))
          .append("]\n")
          .append("  Outputs: [")
          .append(
              action.getOutputs().stream()
                  .map(
                      output ->
                          output.isTreeArtifact()
                              ? output.getExecPathString() + " (TreeArtifact)"
                              : output.getExecPathString())
                  .sorted()
                  .collect(Collectors.joining(", ")))
          .append("]\n");
    }

    if (action instanceof AbstractAction) {
      AbstractAction abstractAction = (AbstractAction) action;
      // TODO(twerth): This handles the fixed environment. We probably want to output the inherited
      // environment as well.
      Iterable<Map.Entry<String, String>> fixedEnvironment =
          abstractAction.getEnvironment().getFixedEnv().toMap().entrySet();
      if (!Iterables.isEmpty(fixedEnvironment)) {
        stringBuilder
            .append("  Environment: [")
            .append(
                Streams.stream(fixedEnvironment)
                    .map(
                        environmentVariable ->
                            environmentVariable.getKey() + "=" + environmentVariable.getValue())
                    .sorted()
                    .collect(Collectors.joining(", ")))
            .append("]\n");
      }
      if (abstractAction.getExecutionInfo() != null) {
        Set<Entry<String, String>> executionInfoSpecifiers =
            abstractAction.getExecutionInfo().entrySet();
        if (!executionInfoSpecifiers.isEmpty()) {
          stringBuilder
              .append("  ExecutionInfo: {")
              .append(
                  executionInfoSpecifiers.stream()
                      .sorted(Map.Entry.comparingByKey())
                      .map(
                          e ->
                              String.format(
                                  "%s: %s",
                                  ShellEscaper.escapeString(e.getKey()),
                                  ShellEscaper.escapeString(e.getValue())))
                      .collect(Collectors.joining(", ")))
              .append("}\n");
        }
      }
    }
    if (options.includeCommandline && action instanceof CommandAction) {
      stringBuilder
          .append("  Command Line: ")
          .append(
              CommandFailureUtils.describeCommand(
                  CommandDescriptionForm.COMPLETE,
                  /* prettyPrintArgs= */ true,
                  ((CommandAction) action).getArguments(),
                  /* environment= */ null,
                  /* cwd= */ null))
          .append("\n");
    }

    if (options.includeParamFiles) {
      // Assumption: if an Action takes a param file as an input, it will be used
      // to provide params to the command.
      for (Artifact input : action.getInputs().toList()) {
        String inputFileName = input.getExecPathString();
        if (getParamFileNameToContentMap().containsKey(inputFileName)) {
          stringBuilder
              .append("  Params File Content (")
              .append(inputFileName)
              .append("):\n    ")
              .append(getParamFileNameToContentMap().get(inputFileName))
              .append("\n");
        }
      }
    }
    Map<String, String> executionInfo = action.getExecutionInfo();
    if (executionInfo != null && !executionInfo.isEmpty()) {
      stringBuilder
          .append("  ExecutionInfo: {")
          .append(
              executionInfo.entrySet().stream()
                  .sorted(Map.Entry.comparingByKey())
                  .map(
                      e ->
                          String.format(
                              "%s: %s",
                              ShellEscaper.escapeString(e.getKey()),
                              ShellEscaper.escapeString(e.getValue())))
                  .collect(Collectors.joining(", ")))
          .append("}\n");
    }

    stringBuilder.append('\n');

    printStream.write(stringBuilder.toString().getBytes(UTF_8));
  }

  /** Lazy initialization of paramFileNameToContentMap. */
  private Map<String, String> getParamFileNameToContentMap() {
    if (paramFileNameToContentMap == null) {
      paramFileNameToContentMap = new HashMap<>();
    }
    return paramFileNameToContentMap;
  }
}
