// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.actiongraph.v2;

import static com.google.devtools.build.lib.skyframe.actiongraph.v2.AqueryOutputHandler.OutputType.BINARY;
import static com.google.devtools.build.lib.skyframe.actiongraph.v2.AqueryOutputHandler.OutputType.DELIMITED_BINARY;
import static com.google.devtools.build.lib.skyframe.actiongraph.v2.AqueryOutputHandler.OutputType.TEXT;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.Action;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.ActionGraphContainer;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.AspectDescriptor;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.Configuration;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.DepSetOfFiles;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.PathFragment;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.RuleClass;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.Target;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.PrintTask.ProtoPrintTask;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.PrintTask.StreamedProtoPrintTask;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.PrintTask.TextProtoPrintTask;
import com.google.protobuf.CodedOutputStream;
import com.google.protobuf.Message;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;

/**
 * Manages the various streamed output channels of aquery. This does not support JSON format.
 * TODO(b/274595070) Remove this class after the flag flip.
 */
public class StreamedOutputHandler implements AqueryOutputHandler {
  private final OutputType outputType;
  private final OutputStream outputStream;
  private final CodedOutputStream codedOutputStream;
  private final PrintStream printStream;

  public StreamedOutputHandler(
      OutputType outputType,
      OutputStream outputStream,
      CodedOutputStream codedOutputStream,
      PrintStream printStream) {
    this.outputType = outputType;
    Preconditions.checkArgument(
        outputType == BINARY || outputType == DELIMITED_BINARY || outputType == TEXT,
        "Only proto, streamed_proto, textproto outputs should be streamed.");
    this.outputStream = outputStream;
    this.codedOutputStream = codedOutputStream;
    this.printStream = printStream;
  }

  @Override
  public void outputArtifact(Artifact message) throws IOException {
    printMessage(message, ActionGraphContainer.ARTIFACTS_FIELD_NUMBER, "artifacts");
  }

  @Override
  public void outputAction(Action message) throws IOException {
    printMessage(message, ActionGraphContainer.ACTIONS_FIELD_NUMBER, "actions");
  }

  @Override
  public void outputTarget(Target message) throws IOException {
    printMessage(message, ActionGraphContainer.TARGETS_FIELD_NUMBER, "targets");
  }

  @Override
  public void outputDepSetOfFiles(DepSetOfFiles message) throws IOException {
    printMessage(message, ActionGraphContainer.DEP_SET_OF_FILES_FIELD_NUMBER, "dep_set_of_files");
  }

  @Override
  public void outputConfiguration(Configuration message) throws IOException {
    printMessage(message, ActionGraphContainer.CONFIGURATION_FIELD_NUMBER, "configuration");
  }

  @Override
  public void outputAspectDescriptor(AspectDescriptor message) throws IOException {
    printMessage(
        message, ActionGraphContainer.ASPECT_DESCRIPTORS_FIELD_NUMBER, "aspect_descriptors");
  }

  @Override
  public void outputRuleClass(RuleClass message) throws IOException {
    printMessage(message, ActionGraphContainer.RULE_CLASSES_FIELD_NUMBER, "rule_classes");
  }

  @Override
  public void outputPathFragment(PathFragment message) throws IOException {
    printMessage(message, ActionGraphContainer.PATH_FRAGMENTS_FIELD_NUMBER, "path_fragments");
  }

  /**
   * Prints the Message to the appropriate output channel.
   *
   * @param message The message to be printed.
   */
  private void printMessage(Message message, int fieldNumber, String messageLabel)
      throws IOException {
    switch (outputType) {
      case BINARY:
        ProtoPrintTask.print(codedOutputStream, message, fieldNumber);
        break;
      case DELIMITED_BINARY:
        StreamedProtoPrintTask.print(outputStream, message, fieldNumber);
        break;
      case TEXT:
        TextProtoPrintTask.print(printStream, message, messageLabel);
        break;
      default:
        throw new IllegalStateException("Unknown outputType " + outputType.formatName());
    }
  }

  @Override
  public void close() throws IOException {
    outputStream.flush();
    codedOutputStream.flush();
    printStream.flush();
  }
}
