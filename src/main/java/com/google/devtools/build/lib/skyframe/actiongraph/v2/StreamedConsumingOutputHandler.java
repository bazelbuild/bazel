// Copyright 2023 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.skyframe.actiongraph.v2.PrintTask.TextProtoPrintTask;
import com.google.protobuf.CodedOutputStream;
import com.google.protobuf.Message;
import java.io.IOException;
import java.io.PrintStream;
import java.util.concurrent.BlockingQueue;

/** Manages the various streamed output channels of aquery. This does not support JSON format. */
public class StreamedConsumingOutputHandler implements AqueryConsumingOutputHandler {

  public static final PrintTask POISON_PILL = ProtoPrintTask.create(null, 0);
  private final OutputType outputType;
  private final CodedOutputStream outputStream;
  private final PrintStream printStream;

  private final BlockingQueue<PrintTask> queue;

  public StreamedConsumingOutputHandler(
      OutputType outputType,
      CodedOutputStream outputStream,
      PrintStream printStream,
      BlockingQueue<PrintTask> queue) {
    this.outputType = outputType;
    Preconditions.checkArgument(
        outputType == BINARY || outputType == TEXT,
        "Only proto and textproto outputs should be streamed.");
    this.outputStream = outputStream;
    this.printStream = printStream;
    this.queue = queue;
  }

  @Override
  public void outputArtifact(Artifact message) {
    addTaskToQueue(message, ActionGraphContainer.ARTIFACTS_FIELD_NUMBER, "artifacts");
  }

  @Override
  public void outputAction(Action message) {
    addTaskToQueue(message, ActionGraphContainer.ACTIONS_FIELD_NUMBER, "actions");
  }

  @Override
  public void outputTarget(Target message) {
    addTaskToQueue(message, ActionGraphContainer.TARGETS_FIELD_NUMBER, "targets");
  }

  @Override
  public void outputDepSetOfFiles(DepSetOfFiles message) {
    addTaskToQueue(message, ActionGraphContainer.DEP_SET_OF_FILES_FIELD_NUMBER, "dep_set_of_files");
  }

  @Override
  public void outputConfiguration(Configuration message) {
    addTaskToQueue(message, ActionGraphContainer.CONFIGURATION_FIELD_NUMBER, "configuration");
  }

  @Override
  public void outputAspectDescriptor(AspectDescriptor message) {
    addTaskToQueue(
        message, ActionGraphContainer.ASPECT_DESCRIPTORS_FIELD_NUMBER, "aspect_descriptors");
  }

  @Override
  public void outputRuleClass(RuleClass message) {
    addTaskToQueue(message, ActionGraphContainer.RULE_CLASSES_FIELD_NUMBER, "rule_classes");
  }

  @Override
  public void outputPathFragment(PathFragment message) {
    addTaskToQueue(message, ActionGraphContainer.PATH_FRAGMENTS_FIELD_NUMBER, "path_fragments");
  }

  @Override
  public void startConsumer() {
    new Thread(new AqueryOutputTaskConsumer(queue)).start();
  }

  @Override
  public void stopConsumer() throws InterruptedException {
    queue.put(POISON_PILL);
  }

  /** Construct the printing task and put it in the queue. */
  void addTaskToQueue(Message message, int fieldNumber, String messageLabel) {
    try {
      queue.put(
          outputType == BINARY
              ? ProtoPrintTask.create(message, fieldNumber)
              : TextProtoPrintTask.create(message, messageLabel));
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
  }

  @Override
  public void close() throws IOException {
    outputStream.flush();
    printStream.flush();
  }

  private class AqueryOutputTaskConsumer implements Runnable {
    private final BlockingQueue<PrintTask> queue;

    AqueryOutputTaskConsumer(BlockingQueue<PrintTask> queue) {
      this.queue = queue;
    }

    @Override
    public void run() {
      try {
        while (true) {
          PrintTask nextTask = queue.take();

          if (nextTask.equals(POISON_PILL)) {
            return;
          }
          switch (outputType) {
            case BINARY:
              ProtoPrintTask protoPrintTask = (ProtoPrintTask) nextTask;
              outputStream.writeMessage(protoPrintTask.fieldNumber(), protoPrintTask.message());
              break;
            case TEXT:
              TextProtoPrintTask textProtoPrintTask = (TextProtoPrintTask) nextTask;
              printStream.print(
                  textProtoPrintTask.messageLabel()
                      + " {\n"
                      + textProtoPrintTask.message()
                      + "}\n");
              break;
            default:
              throw new IllegalStateException("Unknown outputType " + outputType.formatName());
          }
        }
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      } catch (IOException e) {
        throw new IllegalStateException("Unexpected exception: ", e);
      }
    }
  }
}
