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
import java.util.ArrayList;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Callable;

/** Manages the various streamed output channels of aquery. This does not support JSON format. */
public class StreamedConsumingOutputHandler implements AqueryConsumingOutputHandler {

  public static final PrintTask POISON_PILL = ProtoPrintTask.create(null, 0);
  private final OutputType outputType;
  private final OutputStream outputStream;
  private final CodedOutputStream codedOutputStream;
  private final PrintStream printStream;

  private final Object exitLock = new Object();
  private volatile boolean readyToExit = false;
  private final BlockingQueue<PrintTask> queue;

  public StreamedConsumingOutputHandler(
      OutputType outputType,
      OutputStream outputStream,
      CodedOutputStream codedOutputStream,
      PrintStream printStream,
      BlockingQueue<PrintTask> queue) {
    this.outputType = outputType;
    Preconditions.checkArgument(
        outputType == BINARY || outputType == DELIMITED_BINARY || outputType == TEXT,
        "Only proto, streamed_proto and textproto outputs should be streamed.");
    this.outputStream = outputStream;
    this.codedOutputStream = codedOutputStream;
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
  public Callable<Void> startConsumer() {
    return new AqueryOutputTaskConsumer(queue);
  }

  @Override
  public void stopConsumer(boolean discardRemainingTasks) throws InterruptedException {
    if (discardRemainingTasks) {
      queue.drainTo(new ArrayList<>());
    }
    // This lock ensures that the method actually waits until the consumer properly exits,
    // which prevents a race condition with the #close() method below.
    synchronized (exitLock) {
      queue.put(POISON_PILL);
      while (!readyToExit) {
        exitLock.wait();
      }
    }
  }

  /** Construct the printing task and put it in the queue. */
  void addTaskToQueue(Message message, int fieldNumber, String messageLabel) {
    // This means that there was an exception in the consumer.
    if (readyToExit) {
      return;
    }
    PrintTask task;
    switch (outputType) {
      case BINARY:
        task = ProtoPrintTask.create(message, fieldNumber);
        break;
      case DELIMITED_BINARY:
        task = StreamedProtoPrintTask.create(message, fieldNumber);
        break;
      case TEXT:
        task = TextProtoPrintTask.create(message, messageLabel);
        break;
      default:
        throw new IllegalStateException("Unknown outputType: " + outputType);
    }
    try {
      queue.put(task);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
  }

  @Override
  public void close() throws IOException {
    switch (outputType) {
      case BINARY:
        codedOutputStream.flush();
        break;
      case DELIMITED_BINARY:
        outputStream.flush();
        break;
      case TEXT:
        printStream.flush();
        break;
      default:
        throw new IllegalStateException("Unknown outputType: " + outputType);
    }
  }

  // Only runs on 1 single thread.
  private class AqueryOutputTaskConsumer implements Callable<Void> {
    private final BlockingQueue<PrintTask> queue;

    AqueryOutputTaskConsumer(BlockingQueue<PrintTask> queue) {
      this.queue = queue;
    }

    @Override
    public Void call() throws InterruptedException, IOException {
      try {
        while (true) {
          PrintTask nextTask = queue.take();

          if (nextTask.equals(POISON_PILL)) {
            synchronized (exitLock) {
              readyToExit = true;
              exitLock.notify();
            }
            return null;
          }
          switch (outputType) {
            case BINARY:
              ProtoPrintTask.print(codedOutputStream, (ProtoPrintTask) nextTask);
              break;
            case DELIMITED_BINARY:
              StreamedProtoPrintTask.print(outputStream, (StreamedProtoPrintTask) nextTask);
              break;
            case TEXT:
              TextProtoPrintTask.print(printStream, (TextProtoPrintTask) nextTask);
              break;
            default:
              throw new IllegalStateException("Unknown outputType " + outputType.formatName());
          }
        }
      } finally {
        // In case of an exception.
        readyToExit = true;
      }
    }
  }
}
