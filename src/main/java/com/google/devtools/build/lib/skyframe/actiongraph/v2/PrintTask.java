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

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.Action;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.ActionGraphContainer;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.AspectDescriptor;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.Configuration;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.DepSetOfFiles;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.PathFragment;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.RuleClass;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.Target;
import com.google.protobuf.CodedOutputStream;
import com.google.protobuf.Message;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import javax.annotation.Nullable;

/**
 * Represent a task to be consumed by a {@link AqueryConsumingOutputHandler}.
 *
 * <p>We have separate Proto/TextProto subclasses to reduce some memory waste: we'll never need both
 * the fieldNumber and the messageLabel in a PrintTask.
 */
@SuppressWarnings("InterfaceWithOnlyStatics")
public interface PrintTask {

  /** A task for the proto format. */
  @AutoValue
  abstract class ProtoPrintTask implements PrintTask {
    @Nullable
    abstract Message message();

    abstract int fieldNumber();

    public static ProtoPrintTask create(Message message, int fieldNumber) {
      return new AutoValue_PrintTask_ProtoPrintTask(message, fieldNumber);
    }

    public static void print(CodedOutputStream codedOutputStream, ProtoPrintTask task)
        throws IOException {
      print(codedOutputStream, task.message(), task.fieldNumber());
    }

    public static void print(CodedOutputStream codedOutputStream, Message message, int fieldNumber)
        throws IOException {
      codedOutputStream.writeMessage(fieldNumber, message);
    }
  }

  /** A task for the streamed_proto format. */
  @AutoValue
  abstract class StreamedProtoPrintTask implements PrintTask {
    @Nullable
    abstract Message message();

    abstract int fieldNumber();

    public static StreamedProtoPrintTask create(Message message, int fieldNumber) {
      return new AutoValue_PrintTask_StreamedProtoPrintTask(message, fieldNumber);
    }

    public static void print(OutputStream out, StreamedProtoPrintTask task) throws IOException {
      print(out, task.message(), task.fieldNumber());
    }

    public static void print(OutputStream out, Message message, int fieldNumber)
        throws IOException {
      ActionGraphContainer.Builder builder = ActionGraphContainer.newBuilder();
      switch (fieldNumber) {
        case ActionGraphContainer.ARTIFACTS_FIELD_NUMBER:
          builder.addArtifacts((Artifact) message);
          break;
        case ActionGraphContainer.ACTIONS_FIELD_NUMBER:
          builder.addActions((Action) message);
          break;
        case ActionGraphContainer.TARGETS_FIELD_NUMBER:
          builder.addTargets((Target) message);
          break;
        case ActionGraphContainer.DEP_SET_OF_FILES_FIELD_NUMBER:
          builder.addDepSetOfFiles((DepSetOfFiles) message);
          break;
        case ActionGraphContainer.CONFIGURATION_FIELD_NUMBER:
          builder.addConfiguration((Configuration) message);
          break;
        case ActionGraphContainer.ASPECT_DESCRIPTORS_FIELD_NUMBER:
          builder.addAspectDescriptors((AspectDescriptor) message);
          break;
        case ActionGraphContainer.RULE_CLASSES_FIELD_NUMBER:
          builder.addRuleClasses((RuleClass) message);
          break;
        case ActionGraphContainer.PATH_FRAGMENTS_FIELD_NUMBER:
          builder.addPathFragments((PathFragment) message);
          break;
        default:
          throw new IllegalStateException(
              "Unknown ActionGraphContainer field number " + fieldNumber);
      }
      builder.build().writeDelimitedTo(out);
    }
  }

  /** A task for the textproto format. */
  @AutoValue
  abstract class TextProtoPrintTask implements PrintTask {
    @Nullable
    abstract Message message();

    abstract String messageLabel();

    public static TextProtoPrintTask create(Message message, String messageLabel) {
      return new AutoValue_PrintTask_TextProtoPrintTask(message, messageLabel);
    }

    public static void print(PrintStream printStream, TextProtoPrintTask task) {
      print(printStream, task.message(), task.messageLabel());
    }

    public static void print(PrintStream printStream, Message message, String messageLabel) {
      printStream.print(messageLabel + " {\n" + message + "}\n");
    }
  }
}
