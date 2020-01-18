// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.analysis.AnalysisProtosV2.Action;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.ActionGraphContainer;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.AspectDescriptor;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.Configuration;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.DepSetOfFiles;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.PathFragment;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.RuleClass;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.Target;
import com.google.protobuf.util.JsonFormat;
import java.io.IOException;
import java.io.PrintStream;

/** Handles the monolithic output channel. Supports only the JSON format. */
public class MonolithicOutputHandler implements AqueryOutputHandler {
  private final ActionGraphContainer.Builder actionGraphContainerBuilder =
      ActionGraphContainer.newBuilder();
  private final JsonFormat.Printer jsonPrinter = JsonFormat.printer();
  private final PrintStream printStream;

  public MonolithicOutputHandler(PrintStream printStream) {
    this.printStream = printStream;
  }

  @Override
  public void outputArtifact(Artifact message) throws IOException {
    actionGraphContainerBuilder.addArtifacts(message);
  }

  @Override
  public void outputAction(Action message) throws IOException {
    actionGraphContainerBuilder.addActions(message);
  }

  @Override
  public void outputTarget(Target message) throws IOException {
    actionGraphContainerBuilder.addTargets(message);
  }

  @Override
  public void outputDepSetOfFiles(DepSetOfFiles message) throws IOException {
    actionGraphContainerBuilder.addDepSetOfFiles(message);
  }

  @Override
  public void outputConfiguration(Configuration message) throws IOException {
    actionGraphContainerBuilder.addConfiguration(message);
  }

  @Override
  public void outputAspectDescriptor(AspectDescriptor message) throws IOException {
    actionGraphContainerBuilder.addAspectDescriptors(message);
  }

  @Override
  public void outputRuleClass(RuleClass message) throws IOException {
    actionGraphContainerBuilder.addRuleClasses(message);
  }

  @Override
  public void outputPathFragment(PathFragment message) throws IOException {
    actionGraphContainerBuilder.addPathFragments(message);
  }

  @Override
  public void close() throws IOException {
    jsonPrinter.appendTo(actionGraphContainerBuilder.build(), printStream);
    printStream.println();
  }
}
