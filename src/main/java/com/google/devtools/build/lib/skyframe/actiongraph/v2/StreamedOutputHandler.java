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

import com.google.devtools.build.lib.analysis.AnalysisProtosV2.ActionGraphComponent;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.ActionGraphContainer;
import com.google.protobuf.CodedOutputStream;
import com.google.protobuf.util.JsonFormat;
import java.io.IOException;
import java.io.PrintStream;

/** Manages the various streamed output channels of aquery. */
public class StreamedOutputHandler {
  /** Defines the types of proto output this class can handle. */
  public enum OutputType {
    BINARY("proto"),
    TEXT("textproto"),
    JSON("jsonproto");

    private final String formatName;

    OutputType(String formatName) {
      this.formatName = formatName;
    }

    public String formatName() {
      return formatName;
    }
  }

  private final OutputType outputType;
  private final CodedOutputStream outputStream;
  private final PrintStream printStream;
  private final JsonFormat.Printer jsonPrinter = JsonFormat.printer();

  public StreamedOutputHandler(
      OutputType outputType, CodedOutputStream outputStream, PrintStream printStream) {
    this.outputType = outputType;
    this.outputStream = outputStream;
    this.printStream = printStream;
  }

  /**
   * Prints the ActionGraphComponent to the appropriate output channel.
   *
   * @param message The message to be printed.
   */
  public void printActionGraphComponent(ActionGraphComponent message) throws IOException {
    switch (outputType) {
      case BINARY:
        outputStream.writeMessage(
            ActionGraphContainer.ACTION_GRAPH_COMPONENTS_FIELD_NUMBER, message);
        break;
      case TEXT:
        printStream.print(wrapperActionGraphComponent(message));
        break;
      case JSON:
        jsonPrinter.appendTo(message, printStream);
        printStream.println();
        break;
    }
  }

  private static String wrapperActionGraphComponent(ActionGraphComponent message) {
    return "action_graph_components {\n" + message + "}\n";
  }

  /** Called at the end of the query process. */
  public void close() throws IOException {
    switch (outputType) {
      case BINARY:
        outputStream.flush();
        break;
      case TEXT:
      case JSON:
        printStream.flush();
        printStream.close();
        break;
    }
  }
}
