// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.runtime.commands;

import com.google.devtools.build.docgen.SkylarkDocumentationProcessor;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher.ShutdownBlazeServerException;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsProvider;

import java.util.Map;

/**
 * The 'doc_ext' command, which prints the extension API doc.
 */
@Command(name = "doc_ext",
allowResidue = true,
mustRunInWorkspace = false,
shortDescription = "Prints help for commands, or the index.",
help = "resource:skylark.txt")
public final class SkylarkCommand implements BlazeCommand {

  @Override
  public ExitCode exec(BlazeRuntime runtime, OptionsProvider options)
      throws ShutdownBlazeServerException {
    OutErr outErr = runtime.getReporter().getOutErr();
    if (options.getResidue().isEmpty()) {
      printTopLevelAPIDoc(outErr);
      return ExitCode.SUCCESS;
    }
    if (options.getResidue().size() != 1) {
      runtime.getReporter().handle(Event.error("Cannot specify more than one parameters"));
      return ExitCode.COMMAND_LINE_ERROR;
    }
    return printAPIDoc(options.getResidue().get(0), outErr, runtime.getReporter());
  }

  private ExitCode printAPIDoc(String param, OutErr outErr, Reporter reporter) {
    String params[] = param.split("\\.");
    if (params.length > 2) {
      reporter.handle(Event.error("Identifier not found: " + param));
      return ExitCode.COMMAND_LINE_ERROR;
    }
    SkylarkDocumentationProcessor processor = new SkylarkDocumentationProcessor();
    String doc = processor.getCommandLineAPIDoc(params);
    if (doc == null) {
      reporter.handle(Event.error("Identifier not found: " + param));
      return ExitCode.COMMAND_LINE_ERROR;
    }
    outErr.printOut(doc);
    return ExitCode.SUCCESS;
  }

  private void printTopLevelAPIDoc(OutErr outErr) {
    SkylarkDocumentationProcessor processor = new SkylarkDocumentationProcessor();
    outErr.printOut("Top level language modules, methods and objects:\n\n");
    for (Map.Entry<String, String> entry : processor.collectTopLevelModules().entrySet()) {
      outErr.printOut(entry.getKey() + ": " + entry.getValue());
    }
  }

  @Override
  public void editOptions(BlazeRuntime runtime, OptionsParser optionsParser) {}
}
