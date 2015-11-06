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
package com.google.devtools.build.lib.syntax;

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;

/**
 * SkylarkShell is a standalone shell executing Skylark. This is intended for
 * testing purposes and not for end-users. This is very limited (environment is
 * almost empty), but it can be used to play with the language and reproduce
 * bugs. Imports and includes are not supported.
 */
class SkylarkShell {

  private static final String START_PROMPT = ">> ";
  private static final String CONTINUATION_PROMPT = ".. ";

  public static final EventHandler PRINT_HANDLER = new EventHandler() {
      @Override
      public void handle(Event event) {
        System.out.println(event.getMessage());
      }
    };

  private final BufferedReader reader = new BufferedReader(
      new InputStreamReader(System.in, Charset.defaultCharset()));
  private final Mutability mutability = Mutability.create("shell");
  private final Environment env = Environment.builder(mutability)
      .setSkylark().setGlobals(Environment.SKYLARK).setEventHandler(PRINT_HANDLER).build();

  public String read() {
    StringBuilder input = new StringBuilder();
    System.out.print(START_PROMPT);
    try {
      while (true) {
        String line = reader.readLine();
        if (line == null) {
          return null;
        }
        if (line.isEmpty()) {
          return input.toString();
        }
        input.append("\n").append(line);
        System.out.print(CONTINUATION_PROMPT);
      }
    } catch (IOException io) {
      io.printStackTrace();
      return null;
    }
  }

  public void readEvalPrintLoop() {
    String input;
    while ((input = read()) != null) {
      try {
        Object result = env.eval(input);
        if (result != null) {
          System.out.println(Printer.repr(result));
        }
      } catch (Exception e) {
        e.printStackTrace();
      }
    }
  }

  public static void main(String[] args) {
    if (args.length > 0 && args[0].equals("--compiler-debug")) {
      UserDefinedFunction.enableCompiler = true;
      UserDefinedFunction.debugCompiler = true;
      UserDefinedFunction.debugCompilerPrintByteCode = true;
    }
    new SkylarkShell().readEvalPrintLoop();
  }
}
