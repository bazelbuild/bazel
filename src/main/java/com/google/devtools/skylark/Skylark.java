// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.skylark;

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.Printer;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Skylark is a standalone skylark intepreter. The environment doesn't
 * contain Bazel-specific functions and variables. Load statements are
 * forbidden for the moment.
 */
class Skylark {
  private static final String START_PROMPT = ">> ";
  private static final String CONTINUATION_PROMPT = ".. ";

  private static final EventHandler PRINT_HANDLER =
      new EventHandler() {
        @Override
        public void handle(Event event) {
          System.out.println(event.getMessage());
        }
      };

  private final BufferedReader reader =
      new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
  private final Mutability mutability = Mutability.create("interpreter");
  private final Environment env =
      Environment.builder(mutability)
          .setGlobals(Environment.DEFAULT_GLOBALS)
          .setEventHandler(PRINT_HANDLER)
          .build();

  private String prompt() {
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

  /** Provide a REPL evaluating Skylark code. */
  public void readEvalPrintLoop() {
    String input;
    while ((input = prompt()) != null) {
      try {
        Object result = BuildFileAST.eval(env, input);
        if (result != null) {
          System.out.println(Printer.repr(result));
        }
      } catch (Exception e) {
        e.printStackTrace();
      }
    }
  }

  /** Execute a Skylark file. */
  public void execute(String path) {
    String content;
    try {
      content = new String(Files.readAllBytes(Paths.get(path)), StandardCharsets.UTF_8);
      BuildFileAST.eval(env, content);
    } catch (Exception e) {
      System.err.println(e);
    }
  }

  public static void main(String[] args) {
    if (args.length == 0) {
      new Skylark().readEvalPrintLoop();
    } else if (args.length == 1) {
      new Skylark().execute(args[0]);
    } else {
      System.err.println("too many arguments");
      System.exit(1);
    }
  }
}
