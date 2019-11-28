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
package com.google.devtools.starlark;

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Module;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.ParserInput;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.SyntaxError;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Starlark is a standalone starlark intepreter. The environment doesn't
 * contain Bazel-specific functions and variables. Load statements are
 * forbidden for the moment.
 */
class Starlark {
  private static final String START_PROMPT = ">> ";
  private static final String CONTINUATION_PROMPT = ".. ";

  private static final EventHandler PRINT_HANDLER =
      new EventHandler() {
        @Override
        public void handle(Event event) {
          if (event.getKind() == EventKind.ERROR) {
            System.err.println(event.getMessage());
          } else {
            System.out.println(event.getMessage());
          }
        }
      };

  private static final Charset CHARSET = StandardCharsets.ISO_8859_1;
  private final BufferedReader reader =
      new BufferedReader(new InputStreamReader(System.in, CHARSET));
  private final Mutability mutability = Mutability.create("interpreter");
  private final StarlarkThread thread =
      StarlarkThread.builder(mutability)
          .useDefaultSemantics()
          .setGlobals(
              Module.createForBuiltins(com.google.devtools.build.lib.syntax.Starlark.UNIVERSE))
          .setEventHandler(PRINT_HANDLER)
          .build();

  private String prompt() {
    StringBuilder input = new StringBuilder();
    System.out.print(START_PROMPT);
    try {
      String lineSeparator = "";
      while (true) {
        String line = reader.readLine();
        if (line == null) {
          return null;
        }
        if (line.isEmpty()) {
          return input.toString();
        }
        input.append(lineSeparator).append(line);
        lineSeparator = "\n";
        System.out.print(CONTINUATION_PROMPT);
      }
    } catch (IOException io) {
      io.printStackTrace();
      return null;
    }
  }

  /** Provide a REPL evaluating Starlark code. */
  @SuppressWarnings("CatchAndPrintStackTrace")
  public void readEvalPrintLoop() {
    String line;

    // TODO(adonovan): parse a compound statement, like the Python and
    // go.starlark.net REPLs. This requires a new grammar production, and
    // integration with the lexer so that it consumes new
    // lines only until the parse is complete.

    while ((line = prompt()) != null) {
      ParserInput input = ParserInput.fromLines(line);
      try {
        Object result = EvalUtils.execAndEvalOptionalFinalExpression(input, thread);
        if (result != null) {
          System.out.println(com.google.devtools.build.lib.syntax.Starlark.repr(result));
        }
      } catch (SyntaxError ex) {
        for (Event ev : ex.errors()) {
          System.err.println(ev);
        }
      } catch (EvalException ex) {
        // TODO(adonovan): show Starlark (not Java) stack.
        ex.printStackTrace();
      } catch (InterruptedException ex) {
        System.err.println("Interrupted");
      }
    }
  }

  /** Execute a Starlark file. */
  public int executeFile(String path) {
    String content;
    try {
      content = new String(Files.readAllBytes(Paths.get(path)), CHARSET);
      return execute(content);
    } catch (Exception e) {
      e.printStackTrace();
      return 1;
    }
  }

  /** Execute a Starlark file. */
  public int execute(String content) {
    ParserInput input = ParserInput.create(content, null);
    try {
      EvalUtils.exec(input, thread);
      return 0;
    } catch (SyntaxError ex) {
      for (Event ev : ex.errors()) {
        System.err.println(ev);
      }
      return 1;
    } catch (EvalException e) {
      System.err.println(e.print());
      return 1;
    } catch (Exception e) {
      e.printStackTrace(System.err);
      return 1;
    }
  }

  public static void main(String[] args) {
    int ret = 0;
    if (args.length == 0) {
      new Starlark().readEvalPrintLoop();
    } else if (args.length == 1 && !args[0].equals("-c")) {
      ret = new Starlark().executeFile(args[0]);
    } else if (args.length == 2 && args[0].equals("-c")) {
      ret = new Starlark().execute(args[1]);
    } else {
      System.err.println("USAGE: Starlark [-c \"<cmdLineProgram>\" | <fileName>]");
      ret = 1;
    }
    System.exit(ret);
  }
}
