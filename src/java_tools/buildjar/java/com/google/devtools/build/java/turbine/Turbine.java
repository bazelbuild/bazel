// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.java.turbine;

import static com.google.common.collect.Iterables.getOnlyElement;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.CharMatcher;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.java.turbine.javac.JavacTurbine;
import com.google.devtools.build.java.turbine.javac.JavacTurbine.Result;
import com.google.turbine.diag.TurbineError;
import com.google.turbine.main.Main;
import com.google.turbine.options.TurbineOptions;
import com.google.turbine.options.TurbineOptionsParser;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import javax.annotation.Nullable;

/**
 * A turbine entry point that falls back to javac-turbine for failures, and for compilations that
 * use API-generating annotation processors.
 */
public class Turbine {

  public static void main(String[] args) throws Exception {
    System.exit(
        new Turbine(
                /* bugMessage= */ "An exception has occurred in turbine.",
                /* unhelpfulMessage= */ "",
                /* fixImportCommand= */ null)
            .compile(TurbineOptionsParser.parse(ImmutableList.copyOf(args))));
  }

  /** Formats a suggested fix for missing import errors. */
  @FunctionalInterface
  public interface FixImportCommand {
    String formatCommand(String type, String target);
  }

  private final String bugMessage;
  private final String unhelpfulMessage;
  private final @Nullable FixImportCommand fixImportCommand;

  public Turbine(
      String bugMessage, String unhelpfulMessage, @Nullable FixImportCommand fixImportCommand) {
    this.bugMessage = bugMessage;
    this.unhelpfulMessage = unhelpfulMessage;
    this.fixImportCommand = fixImportCommand;
  }

  public int compile(TurbineOptions options) throws IOException {
    return compile(
        options,
        new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.err, UTF_8)), true));
  }

  public int compile(TurbineOptions options, PrintWriter out) throws IOException {
    Throwable turbineCrash = null;
    try {
      if (Main.compile(options)) {
        return 0;
      }
      // fall back to javac for API-generating processors
    } catch (TurbineError e) {
      switch (e.kind()) {
        case TYPE_PARAMETER_QUALIFIER:
          out.println(e.getMessage());
          return 1;
        default:
          turbineCrash = e;
          break;
      }
    } catch (Throwable t) {
      turbineCrash = t;
    }
    if (!options.javacFallback()) {
      if (turbineCrash instanceof TurbineError) {
        TurbineError turbineError = (TurbineError) turbineCrash;
        out.println();
        out.println(turbineError.getMessage());
        switch (turbineError.kind()) {
          case SYMBOL_NOT_FOUND:
            if (fixImportCommand != null && options.targetLabel().isPresent()) {
              out.println();
              Object arg = getOnlyElement(turbineError.args());
              out.println("\033[35m\033[1m** Command to add missing dependencies:\033[0m\n");
              out.println(
                  fixImportCommand.formatCommand(
                      CharMatcher.anyOf("$/").replaceFrom(arg.toString(), '.'),
                      options.targetLabel().get()));
              out.println();
            }
            break;
          default: // fall out
        }
        out.println(unhelpfulMessage);
      } else if (turbineCrash != null) {
        out.println(bugMessage);
        turbineCrash.printStackTrace(out);
      }
      return 1;
    }
    Result result = JavacTurbine.compile(options);
    if (result == Result.OK_WITH_REDUCED_CLASSPATH && turbineCrash != null) {
      out.println(bugMessage);
      turbineCrash.printStackTrace(out);
      result = Result.ERROR;
    }
    return result.exitCode();
  }
}
