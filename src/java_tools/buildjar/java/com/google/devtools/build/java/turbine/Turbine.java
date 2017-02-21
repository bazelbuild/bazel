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

import com.google.devtools.build.java.turbine.javac.JavacTurbine;
import com.google.devtools.build.java.turbine.javac.JavacTurbine.Result;
import com.google.turbine.diag.TurbineError;
import com.google.turbine.main.Main;
import com.google.turbine.options.TurbineOptions;
import com.google.turbine.options.TurbineOptionsParser;
import java.io.IOException;
import java.util.Arrays;

/**
 * A turbine entry point that falls back to javac-turbine for failures, and for compilations that
 * use API-generating annotation processors.
 */
public class Turbine {

  public static void main(String[] args) throws Exception {
    System.exit(new Turbine("An exception has occurred in turbine.").compile(args));
  }

  private final String bugMessage;

  public Turbine(String bugMessage) {
    this.bugMessage = bugMessage;
  }

  public int compile(String[] args) throws IOException {
    Throwable turbineCrash = null;
    try {
      if (Main.compile(args)) {
        return 0;
      }
      // fall back to javac for API-generating processors
    } catch (TurbineError e) {
      switch (e.kind()) {
        case TYPE_PARAMETER_QUALIFIER:
          System.err.println(e.getMessage());
          System.exit(1);
          break;
        default:
          turbineCrash = e;
          break;
      }
    } catch (Throwable t) {
      turbineCrash = t;
    }
    Result result = javacTurbineCompile(args);
    if (result == Result.OK_WITH_REDUCED_CLASSPATH && turbineCrash != null) {
      System.err.println(bugMessage);
      turbineCrash.printStackTrace();
      result = Result.ERROR;
    }
    return result.exitCode();
  }

  Result javacTurbineCompile(String[] args) throws IOException {
    TurbineOptions.Builder options = TurbineOptions.builder();
    TurbineOptionsParser.parse(options, Arrays.asList(args));
    return JavacTurbine.compile(options.build());
  }
}
