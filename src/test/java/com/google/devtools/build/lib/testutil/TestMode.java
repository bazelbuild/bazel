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
package com.google.devtools.build.lib.testutil;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.skylark.SkylarkModules;
import com.google.devtools.build.lib.packages.StarlarkSemanticsOptions;
import com.google.devtools.build.lib.syntax.Module;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.common.options.OptionsParser;
import java.util.Map;

/**
 * Describes a particular testing mode by determining how the appropriate {@code StarlarkThread} has
 * to be created
 */
public abstract class TestMode {
  private static StarlarkSemantics parseSkylarkSemantics(String... skylarkOptions)
      throws Exception {
    OptionsParser parser =
        OptionsParser.builder().optionsClasses(StarlarkSemanticsOptions.class).build();
    parser.parse(skylarkOptions);
    return parser.getOptions(StarlarkSemanticsOptions.class).toSkylarkSemantics();
  }

  public static final TestMode BUILD =
      new TestMode() {
        @Override
        public StarlarkThread createStarlarkThread(
            StarlarkThread.PrintHandler printHandler,
            Map<String, Object> builtins,
            String... skylarkOptions)
            throws Exception {
          StarlarkThread thread =
              StarlarkThread.builder(Mutability.create("build test"))
                  .setGlobals(createModule(builtins))
                  .setSemantics(TestMode.parseSkylarkSemantics(skylarkOptions))
                  .build();
          thread.setPrintHandler(printHandler);
          return thread;
        }
      };

  public static final TestMode SKYLARK =
      new TestMode() {
        @Override
        public StarlarkThread createStarlarkThread(
            StarlarkThread.PrintHandler printHandler,
            Map<String, Object> builtins,
            String... skylarkOptions)
            throws Exception {
          StarlarkThread thread =
              StarlarkThread.builder(Mutability.create("skylark test"))
                  .setGlobals(createModule(builtins))
                  .setSemantics(TestMode.parseSkylarkSemantics(skylarkOptions))
                  .build();
          thread.setPrintHandler(printHandler);
          return thread;
        }
      };

  private static Module createModule(Map<String, Object> builtins) {
    ImmutableMap.Builder<String, Object> envBuilder = ImmutableMap.builder();

    SkylarkModules.addSkylarkGlobalsToBuilder(envBuilder);
    envBuilder.putAll(builtins);
    return Module.createForBuiltins(envBuilder.build());
  }

  public abstract StarlarkThread createStarlarkThread(
      StarlarkThread.PrintHandler printHandler,
      Map<String, Object> builtins,
      String... skylarkOptions)
      throws Exception;
}
