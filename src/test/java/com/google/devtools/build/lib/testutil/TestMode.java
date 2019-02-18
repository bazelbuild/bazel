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
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.SkylarkSemanticsOptions;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Environment.GlobalFrame;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.common.options.OptionsParser;
import java.util.Map;

/**
 * Describes a particular testing mode by determining how the
 * appropriate {@code Environment} has to be created
 */
public abstract class TestMode {
  private static StarlarkSemantics parseSkylarkSemantics(String... skylarkOptions)
      throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(SkylarkSemanticsOptions.class);
    parser.parse(skylarkOptions);
    return parser.getOptions(SkylarkSemanticsOptions.class).toSkylarkSemantics();
  }

  public static final TestMode BUILD =
      new TestMode() {
        @Override
        public Environment createEnvironment(EventHandler eventHandler,
            Map<String, Object> builtins,
            String... skylarkOptions)
            throws Exception {
          return Environment.builder(Mutability.create("build test"))
              .setGlobals(createGlobalFrame(builtins))
              .setEventHandler(eventHandler)
              .setSemantics(TestMode.parseSkylarkSemantics(skylarkOptions))
              .build();
        }
      };

  public static final TestMode SKYLARK =
      new TestMode() {
        @Override
        public Environment createEnvironment(EventHandler eventHandler,
            Map<String, Object> builtins, String... skylarkOptions)
            throws Exception {
          return Environment.builder(Mutability.create("skylark test"))
              .setGlobals(createGlobalFrame(builtins))
              .setEventHandler(eventHandler)
              .setSemantics(TestMode.parseSkylarkSemantics(skylarkOptions))
              .build();
        }
      };

  private static GlobalFrame createGlobalFrame(Map<String, Object> builtins) {
    ImmutableMap.Builder<String, Object> envBuilder = ImmutableMap.builder();

    SkylarkModules.addSkylarkGlobalsToBuilder(envBuilder);
    envBuilder.putAll(builtins);
    return GlobalFrame.createForBuiltins(envBuilder.build());
  }

  public abstract Environment createEnvironment(EventHandler eventHandler,
      Map<String, Object> builtins,
      String... skylarkOptions) throws Exception;
}
