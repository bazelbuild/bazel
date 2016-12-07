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

import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.syntax.BazelLibrary;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Mutability;

/**
 * Describes a particular testing mode by determining how the
 * appropriate {@code Environment} has to be created
 */
public abstract class TestMode {
  public static final TestMode BUILD =
      new TestMode() {
        @Override
        public Environment createEnvironment(EventHandler eventHandler, Environment environment) {
          return Environment.builder(Mutability.create("build test"))
              .setGlobals(environment == null ? BazelLibrary.GLOBALS : environment.getGlobals())
              .setEventHandler(eventHandler)
              .build();
        }
      };

  public static final TestMode SKYLARK =
      new TestMode() {
        @Override
        public Environment createEnvironment(EventHandler eventHandler, Environment environment) {
          return Environment.builder(Mutability.create("skylark test"))
              .setGlobals(environment == null ? BazelLibrary.GLOBALS : environment.getGlobals())
              .setEventHandler(eventHandler)
              .build();
        }
      };

  public abstract Environment createEnvironment(EventHandler eventHandler, Environment environment);
}
