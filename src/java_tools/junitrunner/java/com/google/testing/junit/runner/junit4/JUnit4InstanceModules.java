// Copyright 2012 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.junit4;

import java.util.Arrays;
import java.util.List;

/**
 * Utility classes which hold state or are, for testing purposes, implemented with non-static
 * provider methods.  These types are collected here so they can be cleanly named in the
 * component builder, but still be obvious in module includes and component declarations.
 * These are Dagger legacy modules.
 */
public final class JUnit4InstanceModules {

  /**
   * A module which supplies a JUnit4Config object, which can be overridden at test-time.
   */
  public static final class Config {
    private final List<String> args;

    /**
     * Creates a module that can provide a {@link JUnit4Config} from supplied command-line
     * arguments
     */
    public Config(String... args) {
      this.args = Arrays.asList(args);
    }

    JUnit4Options options() {
      return JUnit4Options.parse(System.getenv(), args);
    }
  }

  private JUnit4InstanceModules() {}
}
