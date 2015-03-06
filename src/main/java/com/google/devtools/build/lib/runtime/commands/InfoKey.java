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

import com.google.devtools.build.lib.Constants;

/**
 * An enumeration of all the valid info keys, excepting the make environment
 * variables.
 */
public enum InfoKey {
  // directories
  WORKSPACE("workspace", "The working directory of the server."),
  INSTALL_BASE("install_base", "The installation base directory."),
  OUTPUT_BASE("output_base",
      "A directory for shared " + Constants.PRODUCT_NAME + " state as well as tool and strategy "
      + "specific subdirectories."),
  EXECUTION_ROOT("execution_root",
      "A directory that makes all input and output files visible to the build."),
  OUTPUT_PATH("output_path", "Output directory"),
  BLAZE_BIN(Constants.PRODUCT_NAME + "-bin",
      "Configuration dependent directory for binaries."),
  BLAZE_GENFILES(Constants.PRODUCT_NAME + "-genfiles",
      "Configuration dependent directory for generated files."),
  BLAZE_TESTLOGS(Constants.PRODUCT_NAME + "-testlogs",
      "Configuration dependent directory for logs from a test run."),

  // logs
  COMMAND_LOG("command_log", "Location of the log containg the output from the build commands."),
  MESSAGE_LOG("message_log" ,
      "Location of a log containing machine readable message in LogMessage protobuf format."),

  // misc
  RELEASE("release", Constants.PRODUCT_NAME + " release identifier"),
  SERVER_PID("server_pid", Constants.PRODUCT_NAME + " process id"),
  PACKAGE_PATH("package_path", "The search path for resolving package labels."),

  // memory statistics
  USED_HEAP_SIZE("used-heap-size", "The amount of used memory in bytes. Note that this is not a "
      + "good indicator of the actual memory use, as it includes any remaining inaccessible "
      + "memory."),
  USED_HEAP_SIZE_AFTER_GC("used-heap-size-after-gc",
      "The amount of used memory in bytes after a call to System.gc().", true),
  COMMITTED_HEAP_SIZE("committed-heap-size",
      "The amount of memory in bytes that is committed for the Java virtual machine to use"),
  MAX_HEAP_SIZE("max-heap-size",
      "The maximum amount of memory in bytes that can be used for memory management."),
  GC_COUNT("gc-count", "Number of garbage collection runs."),
  GC_TIME("gc-time", "The approximate accumulated time spend on garbage collection."),

  // These are deprecated, they still work, when explicitly requested, but are not shown by default

  // These keys print multi-line messages and thus don't play well with grep. We don't print them
  // unless explicitly requested
  DEFAULTS_PACKAGE("defaults-package", "Default packages used as implicit dependencies", true),
  BUILD_LANGUAGE("build-language", "A protobuffer with the build language structure", true),
  DEFAULT_PACKAGE_PATH("default-package-path", "The default package path", true);

  private final String name;
  private final String description;
  private final boolean hidden;

  private InfoKey(String name, String description) {
    this(name, description, false);
  }

  private InfoKey(String name, String description, boolean hidden) {
    this.name = name;
    this.description = description;
    this.hidden = hidden;
  }

  public String getName() {
    return name;
  }

  public String getDescription() {
    return description;
  }

  public boolean isHidden() {
    return hidden;
  }
}
