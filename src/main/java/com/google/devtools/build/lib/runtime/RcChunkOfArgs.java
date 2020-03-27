// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import java.util.List;

/**
 * We receive the rc file arguments from the client in an order that maintains the location of
 * "import" statements, expanding the imported rc file in place so that its args override previous
 * args in the file and are overridden by later arguments. We cannot group the args by rc file for
 * parsing, as we would lose this ordering, so we store them in these "chunks."
 *
 * <p>Each chunk comes from a single rc file, but the args stored here may not contain the entire
 * file if its contents were interrupted by an import statement.
 */
final class RcChunkOfArgs {
  public RcChunkOfArgs(String rcFile, List<String> args) {
    this.rcFile = rcFile;
    this.args = args;
  }

  private final String rcFile;
  private final List<String> args;

  @Override
  public boolean equals(Object o) {
    if (o instanceof RcChunkOfArgs) {
      RcChunkOfArgs other = (RcChunkOfArgs) o;
      return getRcFile().equals(other.getRcFile()) && getArgs().equals(other.getArgs());
    }
    return false;
  }

  @Override
  public int hashCode() {
    return getRcFile().hashCode() + getArgs().hashCode();
  }

  /** The name of the rc file, usually a path. */
  String getRcFile() {
    return rcFile;
  }

  /**
   * The list of arguments specified in this rc "chunk". This is all for a single command (or
   * command:config definition), as different commands will be grouped together, so this list of
   * arguments can all be parsed as a continuous group.
   */
  List<String> getArgs() {
    return args;
  }
}
