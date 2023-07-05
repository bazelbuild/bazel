// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote.options;

/** Describes what kind of remote build outputs to download locally. */
public enum RemoteOutputsMode {

  /** Download all remote outputs locally. */
  ALL,

  /**
   * Generally don't download remote action outputs. The only outputs that are being downloaded are:
   * stdout, stderr and .d and .jdeps files for C++ and Java compilation actions.
   */
  MINIMAL,

  /**
   * Downloads outputs of top level targets. Top level targets are targets specified on the command
   * line. If a top level target has runfile dependencies it will also download those. Intermediate
   * outputs are generally not downloaded (See {@link #MINIMAL}.
   */
  TOPLEVEL;
}
