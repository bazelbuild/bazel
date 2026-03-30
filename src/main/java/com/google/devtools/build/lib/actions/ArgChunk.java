// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.actions;

/**
 * Post-{@linkplain CommandLine#expand() expansion} representation of command line arguments.
 *
 * <p>This differs from {@link CommandLine} in that consuming the arguments is guaranteed to be free
 * of {@link CommandLineExpansionException} and {@link InterruptedException}.
 */
public interface ArgChunk {

  /**
   * Returns the arguments.
   *
   * <p>The returned {@link Iterable} may lazily materialize strings during iteration, so consumers
   * should attempt to avoid iterating more times than necessary.
   */
  Iterable<String> arguments(PathMapper pathMapper);

  /**
   * Counts the total length of all arguments in this chunk.
   *
   * <p>Implementations that lazily materialize strings may be able to compute the total argument
   * length without actually materializing the arguments.
   */
  int totalArgLength(PathMapper pathMapper);
}
