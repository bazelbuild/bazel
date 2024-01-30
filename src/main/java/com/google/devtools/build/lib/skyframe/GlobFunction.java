// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.skyframe.SkyFunction;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;

/**
 * A {@link SkyFunction} for {@link GlobValue}s.
 *
 * <p>This code drives the glob matching process. It has two subclasses, {@link
 * GlobFunctionWithMultipleRecursiveFunctions} and {@link
 * GlobFunctionWithRecursionInSingleFunction}.
 *
 * <p>{@link GlobFunctionWithMultipleRecursiveFunctions} is the canonical implementation of {@link
 * GlobFunction} computation. It recursively creates sub-Glob nodes when handling subdirectories
 * under a package. Although evaluating package glob patterns using such a sub-Glob nodes tree is
 * performance friendly for incremental evaluation, it potentially introduced significant memory
 * overhead when the sub-Glob nodes tree becomes extremely large.
 *
 * <p>{@link GlobFunctionWithRecursionInSingleFunction} is introduced due to two major advantages:
 *
 * <ul>
 *   <li>It can mitigate the memory overhead introduced by the giant sub-Glob nodes tree. {@link
 *       com.google.devtools.build.skyframe.SkyFunction.Environment.SkyKeyComputeState} can store
 *       computation state between skyframe restarts and is discarded after evaluating the glob
 *       node. So there is only one Glob node stored in skyframe per glob pattern.
 *   <li>{@code StateMachine} which enables structured concurrency when querying dependent {@code
 *       SkyKey}s. This leads to much less frequency of skyframe restarts when evaluating a glob
 *       pattern.
 * </ul>
 *
 * <p>Currently, {@link GlobFunctionWithRecursionInSingleFunction} does not work well with
 * incremental blaze query. Since {@link
 * com.google.devtools.build.skyframe.SkyFunction.Environment.SkyKeyComputeState} is not stored
 * between blaze invocations, so skyframe incrementality is totally lost compared to {@link
 * GlobFunctionWithMultipleRecursiveFunctions}. Experiments have also shown significant performance
 * regression when using {@link GlobFunctionWithMultipleRecursiveFunctions} to incrementally
 * evaluate glob pattern in a package with directory structure which is too wide and too deep. So we
 * still decide to keep using {@link GlobFunctionWithMultipleRecursiveFunctions} in such a scenario.
 */
public abstract class GlobFunction implements SkyFunction {

  protected ConcurrentHashMap<String, Pattern> regexPatternCache = new ConcurrentHashMap<>();

  void complete() {
    this.regexPatternCache = new ConcurrentHashMap<>();
  }

  /**
   * Creates the {@link GlobFunction} variant based on the type of {@link SkyframeExecutor}.
   *
   * <p>{@link GlobFunctionWithRecursionInSingleFunction} is not fully supported for incremental
   * evaluation due to performance regression. So in the case when the performance requirement for
   * incremental evaluation is strict, creates the canonical {@link
   * GlobFunctionWithMultipleRecursiveFunctions}.
   */
  public static GlobFunction create(boolean recursionInSingleFunction) {
    return recursionInSingleFunction
        ? new GlobFunctionWithRecursionInSingleFunction()
        : new GlobFunctionWithMultipleRecursiveFunctions();
  }
}
