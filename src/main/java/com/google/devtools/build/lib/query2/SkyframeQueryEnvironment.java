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
package com.google.devtools.build.lib.query2;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageProvider;
import com.google.devtools.build.lib.pkgcache.TargetPatternEvaluator;
import com.google.devtools.build.lib.pkgcache.TransitivePackageLoader;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.syntax.Label;

import java.util.Set;

/**
 * A BlazeQueryEnvironment for Skyframe builds. Currently, this is used to preload transitive
 * closures of targets quickly.
 */
public final class SkyframeQueryEnvironment extends BlazeQueryEnvironment {

  private final TransitivePackageLoader transitivePackageLoader;
  private static final int MAX_DEPTH_FULL_SCAN_LIMIT = 20;

  public SkyframeQueryEnvironment(
      TransitivePackageLoader transitivePackageLoader, PackageProvider packageProvider,
      TargetPatternEvaluator targetPatternEvaluator, boolean keepGoing, int loadingPhaseThreads,
      EventHandler eventHandler, Set<Setting> settings, Iterable<QueryFunction> functions) {
    super(packageProvider, targetPatternEvaluator, keepGoing, loadingPhaseThreads, eventHandler,
        settings, functions);
    this.transitivePackageLoader = transitivePackageLoader;
  }

  @Override
  protected void preloadTransitiveClosure(Set<Target> targets, int maxDepth) throws QueryException {
    if (maxDepth >= MAX_DEPTH_FULL_SCAN_LIMIT) {
      // Only do the full visitation if "maxDepth" is large enough. Otherwise, the benefits of
      // preloading will be outweighed by the cost of doing more work than necessary.
      try {
        transitivePackageLoader.sync(eventHandler, targets, ImmutableSet.<Label>of(), keepGoing,
            loadingPhaseThreads, -1);
      } catch (InterruptedException e) {
        throw new QueryException("interrupted");
      }
    }
  }
}
