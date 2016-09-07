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
package com.google.devtools.build.lib.query2;

import com.google.common.base.Predicate;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageProvider;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.pkgcache.TargetPatternEvaluator;
import com.google.devtools.build.lib.pkgcache.TransitivePackageLoader;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.query2.engine.QueryExpressionEvalListener;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.WalkableGraph.WalkableGraphFactory;

import java.util.List;
import java.util.Set;

import javax.annotation.Nullable;

/** A factory that creates instances of {@code AbstractBlazeQueryEnvironment<Target>}. */
public class QueryEnvironmentFactory {
  /** Creates an appropriate {@link AbstractBlazeQueryEnvironment} based on the given options. */
  public AbstractBlazeQueryEnvironment<Target> create(
      TransitivePackageLoader transitivePackageLoader, WalkableGraphFactory graphFactory,
      PackageProvider packageProvider,
      TargetPatternEvaluator targetPatternEvaluator, boolean keepGoing, boolean strictScope,
      boolean orderedResults, List<String> universeScope, int loadingPhaseThreads,
      Predicate<Label> labelFilter,
      EventHandler eventHandler, Set<Setting> settings, Iterable<QueryFunction> functions,
      QueryExpressionEvalListener<Target> evalListener,
      @Nullable PathPackageLocator packagePath) {
    Preconditions.checkNotNull(universeScope);
    if (canUseSkyQuery(orderedResults, universeScope, packagePath, strictScope, labelFilter)) {
      return new SkyQueryEnvironment(
          keepGoing,
          loadingPhaseThreads,
          eventHandler,
          settings,
          functions,
          evalListener,
          targetPatternEvaluator.getOffset(),
          graphFactory,
          universeScope,
          packagePath);
    } else {
      return new BlazeQueryEnvironment(transitivePackageLoader, packageProvider,
          targetPatternEvaluator, keepGoing, strictScope, loadingPhaseThreads, labelFilter,
          eventHandler, settings, functions, evalListener);
    }
  }

  protected static boolean canUseSkyQuery(
      boolean orderedResults,
      List<String> universeScope,
      @Nullable PathPackageLocator packagePath,
      boolean strictScope,
      Predicate<Label> labelFilter) {
    return !orderedResults
        && !universeScope.isEmpty()
        && packagePath != null
        && strictScope
        && labelFilter == Rule.ALL_LABELS;
  }
}

