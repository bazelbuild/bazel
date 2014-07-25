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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.pkgcache.LoadedPackageProvider;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.SkyKey;

/**
 * Reports cycles between {@link TransitiveTargetValue}s. These indicates cycles between targets
 * (e.g. '//a:foo' depends on '//b:bar' and '//b:bar' depends on '//a:foo').
 */
class TransitiveTargetCycleReporter extends AbstractLabelCycleReporter {

  private static final Predicate<SkyKey> IS_TRANSITIVE_TARGET_SKY_KEY =
      SkyFunctions.isSkyFunction(SkyFunctions.TRANSITIVE_TARGET);

  TransitiveTargetCycleReporter(LoadedPackageProvider loadedPackageProvider) {
    super(loadedPackageProvider);
  }

  @Override
  protected boolean canReportCycle(SkyKey topLevelKey, CycleInfo cycleInfo) {
    return Iterables.all(Iterables.concat(ImmutableList.of(topLevelKey),
        cycleInfo.getPathToCycle(), cycleInfo.getCycle()),
        IS_TRANSITIVE_TARGET_SKY_KEY);
  }

  @Override
  public String prettyPrint(SkyKey key) {
    return getLabel(key).toString();
  }

  @Override
  protected Label getLabel(SkyKey key) {
    return (Label) key.argument();
  }
}
