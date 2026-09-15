// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.pkgcache.PackageProvider;
import com.google.devtools.build.lib.skyframe.TestExpansionValue.TestExpansionKey;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.SkyKey;

/** Reports cycles occurring in during the expansion of <code>test_suite</code> rules. */
class TestExpansionCycleReporter extends AbstractLabelCycleReporter {

  public TestExpansionCycleReporter(PackageProvider packageProvider) {
    super(packageProvider);
  }

  @Override
  protected boolean canReportCycle(SkyKey topLevelKey, CycleInfo cycleInfo) {
    return cycleInfo.getCycle().stream().allMatch(TestExpansionKey.class::isInstance);
  }

  @Override
  protected boolean shouldSkipOnPathToCycle(SkyKey key) {
    return !(key instanceof TestExpansionKey);
  }

  @Override
  protected Label getLabel(SkyKey key) {
    return ((TestExpansionKey) key).getLabel();
  }
}
