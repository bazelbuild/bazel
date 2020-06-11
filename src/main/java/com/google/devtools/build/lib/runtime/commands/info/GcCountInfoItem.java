// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime.commands.info;

import com.google.common.base.Supplier;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.InfoItem;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;

/** Info item for the gc-count. */
public final class GcCountInfoItem extends InfoItem {
  public GcCountInfoItem() {
    super("gc-count", "Number of garbage collection runs.", false);
  }

  @Override
  public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env) {
    // The documentation is not very clear on what it means to have more than
    // one GC MXBean, so we just sum them up.
    long gcCount = 0;
    for (GarbageCollectorMXBean gcBean : ManagementFactory.getGarbageCollectorMXBeans()) {
      gcCount += gcBean.getCollectionCount();
    }
    return print(gcCount + "");
  }
}
