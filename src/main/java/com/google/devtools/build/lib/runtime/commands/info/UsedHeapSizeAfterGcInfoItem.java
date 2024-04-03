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
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.InfoItem;
import com.google.devtools.build.lib.util.HeapOffsetHelper;
import com.google.devtools.build.lib.util.StringUtilities;

/** Info item for the used heap size after garbage collection. */
public final class UsedHeapSizeAfterGcInfoItem extends InfoItem {
  public UsedHeapSizeAfterGcInfoItem() {
    super(
        "used-heap-size-after-gc",
        "The amount of used memory in bytes after a call to System.gc().",
        true);
  }

  @Override
  public byte[] get(
      Supplier<BuildConfigurationValue> configurationSupplier, CommandEnvironment env) {
    System.gc();
    // TODO: b/311665999 - Remove the subtraction of FillerArray once we figure out an alternative.
    return print(
        StringUtilities.prettyPrintBytes(
            InfoItemUtils.getMemoryUsage().getUsed()
                - HeapOffsetHelper.getSizeOfFillerArrayOnHeap()));
  }

}
