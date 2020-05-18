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
import com.google.devtools.build.lib.util.StringUtilities;

/** Info item for the used heap size. */
public final class UsedHeapSizeInfoItem extends InfoItem {
  public UsedHeapSizeInfoItem() {
    super(
        "used-heap-size",
        "The amount of used memory in bytes. Note that this is not a "
            + "good indicator of the actual memory use, as it includes any remaining inaccessible "
            + "memory.",
        false);
  }

  @Override
  public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env) {
    return print(StringUtilities.prettyPrintBytes(InfoItemUtils.getMemoryUsage().getUsed()));
  }
}
