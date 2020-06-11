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

/** Info item for the committed heap size. */
public final class CommitedHeapSizeInfoItem extends InfoItem {
  public CommitedHeapSizeInfoItem() {
    super(
        "committed-heap-size",
        "The amount of memory in bytes that is committed for the Java virtual machine to use",
        false);
  }

  @Override
  public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env) {
    return print(StringUtilities.prettyPrintBytes(InfoItemUtils.getMemoryUsage().getCommitted()));
  }
}
