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

package com.google.devtools.build.lib.runtime.commands.info;

import com.google.common.base.Supplier;
import com.google.devtools.build.lib.actions.LocalHostCapacity;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.InfoItem;

/** Info item for auto detected local resources . */
public final class LocalResourcesInfoItem extends InfoItem {
  public LocalResourcesInfoItem() {
    super("local_resources", "Auto detected local resources", false);
  }

  @Override
  public byte[] get(
      Supplier<BuildConfigurationValue> configurationSupplier, CommandEnvironment env) {
    ResourceSet localResources = LocalHostCapacity.getLocalHostCapacity();

    return print(
        String.format(
            "RAM=%dMB, CPU=%.1f",
            (int) localResources.getMemoryMb(), localResources.getCpuUsage()));
  }
}
