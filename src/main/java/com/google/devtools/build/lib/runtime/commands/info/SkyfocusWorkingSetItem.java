// Copyright 2024 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.InfoItem;
import com.google.devtools.build.lib.util.AbruptExitException;
import java.util.TreeSet;

/**
 * The info entry to print out the working set of files used for Skyfocus. See also {@link
 * com.google.devtools.build.lib.skyframe.SkyframeFocuser}.
 */
public class SkyfocusWorkingSetItem extends InfoItem {

  public SkyfocusWorkingSetItem() {
    super("working_set", "Skyfocus working set", false);
  }

  @Override
  public byte[] get(Supplier<BuildConfigurationValue> configurationSupplier, CommandEnvironment env)
      throws AbruptExitException, InterruptedException {

    ImmutableSet<String> workingSet =
        env.getSkyframeExecutor().getSkyfocusState().workingSetStrings();

    if (workingSet.isEmpty()) {
      return print("No working set found.");
    }

    return print(String.join("\n", new TreeSet<>(workingSet)));
  }
}
