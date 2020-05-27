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

/** Info item for the name and version of the Java runtime environment. */
public final class JavaRuntimeInfoItem extends InfoItem {
  public JavaRuntimeInfoItem() {
    super("java-runtime", "Name and version of the current Java runtime environment.", false);
  }

  @Override
  public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env) {
    return print(
        String.format(
            "%s (build %s) by %s",
            System.getProperty("java.runtime.name", "Unknown runtime"),
            System.getProperty("java.runtime.version", "unknown"),
            System.getProperty("java.vendor", "unknown")));
  }
}
