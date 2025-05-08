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
import java.nio.charset.Charset;

/** Info item for the current character encoding settings. */
public final class CharacterEncodingInfoItem extends InfoItem {
  public CharacterEncodingInfoItem() {
    super(
        "character-encoding",
        "Information about the character encoding used by the running JVM.",
        false);
  }

  @Override
  public byte[] get(
      Supplier<BuildConfigurationValue> configurationSupplier, CommandEnvironment env) {
    return print(
        String.format(
            "file.encoding = %s, defaultCharset = %s, sun.jnu.encoding = %s",
            System.getProperty("file.encoding", "unknown"),
            Charset.defaultCharset().name(),
            System.getProperty("sun.jnu.encoding", "unknown")));
  }
}
