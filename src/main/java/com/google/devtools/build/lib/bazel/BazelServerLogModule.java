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

package com.google.devtools.build.lib.bazel;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;

/** Provides startup options for the Bazel server log handler. */
public class BazelServerLogModule extends BlazeModule {
  /** Logging flags. */
  public static final class Options extends OptionsBase {
    // TODO(b/118755753): The --norotating_server_log option is intended as a temporary "escape
    // hatch" in case switching to the rotating ServerLogHandler breaks things. Remove the option
    // and associated logic once we are confident that the "escape hatch" is not needed.
    @Option(
        name = "rotating_server_log",
        defaultValue = "true", // NOTE: purely decorative, rc files are read by the client.
        documentationCategory = OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
        effectTags = {OptionEffectTag.BAZEL_MONITORING},
        help =
            "Create a new log file when the %{product} server process (re)starts, and update the "
                + "java.log symbolic link to point to the new file. Otherwise, java.log would be a "
                + "regular file which would be overwritten each time the server process starts.")
    public boolean rotatingServerLog;
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getStartupOptions() {
    return ImmutableList.of(Options.class);
  }
}
