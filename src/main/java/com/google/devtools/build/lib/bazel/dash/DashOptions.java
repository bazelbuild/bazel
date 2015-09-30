// Copyright 2015 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.bazel.dash;

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;

/**
 * Options for sending build results to a dashboard.
 */
public class DashOptions extends OptionsBase {

  @Option(
      name = "use_dash",
      defaultValue = "false",
      help = "If build/test results should be sent to a remote dashboard."
  )
  public boolean useDash;

  @Option(
      name = "dash_url",
      defaultValue = "",
      help = "The URL of the dashboard server."
  )
  public String url;

  @Option(
      name = "dash_secret",
      defaultValue = "",
      help = "The path to a file containing a secret shared with the dashboard server "
          + "for writing build results."
  )
  public String secret;

}
