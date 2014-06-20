// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.blaze;

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsBase;

/**
 * Startup option to enable verifiable builds.
 */
public class VerifiableOption extends OptionsBase {
  public static final VerifiableOption DEFAULTS = Options.getDefaults(VerifiableOption.class);

  @Option(name = "verifiable",
      defaultValue = "false",
      category = "hidden",  // Will mature to "server startup"
      help = "If set, restrict the usage of flags which could cause the generation of "
          + "a Build Manifest to be inaccurate.")
  public boolean verifiable;
}
