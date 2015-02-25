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
package com.google.devtools.build.lib.runtime;

import com.google.devtools.common.options.OptionsProvider;

/**
 * An event in which the command line options
 * are discovered.
 */
public class GotOptionsEvent {

  private final OptionsProvider startupOptions;
  private final OptionsProvider options;

  /**
   * Construct the options event.
   *
   * @param startupOptions the parsed startup options
   * @param options the parsed options
   */
  public GotOptionsEvent(OptionsProvider startupOptions, OptionsProvider options) {
    this.startupOptions = startupOptions;
    this.options = options;
  }

  /**
   * @return the parsed startup options
   */
  public OptionsProvider getStartupOptions() {
    return startupOptions;
  }

  /**
   * @return the parsed options.
   */
  public OptionsProvider getOptions() {
    return options;
  }
}
