// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.common.options;

import javax.annotation.Nullable;

/**
 * A read-only interface for options parser results, which only allows to query the options of
 * a specific class, but not e.g. the residue any other information pertaining to the command line.
 */
public interface OptionsClassProvider {
  public static final OptionsClassProvider EMPTY = new OptionsClassProvider() {
    @Override @Nullable
    public <O extends OptionsBase> O getOptions(Class<O> optionsClass) {
      return null;
    }
  };

  /**
   * Returns the options instance for the given {@code optionsClass}, that is,
   * the parsed options, or null if it is not among those available.
   *
   * <p>The returned options should be treated by library code as immutable and
   * a provider is permitted to return the same options instance multiple times.
   */
  @Nullable <O extends OptionsBase> O getOptions(Class<O> optionsClass);
}
