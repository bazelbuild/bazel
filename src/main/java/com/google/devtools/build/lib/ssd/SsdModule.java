// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.ssd;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.cache.DigestUtils;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.common.options.OptionsBase;

/**
 * BlazeModule that applies optimizations to Bazel's internals in order to improve performance when
 * using an SSD.
 */
public final class SsdModule extends BlazeModule {
  @Override
  public Iterable<Class<? extends OptionsBase>> getCommonCommandOptions() {
    return ImmutableList.<Class<? extends OptionsBase>>of(SsdOptions.class);
  }

  @Override
  public void beforeCommand(CommandEnvironment env) {
    SsdOptions options = env.getOptions().getOptions(SsdOptions.class);
    if (options.experimentalMultiThreadedDigest) {
      DigestUtils.setMultiThreadedDigest(options.experimentalMultiThreadedDigest);
    }
  }
}
