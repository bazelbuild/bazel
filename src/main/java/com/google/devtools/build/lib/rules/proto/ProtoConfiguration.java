// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.proto;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.common.options.Option;

import java.util.List;

/**
 * Configuration for Protocol Buffer Libraries.
 */
public class ProtoConfiguration extends Fragment {

  /**
   * Command line options.
   */
  public static class Options extends FragmentOptions {
    @Option(name = "protocopt",
        allowMultiple = true,
        defaultValue = "",
        category = "flags",
        help = "Additional options to pass to the protobuf compiler.")
    public List<String> protocOpts;

    @Option(
      name = "experimental_proto_extra_actions",
      defaultValue = "false",
      category = "experimental",
      help = "Run extra actions for alternative Java api versions in a proto_library."
    )
    public boolean experimentalProtoExtraActions;
  }

  /**
   * Loader class for proto.
   */
  public static class Loader implements ConfigurationFragmentFactory {
    @Override
    public Fragment create(ConfigurationEnvironment env, BuildOptions buildOptions)
        throws InvalidConfigurationException {
      return new ProtoConfiguration(buildOptions.get(Options.class));
    }

    @Override
    public Class<? extends Fragment> creates() {
      return ProtoConfiguration.class;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
      return ImmutableSet.<Class<? extends FragmentOptions>>of(Options.class);
    }
  }

  private final Options options;

  public ProtoConfiguration(Options options) {
    this.options = options;
  }

  public List<String> protocOpts() {
    return options.protocOpts;
  }


  /**
   * Returns true if we will run extra actions for actions that are not run by default. If this
   * is enabled, e.g. all extra_actions for alternative api-versions or language-flavours of a
   * proto_library target are run.
   */
  public boolean runExperimentalProtoExtraActions() {
    return options.experimentalProtoExtraActions;
  }
}
