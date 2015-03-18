// Copyright 2015 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.analysis.util;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.ConfigurationFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.packages.util.MockToolsConfig;

import java.io.IOException;
import java.util.Collection;

/**
 * Create a mock client for the analysis phase, as well as a configuration factory.
 */
public abstract class AnalysisMock {

  /**
   * This is called from test setup to create the mock directory layout needed to create the
   * configuration.
   */
  public abstract void setupMockClient(MockToolsConfig mockToolsConfig) throws IOException;

  public abstract ConfigurationFactory createConfigurationFactory();

  public abstract Collection<String> getOptionOverrides();

  public abstract ImmutableList<Class<? extends FragmentOptions>> getBuildOptions();

  public static class Delegate extends AnalysisMock {
    private final AnalysisMock delegate;

    public Delegate(AnalysisMock delegate) {
      this.delegate = delegate;
    }

    @Override
    public void setupMockClient(MockToolsConfig mockToolsConfig) throws IOException {
      delegate.setupMockClient(mockToolsConfig);
    }

    @Override
    public ConfigurationFactory createConfigurationFactory() {
      return delegate.createConfigurationFactory();
    }

    @Override
    public Collection<String> getOptionOverrides() {
      return delegate.getOptionOverrides();
    }

    @Override
    public ImmutableList<Class<? extends FragmentOptions>> getBuildOptions() {
      return delegate.getBuildOptions();
    }
  }
}
