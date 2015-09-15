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
package com.google.devtools.build.lib.analysis.config;

import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.pkgcache.LoadedPackageProvider;

import java.io.IOException;

/**
 * Extended LoadedPackageProvider which is used during a creation of BuildConfiguration.Fragments.
 */
public interface PackageProviderForConfigurations extends LoadedPackageProvider {
  /**
   * Adds dependency to fileName if needed. Used only in skyframe, for creating correct dependencies
   * for {@link com.google.devtools.build.lib.skyframe.ConfigurationCollectionValue}.
   */
  void addDependency(Package pkg, String fileName) throws LabelSyntaxException, IOException;
  
  /**
   * Returns fragment based on fragment type and build options.
   */
  <T extends Fragment> T getFragment(BuildOptions buildOptions, Class<T> fragmentType) 
      throws InvalidConfigurationException;
  
  /**
   * Returns blaze directories and adds dependency to that value.
   */
  BlazeDirectories getDirectories();
  
  /**
   * Returns true if any dependency is missing (value of some node hasn't been evaluated yet).
   */
  boolean valuesMissing();
}
