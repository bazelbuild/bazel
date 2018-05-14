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
package com.google.devtools.build.lib.analysis.config;

import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import java.io.IOException;

/**
 * A variant of PackageProvider which is used during a creation of BuildConfiguration.Fragments.
 */
public interface PackageProviderForConfigurations {
  ExtendedEventHandler getEventHandler();

  /**
   * Adds dependency to fileName if needed. Used only in skyframe, for creating correct dependencies
   * for {@link com.google.devtools.build.lib.skyframe.ConfigurationFragmentValue}.
   */
  void addDependency(Package pkg, String fileName)
      throws LabelSyntaxException, IOException, InterruptedException;

  /** Returns fragment based on fragment type and build options. */
  <T extends Fragment> T getFragment(BuildOptions buildOptions, Class<T> fragmentType)
      throws InvalidConfigurationException, InterruptedException;
  
  /**
   * Returns true if any dependency is missing (value of some node hasn't been evaluated yet).
   */
  boolean valuesMissing();

  /**
   * Returns the Target identified by "label", loading, parsing and evaluating the package if it is
   * not already loaded. May return {@code null} if the corresponding Skyframe entry requires
   * function evaluation.
   *
   * @throws NoSuchPackageException if the package could not be found
   * @throws NoSuchTargetException if the package was loaded successfully, but the specified {@link
   *     Target} was not found in it
   */
  Target getTarget(Label label)
      throws NoSuchPackageException, NoSuchTargetException, InterruptedException;
}
