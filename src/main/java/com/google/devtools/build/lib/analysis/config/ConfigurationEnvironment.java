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
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.LoadedPackageProvider;
import com.google.devtools.build.lib.pkgcache.PackageProvider;
import com.google.devtools.build.lib.vfs.Path;

/**
 * An environment to support creating BuildConfiguration instances in a hermetic fashion; all
 * accesses to packages or the file system <b>must</b> go through this interface, so that they can
 * be recorded for correct caching.
 */
public interface ConfigurationEnvironment {
  /**
   * Returns a target for the given label, loading it if necessary, and throwing an exception if it
   * does not exist.
   *
   * @deprecated Do not use this method. Configuration fragments should be fairly dumb key-value
   * maps so that they are cheap and easy to create.
   *
   * <p>If you feel the need to use contents of BUILD files in your
   * {@link ConfigurationFragmentFactory}, add an implicit dependency to your rules that use your
   * configuration fragment that point to a rule of a new rule class, and do the computation during
   * the analysis of said rule. The only uses of this method are those we haven't gotten around
   * migrating yet.
   */
  @Deprecated
  Target getTarget(Label label)
      throws NoSuchPackageException, NoSuchTargetException, InterruptedException;

  /**
   * Returns a path for the given file within the given package.
   *
   * @deprecated Do not use this method. Configuration fragments should be fairly dumb key-value
   * maps so that they are cheap and easy to create. If you feel the need to read contents of files
   * in your {@link ConfigurationFragmentFactory}, you have the following options:
   * <ul>
   *   <li>
   *     Add an implicit dependency to the rules that need this configuration fragment, put the
   *     information you need in BUILD files and use it during the analysis of the implicit
   *     dependency
   *   </li>
   *   <li>
   *     Read the file during the execution phase (then it won't be able to affect analysis)
   *   </li>
   *   <li>
   *     Contact the developers of Bazel and we'll figure something out.
   *   </li>
   * </ul>
   */
  @Deprecated
  Path getPath(Package pkg, String fileName) throws InterruptedException;

  /** Returns fragment based on fragment class and build options. */
  <T extends Fragment> T getFragment(BuildOptions buildOptions, Class<T> fragmentType)
      throws InvalidConfigurationException, InterruptedException;
  /**
   * An implementation backed by a {@link PackageProvider} instance.
   */
  public static final class TargetProviderEnvironment implements ConfigurationEnvironment {
    private final LoadedPackageProvider packageProvider;

    public TargetProviderEnvironment(
        PackageProvider packageProvider, ExtendedEventHandler eventHandler) {
      this.packageProvider = new LoadedPackageProvider(packageProvider, eventHandler);
    }

    @Override
    public Target getTarget(final Label label)
        throws NoSuchPackageException, NoSuchTargetException, InterruptedException {
      return packageProvider.getLoadedTarget(label);
    }

    @Override
    public Path getPath(Package pkg, String fileName) {
      return pkg.getPackageDirectory().getRelative(fileName);
    }

    @Override
    public <T extends Fragment> T getFragment(BuildOptions buildOptions, Class<T> fragmentType) {
      throw new UnsupportedOperationException();
    }
  }
}
