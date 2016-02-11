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

package com.google.devtools.build.lib.rules.objc;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Preconditions;

/**
 * Supplies information needed when a dependency serves as an {@code xctest_app}.
 */
@Immutable
public final class XcTestAppProvider implements TransitiveInfoProvider {
  private final Artifact bundleLoader;
  private final Artifact ipa;
  private final ObjcProvider objcProvider;
  private final Iterable<Artifact> linkedLibraries;
  private final Iterable<Artifact> linkedImportedLibraries;
  private final Iterable<Artifact> forceLoadLibraries;

  /**
   * Constructs XcTestAppProvider.
   *
   * @param bundleLoader  the bundle loader to be passed into the linker of the test binary
   * @param ipa  the bundled test application
   * @param objcProvider  an objcProvider to be passed to the depending IosTest target
   * @param linkedLibraries  libraries already linked into the test application, that should not be
   *    linked into the IosTest binary
   * @param linkedImportedLibraries  imported Libraries already linked into the test application,
   *    that should not be linked into the IosTest binary
   * @param forceLoadLibraries  libraries already linked into the test application with --force_load
   *    that should not be linked into the IosTest binary
   */
  XcTestAppProvider(
      Artifact bundleLoader,
      Artifact ipa,
      ObjcProvider objcProvider,
      Iterable<Artifact> linkedLibraries,
      Iterable<Artifact> linkedImportedLibraries,
      Iterable<Artifact> forceLoadLibraries) {
    this.bundleLoader = Preconditions.checkNotNull(bundleLoader);
    this.ipa = Preconditions.checkNotNull(ipa);
    this.objcProvider = Preconditions.checkNotNull(objcProvider);
    this.linkedLibraries = linkedLibraries;
    this.linkedImportedLibraries = linkedImportedLibraries;
    this.forceLoadLibraries = forceLoadLibraries;
  }

  /**
   * The bundle loader, which corresponds to the test app's binary.
   */
  public Artifact getBundleLoader() {
    return bundleLoader;
  }

  public Artifact getIpa() {
    return ipa;
  }

  /**
   * An {@link ObjcProvider} that should be included by any test target that uses this app as its
   * {@code xctest_app}. This is <strong>not</strong> a typical {@link ObjcProvider} - it has
   * certain linker-releated keys omitted, such as {@link ObjcProvider#LIBRARY}, since XcTests have
   * access to symbols in their test rig without linking them into the main test binary.
   */
  public ObjcProvider getObjcProvider() {
    return objcProvider;
  }
  
  /**
   * Returns the list of libraries that were linked into the host application.  These libraries
   * should not also be linked into the test binary, so as to prevent ambiguous references.
   */
  public Iterable<Artifact> getLinkedLibraries() {
    return linkedLibraries;
  }

  /**
   * Returns the list of imported libraries that were linked into the host application.  These
   * libraries  should not also be linked into the test binary, so as to
   * prevent ambiguous references.
   */
  public Iterable<Artifact> getLinkedImportedLibraries() {
    return linkedImportedLibraries;
  }

  /**
   * Returns the list of libraries that were linked into the host application with the --force_load
   * flag.
   */
  public Iterable<Artifact> getForceLoadLibraries() {
    return forceLoadLibraries;
  }
}
