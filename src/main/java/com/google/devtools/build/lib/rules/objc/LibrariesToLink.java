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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FORCE_LOAD_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LIBRARY;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;

/**
 * Determines libraries that should be linked into an objc library.
 */
public class LibrariesToLink {

  private Iterable<Artifact> librariesToLink;
  private Iterable<Artifact> importedLibrariesToLink;
  private Iterable<Artifact> forceLoadLibrariesToLink;

  private LibrariesToLink(
      Iterable<Artifact> librariesToLink,
      Iterable<Artifact> importedLibrariesToLink,
      Iterable<Artifact> forceLoadLibrariesToLink) {
    this.librariesToLink = librariesToLink;
    this.importedLibrariesToLink = importedLibrariesToLink;
    this.forceLoadLibrariesToLink = forceLoadLibrariesToLink;
  }

  /**
   * Returns libraries not already linked in the test binary.
   */
  public Iterable<Artifact> getLibrariesToLink() {
    return librariesToLink;
  }

  /**
   * Returns imported libraries not already linked in the test binary.
   */
  public Iterable<Artifact> getImportedLibrariesToLink() {
    return importedLibrariesToLink;
  }

  /**
   * Returns libraries not already linked into the test binary with the --force_load option.
   */
  public Iterable<Artifact> getForceLoadLibrariesToLink() {
    return forceLoadLibrariesToLink;
  }

  /**
   * Creates an instance of LibrariesToLink for an XcTest application by removing libraries
   * already linked into the test binary.  Should only be used if the xctest_app attribute
   * is set.
   *
   * @param ruleContext the RuleContext of the test rule
   * @param objcProvider the ObjcProvider of the test rule
   */
  public static LibrariesToLink xcTestLibraries(
      RuleContext ruleContext, ObjcProvider objcProvider) {

    Iterable<Artifact> librariesToLink = objcProvider.get(LIBRARY);
    Iterable<Artifact> importedLibrariesToLink = objcProvider.get(IMPORTED_LIBRARY);
    Iterable<Artifact> forceLoadLibrariesToLink = objcProvider.get(FORCE_LOAD_LIBRARY);

    XcTestAppProvider xcTestAppProvider =
        ruleContext.getPrerequisite(IosTest.XCTEST_APP, Mode.TARGET, XcTestAppProvider.class);
    librariesToLink =
        Sets.difference(
            objcProvider.get(LIBRARY).toSet(),
            ImmutableSet.copyOf(xcTestAppProvider.getLinkedLibraries()));
    importedLibrariesToLink =
        Sets.difference(
            objcProvider.get(IMPORTED_LIBRARY).toSet(),
            ImmutableSet.copyOf(xcTestAppProvider.getLinkedImportedLibraries()));
    forceLoadLibrariesToLink =
        Sets.difference(
            objcProvider.get(FORCE_LOAD_LIBRARY).toSet(),
            ImmutableSet.copyOf(xcTestAppProvider.getForceLoadLibraries()));

    return new LibrariesToLink(librariesToLink, importedLibrariesToLink, forceLoadLibrariesToLink);
  }

  /**
   * Creates an instance of LibrariesToLink without subtractions.  For use when not linking an
   * xctest app.
   *
   * @param objcProvider the provider for the rule
   */
  public static LibrariesToLink defaultLibraries(ObjcProvider objcProvider) {
    return new LibrariesToLink(
        objcProvider.get(LIBRARY),
        objcProvider.get(IMPORTED_LIBRARY),
        objcProvider.get(FORCE_LOAD_LIBRARY));
  }
}
