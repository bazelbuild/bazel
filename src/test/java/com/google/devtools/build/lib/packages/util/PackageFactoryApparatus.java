// Copyright 2007-2015 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.packages.util;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.packages.CachingPackageLocator;
import com.google.devtools.build.lib.packages.ConstantRuleVisibility;
import com.google.devtools.build.lib.packages.ExternalPackage;
import com.google.devtools.build.lib.packages.GlobCache;
import com.google.devtools.build.lib.packages.MakeEnvironment;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Package.LegacyBuilder;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageFactory.LegacyGlobber;
import com.google.devtools.build.lib.packages.PackageIdentifier;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.syntax.SkylarkEnvironment;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;

/**
 * An apparatus that creates / maintains a {@link PackageFactory}.
 */
public class PackageFactoryApparatus {

  private final EventCollectionApparatus events;
  private final Scratch scratch;
  private final CachingPackageLocator locator;

  private final PackageFactory factory;

  public PackageFactoryApparatus(EventCollectionApparatus events, Scratch scratch,
      PackageFactory.EnvironmentExtension... environmentExtensions) {
    this.events = events;
    this.scratch = scratch;
    RuleClassProvider ruleClassProvider = TestRuleClassProvider.getRuleClassProvider();

    // This is used only in globbing and will cause us to always traverse
    // subdirectories.
    this.locator = createEmptyLocator();

    factory = new PackageFactory(ruleClassProvider, null,
        ImmutableList.copyOf(environmentExtensions));
  }

  /**
   * Returns the package factory maintained by this apparatus.
   */
  public PackageFactory factory() {
    return factory;
  }

  public CachingPackageLocator getPackageLocator() {
    return locator;
  }

  /**
   * Parses and evaluates {@code buildFile} and returns the resulting {@link Package} instance.
   */
  public Package createPackage(String packageName, Path buildFile)
      throws Exception {
    return createPackage(packageName, buildFile, events.reporter());
  }

  /**
   * Parses and evaluates {@code buildFile} with custom {@code reporter} and returns the resulting
   * {@link Package} instance.
   */
  public Package createPackage(String packageName, Path buildFile,
      Reporter reporter)
      throws Exception {
    try {
      Package pkg = factory.createPackageForTesting(
          PackageIdentifier.createInDefaultRepo(packageName), buildFile, locator, reporter);
      return pkg;
    } catch (InterruptedException e) {
      throw new IllegalStateException(e);
    }
  }

  /**
   * Parses the {@code buildFile} into a {@link BuildFileAST}.
   */
  public BuildFileAST ast(Path buildFile) throws IOException {
    ParserInputSource inputSource = ParserInputSource.create(buildFile);
    return BuildFileAST.parseBuildFile(inputSource, events.reporter(), locator,
        /*parsePython=*/false);
  }

  /**
   * Parses the {@code lines} into a {@link BuildFileAST}.
   */
  public BuildFileAST ast(String fileName, String... lines)
      throws IOException {
    Path file = scratch.file(fileName, lines);
    return ast(file);
  }

  /**
   * Evaluates the {@code buildFileAST} into a {@link Package}.
   */
  public Pair<Package, GlobCache> evalAndReturnGlobCache(String packageName, Path buildFile,
      BuildFileAST buildFileAST) throws InterruptedException {
    PackageIdentifier packageId = PackageIdentifier.createInDefaultRepo(packageName);
    GlobCache globCache = new GlobCache(
        buildFile.getParentDirectory(), packageId, locator, null, TestUtils.getPool());
    LegacyGlobber globber = new LegacyGlobber(globCache);
    ExternalPackage externalPkg = (new ExternalPackage.Builder(
        buildFile.getParentDirectory().getRelative("WORKSPACE"))).build();
    LegacyBuilder resultBuilder = factory.evaluateBuildFile(
        externalPkg, packageId, buildFileAST, buildFile,
        globber, ImmutableList.<Event>of(), ConstantRuleVisibility.PUBLIC, false, false,
        new MakeEnvironment.Builder(), ImmutableMap.<PathFragment, SkylarkEnvironment>of(),
        ImmutableList.<Label>of());
    Package result = resultBuilder.build();
    Event.replayEventsOn(events.reporter(), result.getEvents());
    return Pair.of(result, globCache);
  }

  public Package eval(String packageName, Path buildFile, BuildFileAST buildFileAST)
      throws InterruptedException {
    return evalAndReturnGlobCache(packageName, buildFile, buildFileAST).first;
  }

  /**
   * Evaluates the {@code buildFileAST} into a {@link Package}.
   */
  public Package eval(String packageName, Path buildFile)
      throws InterruptedException, IOException {
    return eval(packageName, buildFile, ast(buildFile));
  }

  /**
   * Creates a package locator that finds no packages.
   */
  public static CachingPackageLocator createEmptyLocator() {
    return new CachingPackageLocator() {
      @Override
      public Path getBuildFileForPackage(String packageName) {
        return null;
      }
    };
  }
}
