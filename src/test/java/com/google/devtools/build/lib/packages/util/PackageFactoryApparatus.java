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
package com.google.devtools.build.lib.packages.util;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.AttributeContainer;
import com.google.devtools.build.lib.packages.CachingPackageLocator;
import com.google.devtools.build.lib.packages.ConstantRuleVisibility;
import com.google.devtools.build.lib.packages.GlobCache;
import com.google.devtools.build.lib.packages.MakeEnvironment;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Package.LegacyBuilder;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageFactory.LegacyGlobber;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Environment.Extension;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;

import java.io.IOException;

/**
 * An apparatus that creates / maintains a {@link PackageFactory}.
 */
public class PackageFactoryApparatus {

  private final EventHandler eventHandler;
  private final PackageFactory factory;

  public PackageFactoryApparatus(
      EventHandler eventHandler, PackageFactory.EnvironmentExtension... environmentExtensions) {
    this.eventHandler = eventHandler;
    RuleClassProvider ruleClassProvider = TestRuleClassProvider.getRuleClassProvider();
    factory =
        new PackageFactory(
            ruleClassProvider,
            null,
            AttributeContainer.ATTRIBUTE_CONTAINER_FACTORY,
            ImmutableList.copyOf(environmentExtensions),
            "test");
  }

  /**
   * Returns the package factory maintained by this apparatus.
   */
  public PackageFactory factory() {
    return factory;
  }

  private CachingPackageLocator getPackageLocator() {
    // This is used only in globbing and will cause us to always traverse
    // subdirectories.
    return createEmptyLocator();
  }

  /**
   * Parses and evaluates {@code buildFile} and returns the resulting {@link Package} instance.
   */
  public Package createPackage(String packageName, Path buildFile) throws Exception {
    return createPackage(PackageIdentifier.createInMainRepo(packageName), buildFile,
        eventHandler);
  }

  /**
   * Parses and evaluates {@code buildFile} with custom {@code eventHandler} and returns the
   * resulting {@link Package} instance.
   */
  public Package createPackage(PackageIdentifier packageIdentifier, Path buildFile,
      EventHandler reporter) throws Exception {
    try {
      Package pkg =
          factory.createPackageForTesting(
              packageIdentifier,
              buildFile,
              getPackageLocator(),
              reporter);
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
    return BuildFileAST.parseBuildFile(inputSource, eventHandler, /*parsePython=*/ false);
  }

  /**
   * Evaluates the {@code buildFileAST} into a {@link Package}.
   */
  public Pair<Package, GlobCache> evalAndReturnGlobCache(String packageName, Path buildFile,
      BuildFileAST buildFileAST) throws InterruptedException {
    PackageIdentifier packageId = PackageIdentifier.createInMainRepo(packageName);
    GlobCache globCache =
        new GlobCache(
            buildFile.getParentDirectory(),
            packageId,
            getPackageLocator(),
            null,
            TestUtils.getPool());
    LegacyGlobber globber = new LegacyGlobber(globCache);
    Package externalPkg =
        Package.newExternalPackageBuilder(
                buildFile.getParentDirectory().getRelative("WORKSPACE"), "TESTING")
            .build();
    LegacyBuilder resultBuilder =
        factory.evaluateBuildFile(
            externalPkg,
            packageId,
            buildFileAST,
            buildFile,
            globber,
            ImmutableList.<Event>of(),
            ConstantRuleVisibility.PUBLIC,
            false,
            new MakeEnvironment.Builder(),
            ImmutableMap.<String, Extension>of(),
            ImmutableList.<Label>of());
    Package result = resultBuilder.build();
    Event.replayEventsOn(eventHandler, result.getEvents());
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
      public Path getBuildFileForPackage(PackageIdentifier packageName) {
        return null;
      }
    };
  }
}
