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
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.packages.AttributeContainer;
import com.google.devtools.build.lib.packages.CachingPackageLocator;
import com.google.devtools.build.lib.packages.ConstantRuleVisibility;
import com.google.devtools.build.lib.packages.GlobCache;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageFactory.LegacyGlobber;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.StarlarkSemanticsOptions;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Environment.Extension;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.common.options.OptionsParser;
import java.io.IOException;

/**
 * An apparatus that creates / maintains a {@link PackageFactory}.
 */
public class PackageFactoryApparatus {

  private final ExtendedEventHandler eventHandler;
  private final PackageFactory factory;

  public PackageFactoryApparatus(
      ExtendedEventHandler eventHandler,
      PackageFactory.EnvironmentExtension... environmentExtensions) {
    this.eventHandler = eventHandler;
    RuleClassProvider ruleClassProvider = TestRuleClassProvider.getRuleClassProvider();
    factory =
        new PackageFactory(
            ruleClassProvider,
            AttributeContainer::new,
            ImmutableList.copyOf(environmentExtensions),
            "test",
            Package.Builder.DefaultHelper.INSTANCE);
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

  /** Parses and evaluates {@code buildFile} and returns the resulting {@link Package} instance. */
  public Package createPackage(String packageName, RootedPath buildFile) throws Exception {
    return createPackage(PackageIdentifier.createInMainRepo(packageName), buildFile, eventHandler);
  }

  public Package createPackage(String packageName, RootedPath buildFile, String skylarkOption)
      throws Exception {
    return createPackage(
        PackageIdentifier.createInMainRepo(packageName), buildFile, eventHandler, skylarkOption);
  }

  /**
   * Parses and evaluates {@code buildFile} with custom {@code eventHandler} and returns the
   * resulting {@link Package} instance.
   */
  public Package createPackage(
      PackageIdentifier packageIdentifier,
      RootedPath buildFile,
      ExtendedEventHandler reporter,
      String skylarkOption)
      throws Exception {

    OptionsParser parser = OptionsParser.newOptionsParser(StarlarkSemanticsOptions.class);
    parser.parse(
        skylarkOption == null
            ? ImmutableList.<String>of()
            : ImmutableList.<String>of(skylarkOption));
    StarlarkSemantics semantics =
        parser.getOptions(StarlarkSemanticsOptions.class).toSkylarkSemantics();

    try {
      Package externalPkg =
          factory
              .newExternalPackageBuilder(
                  RootedPath.toRootedPath(
                      buildFile.getRoot(),
                      buildFile
                          .getRootRelativePath()
                          .getRelative(LabelConstants.WORKSPACE_FILE_NAME)),
                  "TESTING")
              .build();
      Package pkg =
          factory.createPackageForTesting(
              packageIdentifier, externalPkg, buildFile, getPackageLocator(), reporter, semantics);
      return pkg;
    } catch (InterruptedException e) {
      throw new IllegalStateException(e);
    }
  }

  public Package createPackage(
      PackageIdentifier packageIdentifier, RootedPath buildFile, ExtendedEventHandler reporter)
      throws Exception {
    return createPackage(packageIdentifier, buildFile, reporter, null);
  }

  /**
   * Parses the {@code buildFile} into a {@link BuildFileAST}.
   */
  public BuildFileAST ast(Path buildFile) throws IOException {
    byte[] bytes = FileSystemUtils.readWithKnownFileSize(buildFile, buildFile.getFileSize());
    ParserInputSource inputSource = ParserInputSource.create(bytes, buildFile.asFragment());
    return BuildFileAST.parseBuildFile(inputSource, eventHandler);
  }

  /** Evaluates the {@code buildFileAST} into a {@link Package}. */
  public Pair<Package, GlobCache> evalAndReturnGlobCache(
      String packageName, RootedPath buildFile, BuildFileAST buildFileAST)
      throws InterruptedException, NoSuchPackageException {
    PackageIdentifier packageId = PackageIdentifier.createInMainRepo(packageName);
    GlobCache globCache =
        new GlobCache(
            buildFile.asPath().getParentDirectory(),
            packageId,
            getPackageLocator(),
            null,
            TestUtils.getPool(),
            -1);
    LegacyGlobber globber = PackageFactory.createLegacyGlobber(globCache);
    Package externalPkg =
        factory
            .newExternalPackageBuilder(
                RootedPath.toRootedPath(
                    buildFile.getRoot(),
                    buildFile.getRootRelativePath().getParentDirectory().getRelative("WORKSPACE")),
                "TESTING")
            .build();
    Package.Builder resultBuilder =
        factory.evaluateBuildFile(
            externalPkg.getWorkspaceName(),
            packageId,
            buildFileAST,
            buildFile,
            globber,
            ImmutableList.<Event>of(),
            ImmutableList.<Postable>of(),
            ConstantRuleVisibility.PUBLIC,
            StarlarkSemantics.DEFAULT_SEMANTICS,
            ImmutableMap.<String, Extension>of(),
            ImmutableList.<Label>of(),
            /*repositoryMapping=*/ ImmutableMap.of());
    Package result;
    try {
      result = resultBuilder.build();
    } catch (NoSuchPackageException e) {
      // Make sure not to lose events if we fail to construct the package.
      Event.replayEventsOn(eventHandler, resultBuilder.getEvents());
      throw e;
    }
    Event.replayEventsOn(eventHandler, result.getEvents());
    return Pair.of(result, globCache);
  }

  public Package eval(String packageName, RootedPath buildFile, BuildFileAST buildFileAST)
      throws InterruptedException, NoSuchPackageException {
    return evalAndReturnGlobCache(packageName, buildFile, buildFileAST).first;
  }

  /** Evaluates the {@code buildFileAST} into a {@link Package}. */
  public Package eval(String packageName, RootedPath buildFile)
      throws InterruptedException, IOException, NoSuchPackageException {
    return eval(packageName, buildFile, ast(buildFile.asPath()));
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
