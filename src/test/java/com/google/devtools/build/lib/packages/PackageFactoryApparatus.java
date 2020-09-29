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
package com.google.devtools.build.lib.packages;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.Package.Builder.DefaultPackageSettings;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.common.options.OptionsParser;
import java.io.IOException;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.StarlarkFile;

/**
 * A helper class for degenerate tests of the loading phase. Such tests cannot use 'load', for
 * example.
 */
// TODO(adonovan): eliminate this class in due course. Tests should use skyframe.PackageFunction.
class PackageFactoryApparatus {

  private final ExtendedEventHandler eventHandler;
  private final PackageFactory factory;

  PackageFactoryApparatus(ExtendedEventHandler eventHandler) {
    this(eventHandler, PackageValidator.NOOP_VALIDATOR);
  }

  PackageFactoryApparatus(ExtendedEventHandler eventHandler, PackageValidator packageValidator) {
    this.eventHandler = eventHandler;
    RuleClassProvider ruleClassProvider = TestRuleClassProvider.getRuleClassProvider();
    factory =
        new PackageFactory(
            ruleClassProvider,
            PackageFactory.makeDefaultSizedForkJoinPoolForGlobbing(),
            /*environmentExtensions=*/ ImmutableList.of(),
            "test",
            DefaultPackageSettings.INSTANCE,
            packageValidator,
            PackageLoadingListener.NOOP_LISTENER);
  }

  /** Returns the package factory maintained by this apparatus. */
  PackageFactory factory() {
    return factory;
  }

  private CachingPackageLocator getPackageLocator() {
    // This is used only in globbing and will cause us to always traverse
    // subdirectories.
    return createEmptyLocator();
  }

  /** Parses and evaluates {@code buildFile} and returns the resulting {@link Package} instance. */
  Package createPackage(String packageName, RootedPath buildFile) throws Exception {
    return createPackage(PackageIdentifier.createInMainRepo(packageName), buildFile, eventHandler);
  }

  Package createPackage(String packageName, RootedPath buildFile, String starlarkOption)
      throws Exception {
    return createPackage(
        PackageIdentifier.createInMainRepo(packageName), buildFile, eventHandler, starlarkOption);
  }

  /**
   * Parses and evaluates {@code buildFile} with custom {@code eventHandler} and returns the
   * resulting {@link Package} instance.
   */
  Package createPackage(
      PackageIdentifier packageIdentifier,
      RootedPath buildFile,
      ExtendedEventHandler reporter,
      String starlarkOption)
      throws Exception {

    OptionsParser parser =
        OptionsParser.builder().optionsClasses(BuildLanguageOptions.class).build();
    parser.parse(
        starlarkOption == null
            ? ImmutableList.<String>of()
            : ImmutableList.<String>of(starlarkOption));
    StarlarkSemantics semantics =
        parser.getOptions(BuildLanguageOptions.class).toStarlarkSemantics();

    try {
      Package externalPkg =
          factory
              .newExternalPackageBuilder(
                  RootedPath.toRootedPath(
                      buildFile.getRoot(),
                      buildFile
                          .getRootRelativePath()
                          .getRelative(LabelConstants.WORKSPACE_FILE_NAME)),
                  "TESTING",
                  semantics)
              .build();
      Package pkg =
          factory.createPackageForTesting(
              packageIdentifier, externalPkg, buildFile, getPackageLocator(), reporter, semantics);
      return pkg;
    } catch (InterruptedException e) {
      throw new IllegalStateException(e);
    }
  }

  Package createPackage(
      PackageIdentifier packageIdentifier, RootedPath buildFile, ExtendedEventHandler reporter)
      throws Exception {
    return createPackage(packageIdentifier, buildFile, reporter, null);
  }

  /** Parses the {@code buildFile} into a {@link StarlarkFile}. */
  // TODO(adonovan): inline this into all callers. It has nothing to do with PackageFactory.
  StarlarkFile parse(Path buildFile) throws IOException {
    byte[] bytes = FileSystemUtils.readWithKnownFileSize(buildFile, buildFile.getFileSize());
    ParserInput input = ParserInput.fromLatin1(bytes, buildFile.toString());
    StarlarkFile file = StarlarkFile.parse(input);
    Event.replayEventsOn(eventHandler, file.errors());
    return file;
  }

  /** Evaluates the parsed BUILD file {@code file} into a {@link Package}. */
  Pair<Package, GlobCache> evalAndReturnGlobCache(
      String packageName, RootedPath filename, StarlarkFile file)
      throws InterruptedException, NoSuchPackageException {
    PackageIdentifier packageId = PackageIdentifier.createInMainRepo(packageName);
    GlobCache globCache =
        new GlobCache(
            filename.asPath().getParentDirectory(),
            packageId,
            ImmutableSet.of(),
            getPackageLocator(),
            null,
            TestUtils.getPool(),
            -1);
    LegacyGlobber globber = PackageFactory.createLegacyGlobber(globCache);
    Package externalPkg =
        factory
            .newExternalPackageBuilder(
                RootedPath.toRootedPath(
                    filename.getRoot(),
                    filename.getRootRelativePath().getParentDirectory().getRelative("WORKSPACE")),
                "TESTING",
                StarlarkSemantics.DEFAULT)
            .build();
    Package.Builder resultBuilder =
        factory.evaluateBuildFile(
            externalPkg.getWorkspaceName(),
            packageId,
            file,
            filename,
            globber,
            ConstantRuleVisibility.PUBLIC,
            StarlarkSemantics.DEFAULT,
            /*preludeModule=*/ null,
            /*loadedModules=*/ ImmutableMap.of(),
            /*repositoryMapping=*/ ImmutableMap.of());
    Package result;
    try {
      result = resultBuilder.build();
    } finally {
      // Make sure not to lose events if we fail to construct the package.
      Event.replayEventsOn(eventHandler, resultBuilder.getEvents());
    }
    return Pair.of(result, globCache);
  }

  Package eval(String packageName, RootedPath filename, StarlarkFile file)
      throws InterruptedException, NoSuchPackageException {
    return evalAndReturnGlobCache(packageName, filename, file).first;
  }

  /** Evaluates the {@code filename} into a {@link Package}. */
  Package eval(String packageName, RootedPath filename)
      throws InterruptedException, IOException, NoSuchPackageException {
    return eval(packageName, filename, parse(filename.asPath()));
  }

  /** Creates a package locator that finds no packages. */
  static CachingPackageLocator createEmptyLocator() {
    return new CachingPackageLocator() {
      @Override
      public Path getBuildFileForPackage(PackageIdentifier packageName) {
        return null;
      }
    };
  }
}
