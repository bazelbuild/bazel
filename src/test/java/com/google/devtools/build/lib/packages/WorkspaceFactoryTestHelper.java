// Copyright 2018 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.ParserInput;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.common.options.OptionsParser;
import java.io.IOException;
import java.util.List;

/** Parses a WORKSPACE file with the given content. */
class WorkspaceFactoryTestHelper {
  private final Root root;
  private Package.Builder builder;
  private Exception exception;
  private ImmutableList<Event> events;
  private StarlarkSemantics starlarkSemantics;

  private final boolean allowOverride;

  WorkspaceFactoryTestHelper(Root root) {
    this(true, root);
  }

  WorkspaceFactoryTestHelper(boolean allowOverride, Root root) {
    this.root = root;
    this.exception = null;
    this.events = null;
    this.allowOverride = allowOverride;
    this.starlarkSemantics = StarlarkSemantics.DEFAULT;
  }

  void parse(String... args) {
    Path workspaceFilePath = root.getRelative("WORKSPACE");
    try {
      FileSystemUtils.writeIsoLatin1(workspaceFilePath, args);
    } catch (IOException e) {
      fail("Shouldn't happen: " + e.getMessage());
    }
    StoredEventHandler eventHandler = new StoredEventHandler();
    builder =
        Package.newExternalPackageBuilder(
            Package.Builder.DefaultHelper.INSTANCE,
            RootedPath.toRootedPath(root, workspaceFilePath),
            "",
            StarlarkSemantics.DEFAULT);
    WorkspaceFactory factory =
        new WorkspaceFactory(
            builder,
            TestRuleClassProvider.getRuleClassProvider(),
            ImmutableList.<PackageFactory.EnvironmentExtension>of(),
            Mutability.create("test"),
            allowOverride,
            root.asPath(),
            root.asPath(),
            /* defaultSystemJavabaseDir= */ null,
            starlarkSemantics);
    Exception exception = null;
    try {
      byte[] bytes =
          FileSystemUtils.readWithKnownFileSize(workspaceFilePath, workspaceFilePath.getFileSize());
      factory.parseForTesting(
          ParserInput.fromLatin1(bytes, workspaceFilePath.toString()), eventHandler);
    } catch (BuildFileContainsErrorsException e) {
      exception = e;
    } catch (IOException | InterruptedException e) {
      fail("Shouldn't happen: " + e.getMessage());
    }
    this.events = eventHandler.getEvents();
    this.exception = exception;
  }

  public Package getPackage() throws InterruptedException, NoSuchPackageException {
    return builder.build();
  }

  public void assertLexingExceptionThrown() {
    assertThat(exception).isNotNull();
    assertThat(exception).hasMessageThat().contains("Failed to parse /WORKSPACE");
  }

  public String getLexerError() {
    assertThat(events).hasSize(1);
    return events.get(0).getMessage();
  }

  public String getParserError() {
    List<Event> events = builder.getEvents();
    assertThat(events.size()).isGreaterThan(0);
    return events.get(0).getMessage();
  }

  protected void setStarlarkSemantics(String... options) throws Exception {
    starlarkSemantics = parseStarlarkSemanticsOptions(options);
  }

  private static StarlarkSemantics parseStarlarkSemanticsOptions(String... options)
      throws Exception {
    OptionsParser parser =
        OptionsParser.builder().optionsClasses(StarlarkSemanticsOptions.class).build();
    parser.parse(options);
    return parser.getOptions(StarlarkSemanticsOptions.class).toStarlarkSemantics();
  }

}
