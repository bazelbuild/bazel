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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Package.Builder.PackageSettings;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import java.util.List;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.StarlarkFile;

/** Parses a WORKSPACE file with the given content. */
// TODO(adonovan): delete this junk class.
final class WorkspaceFactoryTestHelper {

  private final Root root;
  private Package.Builder builder;
  private StarlarkSemantics starlarkSemantics;

  private final boolean allowWorkspace;

  WorkspaceFactoryTestHelper(Root root) {
    this(true, root);
  }

  WorkspaceFactoryTestHelper(boolean allowWorkspace, Root root) {
    this.root = root;
    this.allowWorkspace = allowWorkspace;
    this.starlarkSemantics = StarlarkSemantics.DEFAULT;
  }

  void parse(String... args) {
    // parse
    Path workspaceFilePath = root.getRelative("WORKSPACE");
    StarlarkFile file =
        StarlarkFile.parse(ParserInput.fromString(Joiner.on("\n").join(args), "WORKSPACE"));
    if (!file.ok()) {
      fail("parse failed: " + file.errors());
      return;
    }

    // execute
    builder =
        Package.newExternalPackageBuilder(
                PackageSettings.DEFAULTS,
                RootedPath.toRootedPath(root, workspaceFilePath),
                "",
                RepositoryMapping.ALWAYS_FALLBACK,
                StarlarkSemantics.DEFAULT)
            .setLoads(ImmutableList.of());
    WorkspaceFactory factory =
        new WorkspaceFactory(
            builder,
            TestRuleClassProvider.getRuleClassProvider(),
            Mutability.create("test"),
            /* allowOverride= */ true,
            allowWorkspace,
            root.asPath(),
            root.asPath(),
            /* defaultSystemJavabaseDir= */ null,
            starlarkSemantics);
    try {
      factory.execute(
          file,
          /* additionalLoadedModules= */ ImmutableMap.of(),
          WorkspaceFileValue.key(RootedPath.toRootedPath(root, workspaceFilePath)));
    } catch (InterruptedException e) {
      fail("Shouldn't happen: " + e.getMessage());
    }
  }

  Package getPackage() throws InterruptedException, NoSuchPackageException {
    return builder.build();
  }

  String getParserError() {
    List<Event> events = builder.getEvents();
    assertThat(events.size()).isGreaterThan(0);
    return events.get(0).getMessage();
  }
}
