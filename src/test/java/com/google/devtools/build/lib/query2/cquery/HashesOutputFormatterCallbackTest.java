// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.cquery;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.events.EventBusEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment;
import com.google.devtools.build.lib.query2.common.CqueryNode;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryParser;
import com.google.devtools.build.lib.query2.query.hash.DirectTargetHashFunction;
import com.google.devtools.build.lib.query2.query.hash.RepoContentDigestFunction;
import com.google.devtools.build.lib.query2.query.hash.SourceFileDigestFunction;
import com.google.devtools.build.lib.query2.testutil.AbstractQueryTest.QueryHelper;
import com.google.devtools.build.lib.query2.testutil.PostAnalysisQueryHelper.AnalysisHelper;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.common.options.Options;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests cquery's {@code --generate_hashes}. */
@RunWith(JUnit4.class)
public class HashesOutputFormatterCallbackTest extends ConfiguredTargetQueryTest {

  @Override
  protected QueryHelper<CqueryNode> createQueryHelper() {
    return new ConfiguredTargetQueryHelper() {
      @Override
      protected AnalysisHelper createAnalysisHelper() {
        return new AnalysisHelper() {
          @Override
          protected AnalysisMock getAnalysisMock() {
            return new AnalysisMock.Delegate(super.getAnalysisMock()) {
              @Override
              public ImmutableMap<SkyFunctionName, SkyFunction> getSkyFunctions(
                  BlazeDirectories directories) {
                return ImmutableMap.<SkyFunctionName, SkyFunction>builder()
                    .putAll(super.getSkyFunctions(directories))
                    .put(DirectTargetHashFunction.NAME, new DirectTargetHashFunction(directories))
                    .put(SourceFileDigestFunction.NAME, new SourceFileDigestFunction())
                    .put(RepoContentDigestFunction.NAME, new RepoContentDigestFunction(directories))
                    .buildOrThrow();
              }
            };
          }
        };
      }
    };
  }

  /**
   * Writes the workspace used by these tests. Not a {@code @Before} method, because it would also
   * apply to the tests inherited from {@link ConfiguredTargetQueryTest}.
   */
  private void writeWorkspace() throws Exception {
    writeFile(
        "defs/defs.bzl",
        """
        def _tc_impl(ctx):
            return [platform_common.ToolchainInfo(src = ctx.file.src)]

        my_toolchain = rule(_tc_impl, attrs = {"src": attr.label(allow_single_file = True)})

        def _impl(ctx):
            pass

        my_rule = rule(
            implementation = _impl,
            attrs = {"srcs": attr.label_list(allow_files = True), "deps": attr.label_list()},
            toolchains = ["//tc:type"],
        )
        """);
    writeFile("defs/BUILD");
    writeFile(
        "tc/BUILD",
        """
        load("//defs:defs.bzl", "my_toolchain")

        toolchain_type(name = "type")

        constraint_setting(name = "flavor")

        constraint_value(name = "a", constraint_setting = ":flavor")

        constraint_value(name = "b", constraint_setting = ":flavor")

        platform(name = "plat_a", constraint_values = [":a"])

        platform(name = "plat_b", constraint_values = [":b"])

        my_toolchain(name = "impl_a", src = "a.txt")

        my_toolchain(name = "impl_b", src = "b.txt")

        toolchain(
            name = "tc_a",
            target_compatible_with = [":a"],
            toolchain = ":impl_a",
            toolchain_type = ":type",
        )

        toolchain(
            name = "tc_b",
            target_compatible_with = [":b"],
            toolchain = ":impl_b",
            toolchain_type = ":type",
        )
        """);
    writeFile("tc/a.txt", "a");
    writeFile("tc/b.txt", "b");
    overwriteFile("MODULE.bazel", "register_toolchains('//tc:all')");
    writeFile(
        "app/BUILD",
        """
        load("//defs:defs.bzl", "my_rule")

        config_setting(
            name = "x",
            define_values = {"mode": "x"},
        )

        my_rule(
            name = "lib",
            srcs = ["lib.txt"],
        )

        my_rule(
            name = "bin",
            srcs = select({
                ":x": ["x.txt"],
                "//conditions:default": ["y.txt"],
            }),
            deps = [":lib"],
        )
        """);
    writeFile("app/lib.txt", "lib");
    writeFile("app/x.txt", "x");
    writeFile("app/y.txt", "y");
    getHelper().useConfiguration("--platforms=//tc:plat_a");
  }

  /** Returns the hash printed for each target, keyed by label and configuration. */
  private Map<String, String> getHashes(String queryExpression) throws Exception {
    QueryExpression expression = QueryParser.parse(queryExpression, getDefaultFunctions());
    Set<String> targetPatternSet = new LinkedHashSet<>();
    expression.collectTargetPatterns(targetPatternSet);
    PostAnalysisQueryEnvironment<CqueryNode> env =
        ((ConfiguredTargetQueryHelper) helper).getPostAnalysisQueryEnvironment(targetPatternSet);
    ByteArrayOutputStream output = new ByteArrayOutputStream();
    HashesOutputFormatterCallback callback =
        new HashesOutputFormatterCallback(
            new Reporter(EventBusEventHandler.createWithNewEventBus()),
            Options.getDefaults(CqueryOptions.class),
            new PrintStream(output),
            getHelper().getSkyframeExecutor(),
            env.getAccessor(),
            ct -> env.getFwdDeps(ImmutableList.of(ct)),
            LabelPrinter.legacy());
    env.evaluateQuery(expression, callback);
    Map<String, String> hashes = new LinkedHashMap<>();
    for (String line : output.toString().split("\n")) {
      int space = line.lastIndexOf(' ');
      hashes.put(line.substring(0, space), line.substring(space + 1));
    }
    return hashes;
  }

  /** Returns the hash of the only configured target with the given label. */
  private static String hashOf(Map<String, String> hashes, String label) {
    ImmutableList<String> matches =
        hashes.entrySet().stream()
            .filter(e -> e.getKey().startsWith(label + " ("))
            .map(Map.Entry::getValue)
            .collect(ImmutableList.toImmutableList());
    assertThat(matches).hasSize(1);
    return matches.get(0);
  }

  @Test
  public void printsOneDeterministicHashPerConfiguredTarget() throws Exception {
    writeWorkspace();
    Map<String, String> hashes = getHashes("//app:bin + //app:lib");

    assertThat(hashes).hasSize(2);
    assertThat(hashOf(hashes, "//app:bin")).matches("[0-9a-f]{64}");
    assertThat(hashOf(hashes, "//app:bin")).isNotEqualTo(hashOf(hashes, "//app:lib"));
    assertThat(getHashes("//app:bin + //app:lib")).isEqualTo(hashes);
  }

  @Test
  public void selectBranchThatIsNotTaken_doesNotAffectHash() throws Exception {
    writeWorkspace();
    Map<String, String> before = getHashes("//app:bin");

    overwriteFile("app/x.txt", "changed x");

    assertThat(getHashes("//app:bin")).isEqualTo(before);
  }

  @Test
  public void selectBranchThatIsTaken_affectsHash() throws Exception {
    writeWorkspace();
    Map<String, String> before = getHashes("//app:bin + //app:lib");

    overwriteFile("app/y.txt", "changed y");

    Map<String, String> after = getHashes("//app:bin + //app:lib");
    assertThat(hashOf(after, "//app:bin")).isNotEqualTo(hashOf(before, "//app:bin"));
    assertThat(hashOf(after, "//app:lib")).isEqualTo(hashOf(before, "//app:lib"));
  }

  @Test
  public void resolvedToolchain_affectsHash() throws Exception {
    writeWorkspace();
    Map<String, String> before = getHashes("//app:lib");

    overwriteFile("tc/a.txt", "changed a");

    assertThat(hashOf(getHashes("//app:lib"), "//app:lib"))
        .isNotEqualTo(hashOf(before, "//app:lib"));
  }

  @Test
  public void toolchainThatIsNotResolved_doesNotAffectHash() throws Exception {
    writeWorkspace();
    Map<String, String> before = getHashes("//app:lib");

    overwriteFile("tc/b.txt", "changed b");

    assertThat(getHashes("//app:lib")).isEqualTo(before);
  }

  @Test
  public void buildOptions_affectHash() throws Exception {
    writeWorkspace();
    String before = hashOf(getHashes("//app:lib"), "//app:lib");

    getHelper().useConfiguration("--platforms=//tc:plat_a", "--define=mode=x");

    assertThat(hashOf(getHashes("//app:lib"), "//app:lib")).isNotEqualTo(before);
  }
}
