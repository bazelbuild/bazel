// Copyright 2022 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.OutputGroupInfo.ValidationMode;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment;
import com.google.devtools.build.lib.query2.common.CqueryNode;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryParser;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.junit.Before;
import org.junit.Test;

/** Tests cquery's {@link --output=files} format. */
public final class FilesOutputFormatterCallbackTest extends ConfiguredTargetQueryTest {

  private final CqueryOptions options = new CqueryOptions();
  private final Reporter reporter = new Reporter(new EventBus());

  @Before
  public void defineSimpleRule() throws Exception {
    writeFile(
        "defs/rules.bzl",
        """
        AspectInfo = provider()
        def _r_impl(ctx):
            default_file = ctx.actions.declare_file(ctx.attr.name + '_default_file')
            output_group_only = ctx.actions.declare_file(ctx.attr.name + '_output_group_only')
            runfile = ctx.actions.declare_file(ctx.attr.name + '_runfile')
            executable_only = ctx.actions.declare_file(ctx.attr.name + '_executable')
            files = [default_file, output_group_only, runfile, executable_only]
            ctx.actions.run_shell(
                outputs = files,
                command = '\\n'.join(['touch %s' % file.path for file in files]),
            )
            return [
                DefaultInfo(
                    executable = executable_only,
                    files = depset(
                        direct = [
                            default_file,
                            ctx.file._implicit_source_dep,
                            ctx.file.explicit_source_dep,
                        ],
                        transitive = [info[DefaultInfo].files for info in ctx.attr.deps]
                    ),
                    runfiles = ctx.runfiles([runfile]),
                ),
                OutputGroupInfo(
                    foobar = [output_group_only, ctx.file.explicit_source_dep],
                ),
            ]
        r = rule(
            implementation = _r_impl,
            executable = True,
            attrs = {
                'deps': attr.label_list(),
                '_implicit_source_dep': attr.label(default = 'rules.bzl', allow_single_file = True),
                'explicit_source_dep': attr.label(allow_single_file = True),
            },
        )
        def _a_impl(target, ctx):
            custom_output_group = ctx.actions.declare_file(target.label.name + '_custom_aspect_a_file')
            shared_output_group = ctx.actions.declare_file(target.label.name + '_shared_aspect_a_file')
            ctx.actions.run_shell(
                outputs = [custom_output_group, shared_output_group],
                command = "touch %s && touch %s" % (custom_output_group.path, shared_output_group.path),
            )
            return [
                OutputGroupInfo(
                    aspect_files = depset([custom_output_group]),
                    foobar = depset([shared_output_group]),
                ),
                # Collides with aspect b.
                AspectInfo(),
            ]
        a = aspect(implementation = _a_impl)
        def _b_impl(target, ctx):
            custom_output_group = ctx.actions.declare_file(target.label.name + '_custom_aspect_b_file')
            shared_output_group = ctx.actions.declare_file(target.label.name + '_shared_aspect_b_file')
            ctx.actions.run_shell(
                outputs = [custom_output_group, shared_output_group],
                command = "touch %s && touch %s" % (custom_output_group.path, shared_output_group.path),
            )
            return [
                OutputGroupInfo(
                    aspect_files = depset([custom_output_group]),
                    foobar = depset([shared_output_group]),
                ),
                # Collides with aspect a.
                AspectInfo(),
            ]
        b = aspect(implementation = _b_impl)
        """);
    writeFile("defs/BUILD", "exports_files(['rules.bzl'])");
    writeFile(
        "pkg/BUILD",
        """
        load("//defs:rules.bzl", "r")

        r(
            name = "main",
            explicit_source_dep = "BUILD",
        )

        r(
            name = "other",
            explicit_source_dep = "BUILD",
            deps = [":main"],
        )
        """);
  }

  private List<String> getOutput(
      String queryExpression, List<String> outputGroups, String... aspects) throws Exception {
    QueryExpression expression = QueryParser.parse(queryExpression, getDefaultFunctions());
    Set<String> targetPatternSet = new LinkedHashSet<>();
    expression.collectTargetPatterns(targetPatternSet);
    PostAnalysisQueryEnvironment<CqueryNode> env =
        ((ConfiguredTargetQueryHelper) helper)
            .getPostAnalysisQueryEnvironment(targetPatternSet, Arrays.asList(aspects));

    ByteArrayOutputStream output = new ByteArrayOutputStream();
    FilesOutputFormatterCallback callback =
        new FilesOutputFormatterCallback(
            reporter,
            options,
            new PrintStream(output),
            getHelper().getSkyframeExecutor(),
            env.getAccessor(),
            // Based on BuildRequest#getTopLevelArtifactContext.
            new TopLevelArtifactContext(
                false,
                false,
                OutputGroupInfo.determineOutputGroups(outputGroups, ValidationMode.OFF, false)));
    env.evaluateQuery(expression, callback);
    return Pattern.compile("\n")
        .splitAsStream(output.toString(UTF_8))
        .filter(line -> !line.isEmpty())
        .collect(toImmutableList());
  }

  @Test
  public void basicQuery_defaultOutputGroup() throws Exception {
    List<String> output = getOutput("//pkg:all", ImmutableList.of());
    var sourceAndGeneratedFiles =
        output.stream()
            .collect(Collectors.<String>partitioningBy(path -> path.matches("^[^/]*-out/.*")));
    assertThat(sourceAndGeneratedFiles.get(false)).containsExactly("pkg/BUILD", "defs/rules.bzl");
    assertContainsExactlyWithBinDirPrefix(
        sourceAndGeneratedFiles.get(true), "pkg/main_default_file", "pkg/other_default_file");
  }

  @Test
  public void basicQuery_defaultAndCustomOutputGroup() throws Exception {
    List<String> output = getOutput("//pkg:main", ImmutableList.of("+foobar"));
    var sourceAndGeneratedFiles =
        output.stream()
            .collect(Collectors.<String>partitioningBy(path -> path.matches("^[^/]*-out/.*")));
    assertThat(sourceAndGeneratedFiles.get(false)).containsExactly("pkg/BUILD", "defs/rules.bzl");
    assertContainsExactlyWithBinDirPrefix(
        sourceAndGeneratedFiles.get(true), "pkg/main_default_file", "pkg/main_output_group_only");
  }

  @Test
  public void basicQuery_customOutputGroupOnly() throws Exception {
    List<String> output = getOutput("//pkg:other", ImmutableList.of("foobar"));
    var sourceAndGeneratedFiles =
        output.stream()
            .collect(Collectors.<String>partitioningBy(path -> path.matches("^[^/]*-out/.*")));
    assertThat(sourceAndGeneratedFiles.get(false)).containsExactly("pkg/BUILD");
    assertContainsExactlyWithBinDirPrefix(
        sourceAndGeneratedFiles.get(true), "pkg/other_output_group_only");
  }

  @Test
  public void withAspect_customOutputGroupOnly() throws Exception {
    List<String> output =
        getOutput("//pkg:other", ImmutableList.of("aspect_files"), "//defs:rules.bzl%a");
    var sourceAndGeneratedFiles =
        output.stream()
            .collect(Collectors.<String>partitioningBy(path -> path.matches("^[^/]*-out/.*")));
    assertThat(sourceAndGeneratedFiles.get(false)).isEmpty();
    assertContainsExactlyWithBinDirPrefix(
        sourceAndGeneratedFiles.get(true), "pkg/other_custom_aspect_a_file");
  }

  @Test
  public void withAspect_sharedOutputGroupOnly() throws Exception {
    List<String> output =
        getOutput("//pkg:other", ImmutableList.of("foobar"), "//defs:rules.bzl%a");
    var sourceAndGeneratedFiles =
        output.stream()
            .collect(Collectors.<String>partitioningBy(path -> path.matches("^[^/]*-out/.*")));
    assertThat(sourceAndGeneratedFiles.get(false)).containsExactly("pkg/BUILD");
    assertContainsExactlyWithBinDirPrefix(
        sourceAndGeneratedFiles.get(true),
        "pkg/other_output_group_only",
        "pkg/other_shared_aspect_a_file");
  }

  @Test
  public void withAspects_customOutputGroupOnly() throws Exception {
    List<String> output =
        getOutput(
            "//pkg:other",
            ImmutableList.of("aspect_files"),
            "//defs:rules.bzl%a",
            "//defs:rules.bzl%b");
    var sourceAndGeneratedFiles =
        output.stream()
            .collect(Collectors.<String>partitioningBy(path -> path.matches("^[^/]*-out/.*")));
    assertThat(sourceAndGeneratedFiles.get(false)).isEmpty();
    assertContainsExactlyWithBinDirPrefix(
        sourceAndGeneratedFiles.get(true),
        "pkg/other_custom_aspect_b_file",
        "pkg/other_custom_aspect_a_file");
  }

  @Test
  public void withAspects_sharedOutputGroupOnly() throws Exception {
    List<String> output =
        getOutput(
            "//pkg:other", ImmutableList.of("foobar"), "//defs:rules.bzl%a", "//defs:rules.bzl%b");
    var sourceAndGeneratedFiles =
        output.stream()
            .collect(Collectors.<String>partitioningBy(path -> path.matches("^[^/]*-out/.*")));
    assertThat(sourceAndGeneratedFiles.get(false)).containsExactly("pkg/BUILD");
    assertContainsExactlyWithBinDirPrefix(
        sourceAndGeneratedFiles.get(true),
        "pkg/other_output_group_only",
        "pkg/other_shared_aspect_b_file",
        "pkg/other_shared_aspect_a_file");
  }

  private static void assertContainsExactlyWithBinDirPrefix(
      List<String> output, String... binDirRelativePaths) {
    if (binDirRelativePaths.length == 0) {
      assertThat(output).isEmpty();
      return;
    }

    // Extract the configuration-dependent bin dir from the first output.
    assertThat(output).isNotEmpty();
    String firstPath = output.get(0);
    String binDir = firstPath.substring(0, firstPath.indexOf("bin/") + "bin/".length());

    assertThat(output)
        .containsExactly(
            Arrays.stream(binDirRelativePaths)
                .map(binDirRelativePath -> binDir + binDirRelativePath)
                .toArray());
  }
}
