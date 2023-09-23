package com.google.devtools.build.lib.analysis.actions;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.stream.Collectors;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;

/** Tests for {@link StrippingPathMapper}. */
@RunWith(JUnit4.class)
public class StrippingPathMapperTest extends BuildViewTestCase {

  @Before
  public void setUp() throws Exception {
    useConfiguration("--experimental_output_paths=strip");
  }

  @Test
  public void javaLibraryWithJavacopts() throws Exception {
    scratch.file(
        "java/com/google/test/BUILD",
        "genrule(",
        "    name = 'gen_b',",
        "    outs = ['B.java'],",
        "    cmd = '<some command>',",
        ")",
        "genrule(",
        "    name = 'gen_c',",
        "    outs = ['C.java'],",
        "    cmd = '<some command>',",
        ")",
        "java_library(",
        "    name = 'a',",
        "    javacopts = [",
        "        '-XepOpt:foo:bar=$(location B.java)',",
        "        '-XepOpt:baz=$(location C.java),$(location B.java)',",
        "    ],",
        "    srcs = [",
        "        'A.java',",
        "        'B.java',",
        "        'C.java',",
        "    ],",
        ")");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//java/com/google/test:a");
    Artifact compiledArtifact =
        JavaInfo.getProvider(JavaCompilationArgsProvider.class, configuredTarget)
            .getDirectCompileTimeJars()
            .toList()
            .get(0);
    SpawnAction action = (SpawnAction) getGeneratingAction(compiledArtifact);
    Spawn spawn = action.getSpawn(new ActionExecutionContextBuilder().build());

    assertThat(spawn.getPathMapper().isNoop()).isFalse();
    assertThat(
            spawn.getArguments().stream()
                .filter(arg -> arg.contains("java/com/google/test/"))
                .collect(toImmutableList()))
        .containsExactly(
            "java/com/google/test/A.java",
            "bazel-out/bin/java/com/google/test/B.java",
            "bazel-out/bin/java/com/google/test/C.java",
            "bazel-out/bin/java/com/google/test/liba-hjar.jar",
            "bazel-out/bin/java/com/google/test/liba-hjar.jdeps",
            "-XepOpt:foo:bar=bazel-out/bin/java/com/google/test/B.java",
            "-XepOpt:baz=bazel-out/bin/java/com/google/test/C.java,bazel-out/bin/java/com/google/test/B.java");
  }
}
