package com.google.devtools.build.lib.rules;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.List;
import java.util.stream.Collectors;

import static com.google.common.truth.Truth.assertThat;

@RunWith(JUnit4.class)
public class DataRunfilesPropagationTest extends BuildViewTestCase {
  @Test
  public void testRunfilesPropagation() throws Exception {
    scratch.file(
        "a/BUILD",
        "cc_binary(",
        "    name = 'libhello-lib.so',",
        "    srcs = ['hello-lib.cc', 'hello-lib.h'],",
        "    linkshared = 1,",
        "    data = ['file1.dat'],",
        ")",
        "",
        "cc_import(",
        "    name = 'hello-lib-import',",
        "    shared_library = ':libhello-lib.so',",
        "    data = ['file2.dat'],",
        ")",
        "",
        "# Create a new cc_library to also include the headers needed for the shared library",
        "cc_library(",
        "    name = 'hello-lib',",
        "    hdrs = ['hello-lib.h'],",
        "    deps = ['hello-lib-import'],",
        "    data = ['file3.dat'],",
        ")",
        "",
        "cc_test(",
        "    name = 'hello-world',",
        "    srcs = ['hello-world.cc'],",
        "    deps = [':hello-lib'],",
        "    data = ['file4.dat'],",
        ")");

    ConfiguredTarget target = getConfiguredTarget("//a:hello-world");
    List<String> runfiles =
        getDataRunfiles(target).getArtifacts().toList().stream()
            .map(Artifact::getFilename)
            .collect(Collectors.toList());
    ImmutableList<String> names = ImmutableList.of("file1.dat", "file2.dat", "file3.dat", "file4.dat");
    assertThat(runfiles).containsAtLeastElementsIn(names);
  }
}
