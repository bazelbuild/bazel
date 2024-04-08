package com.google.devtools.build.lib;

import static com.google.common.truth.Truth.assertThat;
import static java.util.stream.Collectors.toMap;

import com.google.devtools.build.docgen.builtin.BuiltinProtos;
import com.google.devtools.build.runfiles.Runfiles;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class BuiltinProtoSmokeTest {
  static BuiltinProtos.Builtins builtins;

  @BeforeClass
  public static void loadProto() throws IOException {
    Path protoPath =
        Path.of(Runfiles.preload().unmapped().rlocation(System.getenv("BUILTIN_PROTO")));
    try (InputStream inputStream = Files.newInputStream(protoPath);
        BufferedInputStream bufferedInputStream = new BufferedInputStream(inputStream)) {
      builtins = BuiltinProtos.Builtins.parseFrom(bufferedInputStream);
    }
  }

  @Test
  public void hasGlobalCallableFromEachApiContext() {
    assertThat(
            builtins.getGlobalList().stream()
                .filter(BuiltinProtos.Value::hasCallable)
                .filter(global -> !global.getCallable().getParamList().isEmpty())
                .collect(toMap(BuiltinProtos.Value::getName, BuiltinProtos.Value::getApiContext)))
        .containsAtLeast(
            "range", BuiltinProtos.ApiContext.ALL,
            "glob", BuiltinProtos.ApiContext.BUILD,
            "DefaultInfo", BuiltinProtos.ApiContext.BZL,
            "bazel_dep", BuiltinProtos.ApiContext.MODULE);
  }
}
