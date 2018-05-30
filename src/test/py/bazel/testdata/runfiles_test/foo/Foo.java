// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import com.google.devtools.build.runfiles.Runfiles;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;

/** A mock Java binary only used in tests, to exercise the Java Runfiles library. */
public class Foo {
  public static void main(String[] args) throws IOException, InterruptedException {
    System.out.println("Hello Java Foo!");
    Runfiles r = Runfiles.create();
    System.out.println("rloc=" + r.rlocation("foo_ws/foo/datadep/hello.txt"));

    for (String lang : new String[] {"py", "java", "sh", "cc"}) {
      String path = r.rlocation(childBinaryName(lang));
      if (path == null || path.isEmpty()) {
        throw new IOException("cannot find child binary for " + lang);
      }

      ProcessBuilder pb = new ProcessBuilder(path);
      pb.environment().putAll(r.getEnvVars());
      if (isWindows()) {
        pb.environment().put("SYSTEMROOT", System.getenv("SYSTEMROOT"));
      }
      Process p = pb.start();
      if (!p.waitFor(3, TimeUnit.SECONDS)) {
        throw new IOException("child process for " + lang + " timed out");
      }
      if (p.exitValue() != 0) {
        throw new IOException(
            "child process for " + lang + " failed: " + readStream(p.getErrorStream()));
      }
      System.out.printf(readStream(p.getInputStream()));
    }
  }

  private static boolean isWindows() {
    return File.separatorChar == '\\';
  }

  private static String childBinaryName(String lang) {
    if (isWindows()) {
      return "foo_ws/bar/bar-" + lang + ".exe";
    } else {
      return "foo_ws/bar/bar-" + lang;
    }
  }

  private static String readStream(InputStream stm) throws IOException {
    StringBuilder result = new StringBuilder();
    try (BufferedReader r =
        new BufferedReader(new InputStreamReader(stm, StandardCharsets.UTF_8))) {
      String line = null;
      while ((line = r.readLine()) != null) {
        line = line.trim(); // trim CRLF on Windows, LF on Linux
        result.append(line).append("\n"); // ensure uniform line ending
      }
    }
    return result.toString();
  }
}
