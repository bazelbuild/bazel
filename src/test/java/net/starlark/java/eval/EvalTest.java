// Copyright 2020 The Bazel Authors. All rights reserved.
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

package net.starlark.java.eval;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.Files;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FileOptions;
import com.google.devtools.build.lib.syntax.Module;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.ParserInput;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.SyntaxError;
import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkGlobalLibrary;
import net.starlark.java.annot.StarlarkMethod;

/** Tests of Starlark evaluator. */
@StarlarkGlobalLibrary
public final class EvalTest {

  // Tests for Starlark.
  //
  // In each test file, chunks are separated by "\n---\n".
  // Each chunk is evaluated separately.
  // Use "###" to specify the expected error.
  // If there is no "###", the test will succeed iff there is no error.
  //
  // Within the file, the assert_ and assert_eq functions may be used to
  // report errors without stopping the program. (They are not evaluation
  // errors that can be caught with a '###' expectation.)

  // TODO(adonovan): improve this test driver (following go.starlark.net):
  //
  // - use a proper quotation syntax (Starlark string literals) in '### "foo"' expectations.
  // - extract support for "chunked files" into a library
  //   and reuse it for tests of lexer, parser, resolver.
  // - separate static tests entirely. They can use the same
  //   notation, but we shouldn't be mixing static and dynamic tests.
  // - don't interpret the pattern as "either a substring or a regexp".
  //   Be consistent: always use regexp.
  // - require that some frame of each EvalError match the file/line of the expectation.

  interface Reporter {
    void reportError(StarlarkThread thread, String message);
  }

  @StarlarkMethod(
      name = "assert_",
      documented = false,
      parameters = {
        @Param(name = "cond", noneable = true),
        @Param(name = "msg", defaultValue = "'assertion failed'"),
      },
      useStarlarkThread = true)
  public Object assertStarlark(Object cond, String msg, StarlarkThread thread)
      throws EvalException {
    if (!Starlark.truth(cond)) {
      thread.getThreadLocal(Reporter.class).reportError(thread, "assert_: " + msg);
    }
    return Starlark.NONE;
  }

  @StarlarkMethod(
      name = "assert_eq",
      documented = false,
      parameters = {
        @Param(name = "x", noneable = true),
        @Param(name = "y", noneable = true),
      },
      useStarlarkThread = true)
  public Object assertEq(Object x, Object y, StarlarkThread thread) throws EvalException {
    // TODO(adonovan): use Starlark.equals.
    if (!x.equals(y)) {
      String msg = String.format("assert_eq: %s != %s", Starlark.repr(x), Starlark.repr(y));
      thread.getThreadLocal(Reporter.class).reportError(thread, msg);
    }
    return Starlark.NONE;
  }

  private static boolean ok = true;

  public static void main(String[] args) throws Exception {
    File root = new File("third_party/bazel"); // blaze
    if (!root.exists()) {
      root = new File("."); // bazel
    }
    File testdata = new File(root, "src/test/java/net/starlark/java/eval/testdata");
    for (String name : testdata.list()) {
      File file = new File(testdata, name);
      String content = Files.asCharSource(file, UTF_8).read();
      int linenum = 1;
      for (String chunk : Splitter.on("\n---\n").split(content)) {
        // prepare chunk
        StringBuilder buf = new StringBuilder();
        for (int i = 1; i < linenum; i++) {
          buf.append('\n');
        }
        buf.append(chunk);
        if (false) {
          System.err.printf("%s:%d: <<%s>>\n", file, linenum, buf);
        }

        // extract "### string" expectations
        Map<String, Integer> expectations = new HashMap<>();
        for (int i = 0; true; i += "###".length()) {
          i = chunk.indexOf("###", i);
          if (i < 0) {
            break;
          }
          int j = chunk.indexOf("\n", i);
          if (j < 0) {
            j = chunk.length();
          }
          String pattern = chunk.substring(i + 3, j).trim();
          int line = linenum + newlines(chunk.substring(0, i));
          if (false) {
            System.err.printf("%s:%d: expectation '%s'\n", file, line, pattern);
          }
          expectations.put(pattern, line);
        }

        // parse & execute
        ParserInput input = ParserInput.fromString(buf.toString(), file.toString());
        ImmutableMap.Builder<String, Object> predeclared = ImmutableMap.builder();
        Starlark.addMethods(predeclared, new EvalTest()); // e.g. assert_eq
        StarlarkSemantics semantics = StarlarkSemantics.DEFAULT;
        Module module = Module.withPredeclared(semantics, predeclared.build());
        try (Mutability mu = Mutability.create("test")) {
          StarlarkThread thread = new StarlarkThread(mu, semantics);
          thread.setThreadLocal(Reporter.class, EvalTest::reportError);
          Starlark.execFile(input, FileOptions.DEFAULT, module, thread);

        } catch (SyntaxError.Exception ex) {
          // parser/resolver errors
          for (SyntaxError err : ex.errors()) {
            if (!expected(expectations, err.message())) {
              System.err.println(err); // includes location
              ok = false;
            }
          }

        } catch (EvalException ex) {
          // evaluation error
          //
          // TODO(adonovan): the old logic checks only that each error is matched
          // by at least one expectation. Instead, ensure that errors
          // and expections match exactly. Furthermore, look only at errors
          // whose stack has a frame with a file/line that matches the expectation.
          // This requires inspecting EvalException stack.
          if (!expected(expectations, ex.getMessage())) {
            System.err.println(ex.getMessageWithStack());
            ok = false;
          }

        } catch (Throwable ex) {
          // unhandled exception (incl. InterruptedException)
          System.err.printf(
              "%s:%d: unhandled %s in this chunk: %s\n",
              file, linenum, ex.getClass().getSimpleName(), ex.getMessage());
          ex.printStackTrace();
          ok = false;
        }

        // unmatched expectations
        for (Map.Entry<String, Integer> e : expectations.entrySet()) {
          System.err.printf("%s:%d: unmatched expectation: %s\n", file, e.getValue(), e.getKey());
          ok = false;
        }

        // advance line number
        linenum += newlines(chunk) + 2; // for "\n---\n"
      }
    }
    if (!ok) {
      System.exit(1);
    }
  }

  // Called by assert_ and assert_eq when the test encounters an error.
  // Does not stop the program; multiple failures may be reported in a single run.
  private static void reportError(StarlarkThread thread, String message) {
    System.err.printf("Traceback (most recent call last):\n");
    List<StarlarkThread.CallStackEntry> stack = thread.getCallStack();
    stack = stack.subList(0, stack.size() - 1); // pop the built-in function
    for (StarlarkThread.CallStackEntry fr : stack) {
      System.err.printf("%s: called from %s\n", fr.location, fr.name);
    }
    System.err.println("Error: " + message);
    ok = false;
  }

  private static boolean expected(Map<String, Integer> expectations, String message) {
    for (String pattern : expectations.keySet()) {
      if (message.contains(pattern) || message.matches(".*" + pattern + ".*")) {
        expectations.remove(pattern);
        return true;
      }
    }
    return false;
  }

  private static int newlines(String s) {
    int n = 0;
    for (int i = 0; i < s.length(); i++) {
      if (s.charAt(i) == '\n') {
        n++;
      }
    }
    return n;
  }
}
