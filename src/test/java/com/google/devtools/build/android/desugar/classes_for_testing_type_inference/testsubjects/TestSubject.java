// Copyright 2017 The Bazel Authors. All rights reserved.
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
package testsubjects;

import java.io.BufferedReader;
import java.io.Closeable;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.sql.Connection;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.List;
import java.util.function.BinaryOperator;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/**
 * Test subject for testing bytecode type inference {@link
 * com.google.devtools.build.android.desugar.BytecodeTypeInference}
 */
public class TestSubject {

  private static int VALUE_ONE = 1;
  private static int VALUE_TWO = 2;

  static int catchTest(Object key, Object value) {
    if (!(key instanceof String)) {
      return VALUE_ONE;
    }
    try {
      Pattern.compile((String) key);
    } catch (PatternSyntaxException e) {
      return VALUE_TWO;
    }
    return VALUE_ONE;
  }

  public static void assertEquals(String message, double expected, double actual, double delta) {
    if (Double.compare(expected, actual) == 0) {
      return;
    }
    if (!(Math.abs(expected - actual) <= delta)) {
      throw new RuntimeException(message + new Double(expected) + new Double(actual));
    }
  }

  /** A simple resource implementation which implements Closeable. */
  public static class SimpleResource implements Closeable {

    public void call(boolean throwException) {
      if (throwException) {
        throw new RuntimeException("exception in call()");
      }
    }

    @Override
    public void close() throws IOException {
      throw new IOException("exception in close().");
    }
  }

  public static void simpleTryWithResources() throws Exception {
    // Throwable.addSuppressed(Throwable) should be called in the following block.
    try (SimpleResource resource = new SimpleResource()) {
      resource.call(true);
    }
  }

  private static long internalCompare(long a, long b, BinaryOperator<Long> func) {
    return func.apply(a, b);
  }

  public void closeResourceArray(Statement[] resources) throws Exception {
    for (Statement stmt : resources) {
      closeResource(stmt, null);
    }
  }

  public void closeResourceMultiArray(Statement[][] resources) throws Exception {
    for (Statement[] stmts : resources) {
      for (Statement stmt : stmts) {
        closeResource(stmt, null);
      }
    }
  }

  public void closeResourceArrayList(List<Statement> resources) throws Exception {
    for (Statement stmt : resources) {
      closeResource(stmt, null);
    }
  }

  public void closeSqlStmt(Connection connection) throws Exception {
    Statement stmt = null;

    try {
      stmt = connection.createStatement();
    } catch (SQLException e) {
      closeResource(stmt, e);
    }
    closeResource(stmt, null);
  }

  public void closeResource(AutoCloseable resource, Throwable suppressor) throws Exception {
    if (resource == null) {
      return;
    }
    try {
      resource.close();
    } catch (Exception e) {
      if (suppressor != null) {
        suppressor.addSuppressed(e);
      }
      throw e;
    }
  }

  public static int intAdd(int i, int j) {
    int tmp = i;
    tmp++;
    ++tmp;
    tmp += j;
    tmp--;
    --tmp;
    tmp -= j;
    tmp *= j;
    tmp /= j;
    tmp = tmp % j;
    tmp = tmp << 2;
    tmp = tmp >> j;
    tmp = tmp >>> 3;
    long longTemp = tmp;
    longTemp = longTemp << j;
    return (int) longTemp;
  }

  public static Number createNumberWithDiamond(boolean flag) {
    Number n = null;
    if (flag) {
      n = new Integer(1);
    } else {
      n = new Double(1);
    }
    return n;
  }

  public static Object[][] createMultiObjectArray() {
    return new Object[0][0];
  }

  public static Object[] createObjectArray() {
    return new Object[0];
  }

  public static int[] createIntArray() {
    return new int[0];
  }

  public static void staticEmpty1() {}

  public void instanceEmpty1() {}

  public static boolean identity(boolean result) {
    return result;
  }

  public static boolean identity2(boolean result) {
    boolean temp = result;
    return temp;
  }

  public void readFile(File file) throws Exception {
    try (AutoCloseable reader = new BufferedReader(new FileReader(file));
        AutoCloseable reader2 = new BufferedReader(new FileReader(file));
        AutoCloseable reader3 = new BufferedReader(new FileReader(file));
        AutoCloseable reader4 = new BufferedReader(new FileReader(file))) {

    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public double testWithDoubleTypes() {
    double result = 1;
    for (double i = 1; i < 22; i = i + 1) {
      System.out.println(i);
      result += i;
    }
    return result;
  }

  public float testWithFloatAndDoubleTypes() {
    float result = 1;
    for (double i = 1; i < 22; i = i + 1) {
      System.out.println(i);
      result += (float) i;
    }
    return result;
  }
}
