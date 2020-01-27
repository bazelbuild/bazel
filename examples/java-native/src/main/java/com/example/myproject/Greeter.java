package com.example.myproject;

import java.io.InputStream;
import java.io.PrintStream;
import java.util.Scanner;

/**
 * Prints a greeting which can be customized by building with resources and/or passing in command-
 * line arguments.
 *
 * <p>Building and running this file will print "Hello world". Build and run the
 * //examples/java-native/src/main/java/com/example/myproject:hello-world target to demonstrate
 * this.</p>
 *
 * <p>If this is built with a greeting.txt resource included, it will replace "Hello" with the
 * contents of greeting.txt. The
 * //examples/java-native/src/main/java/com/example/myproject:hello-resources target demonstrates
 * this.</p>
 *
 * <p>If arguments are passed to the binary on the command line, the first argument will replace
 * "world" in the output. See //examples/java-native/src/test/java/com/example/myproject:hello's
 * argument test.</p>
 */
public class Greeter {
  static PrintStream out = System.out;

  public static String convertStreamToString(InputStream is) throws Exception {
    try (Scanner s = new Scanner(is)) {
      s.useDelimiter("\n");
      return s.hasNext() ? s.next() : "";
    }
  }

  public void hello(String obj) throws Exception {
    String greeting = "Hello";
    InputStream stream  = Greeter.class.getResourceAsStream("/greeting.txt");
    if (stream != null) {
      greeting = convertStreamToString(stream).trim();
    }
    out.println(greeting + " " + obj);
  }

  public static void main(String... args) throws Exception {
    Greeter g = new Greeter();
    String obj = args.length > 0 ? args[0] : "world";
    g.hello(obj);
  }
}
