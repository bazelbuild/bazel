package com.example.myproject;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.io.PrintStream;
import java.util.Scanner;

/**
 * Prints a greeting which can be customized by building with data and/or passing in command-
 * line arguments.
 */
public class Greeter {
  static PrintStream out = System.out;

  public static String getRunfiles() {
    String path = System.getenv("JAVA_RUNFILES");
    if (path == null) {
      path = System.getenv("TEST_SRCDIR");
    }
    return path;
  }

  public static String convertStreamToString(InputStream is) throws Exception {
    Scanner s = new Scanner(is).useDelimiter("\n");
    return s.hasNext() ? s.next() : "";
  }

  public void hello(String obj) throws Exception {
    String greeting = "Hello";
    try {
      String greetFile = getRunfiles()
          + "/io_bazel/examples/java-skylark/src/main/resources/greeting.txt";
      greeting = convertStreamToString(new FileInputStream(greetFile));
    } catch (FileNotFoundException e) {
      // use default.
    }
    out.println(greeting + " " + obj);
  }

  public static void main(String... args) throws Exception {
    Greeter g = new Greeter();
    String obj = args.length > 0 ? args[0] : "world";
    g.hello(obj);
  }
};
