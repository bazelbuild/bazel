package test;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.util.Scanner;

class Greeter {
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
      String greetFile = getRunfiles() + "/examples/java_oss/greeting.txt";
      greeting = convertStreamToString(new FileInputStream(greetFile));
    } catch (FileNotFoundException e) {
      // use default.
    }
    System.out.println(greeting + " " + obj);
  }

  public static void main(String []args) throws Exception {
    Greeter g = new Greeter();
    String obj = args.length > 0 ? args[0] : "world";
    g.hello(obj);
  }
};
