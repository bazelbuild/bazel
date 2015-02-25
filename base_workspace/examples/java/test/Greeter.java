package test;

import java.io.InputStream;
import java.io.PrintStream;
import java.util.Scanner;

class Greeter {
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
      greeting = convertStreamToString(stream);
    }
    out.println(greeting + " " + obj);
  }

  public static void main(String... args) throws Exception {
    Greeter g = new Greeter();
    String obj = args.length > 0 ? args[0] : "world";
    g.hello(obj);
  }
}
