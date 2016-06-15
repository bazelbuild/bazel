package com.example.myproject;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

/**
 * Tests using a resource file to replace "Hello" in the output.
 */
public class TestCustomGreeting {

  @Test
  public void testNoArgument() throws Exception {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    Greeter.out = new PrintStream(out);
    Greeter.main();
    assertEquals("Bye world", new String(out.toByteArray(), StandardCharsets.UTF_8).trim());
  }

  @Test
  public void testWithArgument() throws Exception {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    Greeter.out = new PrintStream(out);
    Greeter.main("toto");
    assertEquals("Bye toto", new String(out.toByteArray(), StandardCharsets.UTF_8).trim());
  }

}
