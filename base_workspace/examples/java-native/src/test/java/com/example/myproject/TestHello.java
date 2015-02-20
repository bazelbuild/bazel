package com.example.myproject;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

/**
 * Tests different numbers of arguments to main().
 *
 * <p>With an empty args array, {@link Greeter} should print "Hello world". If there are one or more
 * args, {@link Greeter} should print "Hello &lt;arg[0]&gt;".</p>
 */
public class TestHello {

  @Test
  public void testNoArgument() throws Exception {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    Greeter.out = new PrintStream(out);
    Greeter.main();
    assertEquals("Hello world\n", new String(out.toByteArray(), StandardCharsets.UTF_8));
  }

  @Test
  public void testWithArgument() throws Exception {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    Greeter.out = new PrintStream(out);
    Greeter.main("toto");
    assertEquals("Hello toto\n", new String(out.toByteArray(), StandardCharsets.UTF_8));
  }

}
