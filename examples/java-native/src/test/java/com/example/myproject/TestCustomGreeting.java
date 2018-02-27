package com.example.myproject;

import static org.junit.Assert.fail;
import static org.junit.Assert.assertEquals;

import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
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
    byte[] hd = new byte[100];
    int size = Greeter.readGreeter(hd);
    if (size < 1) {
      fail("size < 1");
    }
        StringBuilder sb = new StringBuilder();
        sb.append("DEBUG[Greeter]");
        for (int i = 0; i < size; ++i) {
          sb.append(String.format(" 0x%02x (%c)", hd[i], hd[i] >=32 ? hd[i] : 32));
        }
    fail(sb.toString());
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
