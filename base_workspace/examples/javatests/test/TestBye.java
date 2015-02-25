package test;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

public class TestBye {

  @Test
  public void testNoArgument() throws Exception {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    Greeter.out = new PrintStream(out);
    Greeter.main();
    assertEquals("Bye world\n", new String(out.toByteArray(), StandardCharsets.UTF_8));
  }

  @Test
  public void testWithArgument() throws Exception {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    Greeter.out = new PrintStream(out);
    Greeter.main("toto");
    assertEquals("Bye toto\n", new String(out.toByteArray(), StandardCharsets.UTF_8));
  }

}
