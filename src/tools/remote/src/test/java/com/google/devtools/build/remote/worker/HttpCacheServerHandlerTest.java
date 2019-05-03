package com.google.devtools.build.remote.worker;

import static org.junit.Assert.*;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class HttpCacheServerHandlerTest {

  private HttpCacheServerHandler handler = new HttpCacheServerHandler();

  @Test
  public void isUriValidWhenValidUri() {
    assertTrue(handler.isUriValid("http://some-path.co.uk:8080/ac/1111111111111111111111111111111111111111111111111111111111111111"));
    assertTrue(handler.isUriValid("http://127.12.12.0:8080/ac/1111111111111111111111111111111111111111111111111111111111111111"));
    assertTrue(handler.isUriValid("http://localhost:8080/ac/1111111111111111111111111111111111111111111111111111111111111111"));
    assertTrue(handler.isUriValid("https://localhost:8080/ac/1111111111111111111111111111111111111111111111111111111111111111"));
    assertTrue(handler.isUriValid("localhost:8080/ac/1111111111111111111111111111111111111111111111111111111111111111"));

    assertTrue(handler.isUriValid("http://some-path.co.uk:8080/cas/1111111111111111111111111111111111111111111111111111111111111111"));
    assertTrue(handler.isUriValid("http://127.12.12.0:8080/cas/1111111111111111111111111111111111111111111111111111111111111111"));
    assertTrue(handler.isUriValid("http://localhost:8080/cas/1111111111111111111111111111111111111111111111111111111111111111"));
    assertTrue(handler.isUriValid("https://localhost:8080/cas/1111111111111111111111111111111111111111111111111111111111111111"));
    assertTrue(handler.isUriValid("localhost:8080/cas/1111111111111111111111111111111111111111111111111111111111111111"));
  }

  @Test
  public void isUriValidWhenInvalidUri() {
    assertFalse(handler.isUriValid("http://localhost:8080/ac_1111111111111111111111111111111111111111111111111111111111111111"));
    assertFalse(handler.isUriValid("http://localhost:8080/cas_1111111111111111111111111111111111111111111111111111111111111111"));
    assertFalse(handler.isUriValid("http://localhost:8080/ac/111111111111111111111"));
    assertFalse(handler.isUriValid("http://localhost:8080/cas/111111111111111111111"));
    assertFalse(handler.isUriValid("http://localhost:8080/cas/823rhf&*%OL%_^"));
    assertFalse(handler.isUriValid("http://localhost:8080/ac/823rhf&*%OL%_^"));
  }
}