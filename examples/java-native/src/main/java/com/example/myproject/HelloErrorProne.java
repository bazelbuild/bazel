package com.example.myproject;

/** Sanity check for Error Prone integration. */
public class HelloErrorProne {
  public static void main (String[] args) {
    boolean result;
    byte b = 0;
    result = b == 255;
  }
}
