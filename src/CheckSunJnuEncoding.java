public class CheckSunJnuEncoding {
  public static void main(String[] args) {
    String sunJnuEncoding = System.getProperty("sun.jnu.encoding");
    if (!"UTF-8".equals(sunJnuEncoding)) {
      System.err.println("ERROR: sun.jnu.encoding is not UTF-8: " + sunJnuEncoding);
      System.exit(1);
    }
  }
}
