package test;

class Greeter {
  public void hello(String obj) {
    System.out.println("hello " + obj);
  }
  public static void main(String []args) {
    Greeter g = new Greeter();
    String obj = args.length > 1 ? args[1] : "world";
    g.hello(obj);
  }
};
