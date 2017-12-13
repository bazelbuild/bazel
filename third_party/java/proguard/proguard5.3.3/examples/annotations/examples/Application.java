import proguard.annotation.KeepApplication;

/**
 * This application illustrates the use of annotations for configuring ProGuard.
 *
 * You can compile it with:
 *     javac -classpath ../lib/annotations.jar Application.java
 * You can then process it with:
 *     java -jar ../../../lib/proguard.jar @ ../examples.pro
 *
 * The annotation will preserve the class and its main method,
 * as a result of the specifications in lib/annotations.pro.
 */
@KeepApplication
public class Application
{
    public static void main(String[] args)
    {
        System.out.println("The answer is 42");
    }
}
