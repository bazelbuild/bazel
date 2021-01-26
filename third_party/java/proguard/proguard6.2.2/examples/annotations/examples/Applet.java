import proguard.annotation.*;

/**
 * This applet illustrates the use of annotations for configuring ProGuard.
 *
 * You can compile it with:
 *     javac -classpath ../lib/annotations.jar Applet.java
 * You can then process it with:
 *     java -jar ../../../lib/proguard.jar @ ../examples.pro
 *
 * The annotation will preserve the class and its essential methods,
 * as a result of the specifications in lib/annotations.pro.
 */
@Keep
public class Applet extends java.applet.Applet
{
    // Implementations for Applet.

    public void init()
    {
        // ...
    }
}
