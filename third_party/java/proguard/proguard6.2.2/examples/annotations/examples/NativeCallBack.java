import proguard.annotation.*;

/**
 * This application illustrates the use of annotations for configuring ProGuard.
 *
 * You can compile it with:
 *     javac -classpath ../lib/annotations.jar NativeCallBack.java
 * You can then process it with:
 *     java -jar ../../../lib/proguard.jar @ ../examples.pro
 *
 * The annotation will preserve the class and its main method,
 * as a result of the specifications in lib/annotations.pro.
 */
@KeepApplication
public class NativeCallBack
{
    /**
     * Suppose this is a native method that computes an answer.
     *
     * The -keep option for native methods in the regular ProGuard
     * configuration will make sure it is not removed or renamed when
     * processing this code.
     */
    public native int computeAnswer();


    /**
     * Suppose this method is called back from the above native method.
     *
     * ProGuard would remove it, because it is not referenced from java.
     * The annotation will make sure it is preserved anyhow.
     */
    @Keep
    public int getAnswer()
    {
        return 42;
    }


    /**
     * The main entry point of the application.
     *
     * The @KeepApplication annotation of this class will make sure it is not
     * removed or renamed when processing this code.
     */
    public static void main(String[] args)
    {
        int answer = new NativeCallBack().computeAnswer();

        System.out.println("The answer is " + answer);
    }
}
