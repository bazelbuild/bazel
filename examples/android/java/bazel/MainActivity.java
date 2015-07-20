package bazel;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;

/**
 * Main class for the Bazel Android "Hello, World" app.
 */
public class MainActivity extends Activity {
  @Override
  public void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    Log.v("Bazel", "Hello, Android");
    Log.v("Bazel", "Lib says: " + Lib.message());
    System.loadLibrary("jni");
    Log.v("Bazel", "JNI says: " + Jni.hello());
  }
}
