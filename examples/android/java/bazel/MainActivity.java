package bazel;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;

/**
 * Main class for the Bazel Android "Hello, World" app.
 */
public class MainActivity extends AppCompatActivity {
  @Override
  public void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    Log.v("Bazel", "Hello, Android");
    Log.v("Bazel", "Lib says: " + Lib.message());
    System.loadLibrary("hello_world");
    Log.v("Bazel", "JNI says: " + Jni.hello());
  }
}
