package com.github.luben.zstd;

import com.github.luben.zstd.Zstd;
import com.github.luben.zstd.util.Native;
import android.support.test.InstrumentationRegistry;
import android.support.test.runner.AndroidJUnit4;

import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * Instrumented test, which will execute on an Android device.
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
@RunWith(AndroidJUnit4.class)
public class LoadTest {
    @Test
    public void loading() throws Exception {
        Native.load();
        byte[] in = new byte[0];
        byte[] compressed = Zstd.compress(in);
        byte[] ob = new byte[100];
        assert(Zstd.decompress(ob, compressed) == 0);
    }
}
