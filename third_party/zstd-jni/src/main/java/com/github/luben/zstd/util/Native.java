package com.github.luben.zstd.util;

import com.google.devtools.build.runfiles.Runfiles;
import java.io.IOException;
import java.lang.UnsatisfiedLinkError;


public enum Native {
    ;

    private static final String libnameShort = "zstd-jni";
    private static final String libname = "lib" + libnameShort;

    private static String osName() {
        String os = System.getProperty("os.name").toLowerCase().replace(' ', '_');
        if (os.startsWith("win")){
            return "win";
        } else if (os.startsWith("mac")) {
            return "darwin";
        } else {
            return os;
        }
    }

    private static String libExtension() {
        if (osName().contains("os_x") || osName().contains("darwin")) {
            return "dylib";
         } else if (osName().contains("win")) {
            return "dll";
        } else {
            return "so";
        }
    }

    private static String libraryFilename() {
        return libname + "." + libExtension();
    }

    private static boolean loaded = false;

    public static synchronized boolean isLoaded() {
        return loaded;
    }

    public static synchronized void load() {
        if (loaded) {
            return;
        }
        try {
            System.loadLibrary(libnameShort);
        } catch (UnsatisfiedLinkError e) {
            // Try to load from runfiles, esp. for Windows.
            loadFromRunfiles();
        }
    }

    private static synchronized void loadFromRunfiles() {
        Runfiles runfiles = null;
        try {
            runfiles = Runfiles.create();
        } catch (IOException e) {
            throw new RuntimeException("Unable to locate runfiles after failing to load " +
                                       libnameShort + " from system. Giving up.", e);
        }

        String rloc = runfiles.rlocation("io_bazel/third_party/zstd-jni/" + libraryFilename());
        if (rloc == null) {
            rloc = runfiles.rlocation("bazel_tools/third_party/zstd-jni/" + libraryFilename());
            if (rloc == null) {
                throw new RuntimeException("Unable to find JNI library in runfiles: " + libraryFilename());
            }
        }
        try {
            System.load(rloc);
        } catch (UnsatisfiedLinkError e) {
            throw new RuntimeException("Unable to load JNI library from runfiles: " + rloc, e);
        }
    }
}
