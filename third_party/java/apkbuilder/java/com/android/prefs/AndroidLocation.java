/*
 * Copyright (C) 2008 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.android.prefs;

import com.android.annotations.NonNull;

import java.io.File;

/**
 * Manages the location of the android files (including emulator files, ddms config, debug keystore)
 */
public final class AndroidLocation {

    /**
     * The name of the .android folder returned by {@link #getFolder()}.
     */
    public static final String FOLDER_DOT_ANDROID = ".android";

    /**
     * Virtual Device folder inside the path returned by {@link #getFolder()}
     */
    public static final String FOLDER_AVD = "avd";

    /**
     * Throw when the location of the android folder couldn't be found.
     */
    public static final class AndroidLocationException extends Exception {
        private static final long serialVersionUID = 1L;

        public AndroidLocationException(String string) {
            super(string);
        }
    }

    private static String sPrefsLocation = null;

    /**
     * Enum describing which variables to check and whether they should
     * be checked via {@link System#getProperty(String)} or {@link System#getenv()} or both.
     */
    public enum EnvVar {
        ANDROID_SDK_HOME("ANDROID_SDK_HOME", true,  true),  // both sys prop and env var
        USER_HOME       ("user.home",        true,  false), // sys prop only
        HOME            ("HOME",             false, true);  // env var only

        final String mName;
        final boolean mIsSysProp;
        final boolean mIsEnvVar;

        private EnvVar(String name, boolean isSysProp, boolean isEnvVar) {
            mName = name;
            mIsSysProp = isSysProp;
            mIsEnvVar = isEnvVar;
        }

        public String getName() {
            return mName;
        }
    }

    /**
     * Returns the folder used to store android related files.
     * @return an OS specific path, terminated by a separator.
     * @throws AndroidLocationException
     */
    @NonNull
    public static final String getFolder() throws AndroidLocationException {
        if (sPrefsLocation == null) {
            String home = findValidPath(new EnvVar[] { EnvVar.ANDROID_SDK_HOME,
                                                       EnvVar.USER_HOME,
                                                       EnvVar.HOME });

            // if the above failed, we throw an exception.
            if (home == null) {
                throw new AndroidLocationException(
                        "Unable to get the Android SDK home directory.\n" +
                        "Make sure the environment variable ANDROID_SDK_HOME is set up.");
            } else {
                sPrefsLocation = home;
                if (!sPrefsLocation.endsWith(File.separator)) {
                    sPrefsLocation += File.separator;
                }
                sPrefsLocation += FOLDER_DOT_ANDROID + File.separator;
            }
        }

        // make sure the folder exists!
        File f = new File(sPrefsLocation);
        if (f.exists() == false) {
            try {
                f.mkdir();
            } catch (SecurityException e) {
                AndroidLocationException e2 = new AndroidLocationException(String.format(
                        "Unable to create folder '%1$s'. " +
                        "This is the path of preference folder expected by the Android tools.",
                        sPrefsLocation));
                e2.initCause(e);
                throw e2;
            }
        } else if (f.isFile()) {
            throw new AndroidLocationException(sPrefsLocation +
                    " is not a directory! " +
                    "This is the path of preference folder expected by the Android tools.");
        }

        return sPrefsLocation;
    }

    /**
     * Resets the folder used to store android related files. For testing.
     */
    public static final void resetFolder() {
        sPrefsLocation = null;
    }

    /**
     * Checks a list of system properties and/or system environment variables for validity, and
     * existing director, and returns the first one.
     * @param vars The variables to check. Order does matter.
     * @return the content of the first property/variable that is a valid directory.
     */
    private static String findValidPath(EnvVar... vars) {
        for (EnvVar var : vars) {
            String path;
            if (var.mIsSysProp) {
                path = checkPath(System.getProperty(var.mName));
                if (path != null) {
                    return path;
                }
            }

            if (var.mIsEnvVar) {
                path = checkPath(System.getenv(var.mName));
                if (path != null) {
                    return path;
                }
            }
        }

        return null;
    }

    private static String checkPath(String path) {
        if (path != null) {
            File f = new File(path);
            if (f.isDirectory()) {
                return path;
            }
        }
        return null;
    }
}
