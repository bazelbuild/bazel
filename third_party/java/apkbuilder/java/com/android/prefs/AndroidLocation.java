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
import com.android.annotations.Nullable;
import com.android.utils.FileUtils;

import java.io.File;

/**
 * Manages the location of the android files (including emulator files, ddms config, debug keystore)
 */
public final class AndroidLocation {

    /**
     * The name of the .android folder returned by {@link #getFolder}.
     */
    public static final String FOLDER_DOT_ANDROID = ".android";

    /**
     * Virtual Device folder inside the path returned by {@link #getFolder}
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
    private static String sAvdLocation = null;

    /**
     * Enum describing which variables to check and whether they should
     * be checked via {@link System#getProperty(String)} or {@link System#getenv()} or both.
     */
    public enum EnvVar {
        ANDROID_AVD_HOME("ANDROID_AVD_HOME", true,  true),  // both sys prop and env var
        ANDROID_SDK_HOME("ANDROID_SDK_HOME", true,  true),  // both sys prop and env var
        USER_HOME       ("user.home",        true,  false), // sys prop only
        HOME            ("HOME",             false, true);  // env var only

        final String mName;
        final boolean mIsSysProp;
        final boolean mIsEnvVar;

        EnvVar(String name, boolean isSysProp, boolean isEnvVar) {
            mName = name;
            mIsSysProp = isSysProp;
            mIsEnvVar = isEnvVar;
        }

        public String getName() {
            return mName;
        }

        @Nullable
        public String validatePath(boolean silent) throws AndroidLocationException {
            String path;
            if (mIsSysProp) {
                path = checkPath(System.getProperty(mName), silent);
                if (path != null) {
                    return path;
                }
            }

            if (mIsEnvVar) {
                path = checkPath(System.getenv(mName), silent);
                if (path != null) {
                    return path;
                }
            }
            return null;
        }

        @Nullable
        private String checkPath(@Nullable String path, boolean silent)
          throws AndroidLocationException {
            if (path == null) {
                return null;
            }
            File file = new File(path);
            if (!file.isDirectory()) {
                return null;
            }
            if (!(this == ANDROID_SDK_HOME && isSdkRootWithoutDotAndroid(file))) {
                return path;
            }
            if (!silent) {
                throw new AndroidLocationException(String.format(
                  "ANDROID_SDK_HOME is set to the root of your SDK: %1$s\n" +
                  "This is the path of the preference folder expected by the Android tools.\n" +
                  "It should NOT be set to the same as the root of your SDK.\n" +
                  "Please set it to a different folder or do not set it at all.\n" +
                  "If this is not set we default to: %2$s",
                  path, findValidPath(USER_HOME, HOME)));
            }
            return null;
        }

        private static boolean isSdkRootWithoutDotAndroid(@NonNull File folder) {
            return subFolderExist(folder, "platforms") &&
                   subFolderExist(folder, "platform-tools") &&
                   !subFolderExist(folder, FOLDER_DOT_ANDROID);
        }

        private static boolean subFolderExist(@NonNull File folder, @NonNull String subFolder) {
            return new File(folder, subFolder).isDirectory();
        }
    }

    /**
     * Returns the folder used to store android related files.
     * If the folder is not created yet, it will be created here.
     * @return an OS specific path, terminated by a separator.
     * @throws AndroidLocationException
     */
    public static String getFolder() throws AndroidLocationException {
        if (sPrefsLocation == null) {
            sPrefsLocation = findHomeFolder();
        }

        // make sure the folder exists!
        File f = new File(sPrefsLocation);
        if (!f.exists()) {
            try {
                FileUtils.mkdirs(f);
            } catch (SecurityException e) {
                AndroidLocationException e2 = new AndroidLocationException(String.format(
                  "Unable to create folder '%1$s'. " +
                  "This is the path of preference folder expected by the Android tools.",
                  sPrefsLocation));
                e2.initCause(e);
                throw e2;
            }
        } else if (f.isFile()) {
            throw new AndroidLocationException(String.format(
              "%1$s is not a directory!\n" +
              "This is the path of preference folder expected by the Android tools.", sPrefsLocation));
        }
        return sPrefsLocation;
    }

    /**
     * Returns the folder used to store android related files.
     * This method will not create the folder if it doesn't exist yet.\
     *
     * @return an OS specific path, terminated by a separator or null
     *         if no path is found or an error occurred.
     */
    public static String getFolderWithoutWrites() {
        if (sPrefsLocation == null) {
            try {
                sPrefsLocation = findHomeFolder();
            }
            catch (AndroidLocationException e) {
                return null;
            }
        }
        return sPrefsLocation;
    }


    /**
     * Check the if ANDROID_SDK_HOME variable points to a SDK.
     * If it points to an SDK
     * @throws AndroidLocationException
     */
    public static void checkAndroidSdkHome() throws AndroidLocationException {
        EnvVar.ANDROID_SDK_HOME.validatePath(false);
    }

    /**
     * Returns the folder where the users AVDs are stored.
     * @return an OS specific path, terminated by a separator.
     * @throws AndroidLocationException
     */
    @NonNull
    public static String getAvdFolder() throws AndroidLocationException {
        if (sAvdLocation == null) {
            String home = findValidPath(EnvVar.ANDROID_AVD_HOME);
            if (home == null) {
                home = getFolder() + FOLDER_AVD;
            }
            sAvdLocation = home;
            if (!sAvdLocation.endsWith(File.separator)) {
                sAvdLocation += File.separator;
            }
        }
        return sAvdLocation;
    }

    private static String findHomeFolder()
      throws AndroidLocationException {
        String home = findValidPath(EnvVar.ANDROID_SDK_HOME, EnvVar.USER_HOME, EnvVar.HOME);

        // if the above failed, we throw an exception.
        if (home == null) {
            throw new AndroidLocationException(
              "Unable to get the Android SDK home directory.\n" +
              "Make sure the environment variable ANDROID_SDK_HOME is set up.");
        }
        if (!home.endsWith(File.separator)) {
            home += File.separator;
        }
        return home + FOLDER_DOT_ANDROID + File.separator;
    }

    /**
     * Resets the folder used to store android related files. For testing.
     */
    public static void resetFolder() {
        sPrefsLocation = null;
        sAvdLocation = null;
    }

    /**
     * Checks a list of system properties and/or system environment variables for validity,
     * and returns the first one.
     * @param vars The variables to check. Order does matter.
     * @return the content of the first property/variable that is a valid directory.
     */
    @Nullable
    private static String findValidPath(EnvVar... vars) throws AndroidLocationException {
        for (EnvVar var : vars) {
            String path = var.validatePath(true);
            if (path != null) {
                return path;
            }
        }
        return null;
    }
}
