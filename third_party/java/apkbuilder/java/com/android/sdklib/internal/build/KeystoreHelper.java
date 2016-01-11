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

package com.android.sdklib.internal.build;

import com.android.annotations.Nullable;
import com.android.sdklib.internal.build.DebugKeyProvider.IKeyGenOutput;
import com.android.sdklib.internal.build.DebugKeyProvider.KeytoolException;
import com.android.utils.GrabProcessOutput;
import com.android.utils.GrabProcessOutput.IProcessOutput;
import com.android.utils.GrabProcessOutput.Wait;

import java.io.File;
import java.io.IOException;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.UnrecoverableEntryException;
import java.security.cert.CertificateException;
import java.util.ArrayList;

/**
 * A Helper to create new keystore/key.
 *
 * @deprecated Use Android-Builder instead
 */
@Deprecated
public final class KeystoreHelper {

    /**
     * Creates a new store
     * @param osKeyStorePath the location of the store
     * @param storeType an optional keystore type, or <code>null</code> if the default is to
     * be used.
     * @param output an optional {@link IKeyGenOutput} object to get the stdout and stderr
     * of the keytool process call.
     * @throws KeyStoreException
     * @throws NoSuchAlgorithmException
     * @throws CertificateException
     * @throws UnrecoverableEntryException
     * @throws IOException
     * @throws KeytoolException
     */
    public static boolean createNewStore(
            String osKeyStorePath,
            String storeType,
            String storePassword,
            String alias,
            String keyPassword,
            String description,
            int validityYears,
            final IKeyGenOutput output)
            throws KeyStoreException, NoSuchAlgorithmException, CertificateException,
            UnrecoverableEntryException, IOException, KeytoolException {

        // get the executable name of keytool depending on the platform.
        String os = System.getProperty("os.name");

        String keytoolCommand;
        if (os.startsWith("Windows")) {
            keytoolCommand = "keytool.exe";
        } else {
            keytoolCommand = "keytool";
        }

        String javaHome = System.getProperty("java.home");

        if (javaHome != null && javaHome.length() > 0) {
            keytoolCommand = javaHome + File.separator + "bin" + File.separator + keytoolCommand;
        }

        // create the command line to call key tool to build the key with no user input.
        ArrayList<String> commandList = new ArrayList<String>();
        commandList.add(keytoolCommand);
        commandList.add("-genkey");
        commandList.add("-alias");
        commandList.add(alias);
        commandList.add("-keyalg");
        commandList.add("RSA");
        commandList.add("-dname");
        commandList.add(description);
        commandList.add("-validity");
        commandList.add(Integer.toString(validityYears * 365));
        commandList.add("-keypass");
        commandList.add(keyPassword);
        commandList.add("-keystore");
        commandList.add(osKeyStorePath);
        commandList.add("-storepass");
        commandList.add(storePassword);
        if (storeType != null) {
            commandList.add("-storetype");
            commandList.add(storeType);
        }

        String[] commandArray = commandList.toArray(new String[commandList.size()]);

        // launch the command line process
        int result = 0;
        try {
            Process process = Runtime.getRuntime().exec(commandArray);
            result = GrabProcessOutput.grabProcessOutput(
                    process,
                    Wait.WAIT_FOR_READERS,
                    new IProcessOutput() {
                        @Override
                        public void out(@Nullable String line) {
                            if (line != null) {
                                if (output != null) {
                                    output.out(line);
                                } else {
                                    System.out.println(line);
                                }
                            }
                        }

                        @Override
                        public void err(@Nullable String line) {
                            if (line != null) {
                                if (output != null) {
                                    output.err(line);
                                } else {
                                    System.err.println(line);
                                }
                            }
                        }
                    });
        } catch (Exception e) {
            // create the command line as one string for debugging purposes
            StringBuilder builder = new StringBuilder();
            boolean firstArg = true;
            for (String arg : commandArray) {
                boolean hasSpace = arg.indexOf(' ') != -1;

                if (firstArg == true) {
                    firstArg = false;
                } else {
                    builder.append(' ');
                }

                if (hasSpace) {
                    builder.append('"');
                }

                builder.append(arg);

                if (hasSpace) {
                    builder.append('"');
                }
            }

            throw new KeytoolException("Failed to create key: " + e.getMessage(),
                    javaHome, builder.toString());
        }

        if (result != 0) {
            return false;
        }

        return true;
    }
}
