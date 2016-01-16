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

import com.android.prefs.AndroidLocation;
import com.android.prefs.AndroidLocation.AndroidLocationException;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.PrivateKey;
import java.security.UnrecoverableEntryException;
import java.security.UnrecoverableKeyException;
import java.security.cert.Certificate;
import java.security.cert.CertificateException;

/**
 * A provider of a dummy key to sign Android application for debugging purpose.
 * <p/>This provider uses a custom keystore to create and store a key with a known password.
 * 
 * @deprecated Use Android-Builder instead
 */
@Deprecated
public class DebugKeyProvider {

    public interface IKeyGenOutput {
        public void out(String message);
        public void err(String message);
    }

    private static final String PASSWORD_STRING = "android";
    private static final char[] PASSWORD_CHAR = PASSWORD_STRING.toCharArray();
    private static final String DEBUG_ALIAS = "AndroidDebugKey";

    // Certificate CN value. This is a hard-coded value for the debug key.
    // Android Market checks against this value in order to refuse applications signed with
    // debug keys.
    private static final String CERTIFICATE_DESC = "CN=Android Debug,O=Android,C=US";

    private KeyStore.PrivateKeyEntry mEntry;

    public static class KeytoolException extends Exception {
        /** default serial uid */
        private static final long serialVersionUID = 1L;
        private String mJavaHome = null;
        private String mCommandLine = null;

        KeytoolException(String message) {
            super(message);
        }

        KeytoolException(String message, String javaHome, String commandLine) {
            super(message);

            mJavaHome = javaHome;
            mCommandLine = commandLine;
        }

        public String getJavaHome() {
            return mJavaHome;
        }

        public String getCommandLine() {
            return mCommandLine;
        }
    }

    /**
     * Creates a provider using a keystore at the given location.
     * <p/>The keystore, and a new random android debug key are created if they do not yet exist.
     * <p/>Password for the store/key is <code>android</code>, and the key alias is
     * <code>AndroidDebugKey</code>.
     * @param osKeyStorePath the OS path to the keystore, or <code>null</code> if the default one
     * is to be used.
     * @param storeType an optional keystore type, or <code>null</code> if the default is to
     * be used.
     * @param output an optional {@link IKeyGenOutput} object to get the stdout and stderr
     * of the keytool process call.
     * @throws KeytoolException If the creation of the debug key failed.
     * @throws AndroidLocationException
     */
    public DebugKeyProvider(String osKeyStorePath, String storeType, IKeyGenOutput output)
            throws KeyStoreException, NoSuchAlgorithmException, CertificateException,
            UnrecoverableEntryException, IOException, KeytoolException, AndroidLocationException {

        if (osKeyStorePath == null) {
            osKeyStorePath = getDefaultKeyStoreOsPath();
        }

        if (loadKeyEntry(osKeyStorePath, storeType) == false) {
            // create the store with the key
            createNewStore(osKeyStorePath, storeType, output);
        }
    }

    /**
     * Returns the OS path to the default debug keystore.
     *
     * @return The OS path to the default debug keystore.
     * @throws KeytoolException
     * @throws AndroidLocationException
     */
    public static String getDefaultKeyStoreOsPath()
            throws KeytoolException, AndroidLocationException {
        String folder = AndroidLocation.getFolder();
        if (folder == null) {
            throw new KeytoolException("Failed to get HOME directory!\n");
        }
        String osKeyStorePath = folder + "debug.keystore";

        return osKeyStorePath;
    }

    /**
     * Returns the debug {@link PrivateKey} to use to sign applications for debug purpose.
     * @return the private key or <code>null</code> if its creation failed.
     */
    @SuppressWarnings("unused") // the thrown Exceptions are not actually thrown
    public PrivateKey getDebugKey() throws KeyStoreException, NoSuchAlgorithmException,
            UnrecoverableKeyException, UnrecoverableEntryException {
        if (mEntry != null) {
            return mEntry.getPrivateKey();
        }

        return null;
    }

    /**
     * Returns the debug {@link Certificate} to use to sign applications for debug purpose.
     * @return the certificate or <code>null</code> if its creation failed.
     */
    @SuppressWarnings("unused") // the thrown Exceptions are not actually thrown
    public Certificate getCertificate() throws KeyStoreException, NoSuchAlgorithmException,
            UnrecoverableKeyException, UnrecoverableEntryException {
        if (mEntry != null) {
            return mEntry.getCertificate();
        }

        return null;
    }

    /**
     * Loads the debug key from the keystore.
     * @param osKeyStorePath the OS path to the keystore.
     * @param storeType an optional keystore type, or <code>null</code> if the default is to
     * be used.
     * @return <code>true</code> if success, <code>false</code> if the keystore does not exist.
     */
    private boolean loadKeyEntry(String osKeyStorePath, String storeType) throws KeyStoreException,
            NoSuchAlgorithmException, CertificateException, IOException,
            UnrecoverableEntryException {
        FileInputStream fis = null;
        try {
            KeyStore keyStore = KeyStore.getInstance(
                    storeType != null ? storeType : KeyStore.getDefaultType());
            fis = new FileInputStream(osKeyStorePath);
            keyStore.load(fis, PASSWORD_CHAR);
            mEntry = (KeyStore.PrivateKeyEntry)keyStore.getEntry(
                    DEBUG_ALIAS, new KeyStore.PasswordProtection(PASSWORD_CHAR));
        } catch (FileNotFoundException e) {
            return false;
        } finally {
            if (fis != null) {
                try {
                    fis.close();
                } catch (IOException e) {
                    // pass
                }
            }
        }

        return true;
    }

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
    private void createNewStore(String osKeyStorePath, String storeType, IKeyGenOutput output)
            throws KeyStoreException, NoSuchAlgorithmException, CertificateException,
            UnrecoverableEntryException, IOException, KeytoolException {

        if (KeystoreHelper.createNewStore(osKeyStorePath, storeType, PASSWORD_STRING, DEBUG_ALIAS,
                PASSWORD_STRING, CERTIFICATE_DESC, 30 /* validity*/, output)) {
            loadKeyEntry(osKeyStorePath, storeType);
        }
    }
}
