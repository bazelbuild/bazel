/*
 * Copyright (C) 2010 The Android Open Source Project
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

package com.android.sdklib.build;

import com.android.SdkConstants;
import com.android.sdklib.internal.build.DebugKeyProvider;
import com.android.sdklib.internal.build.DebugKeyProvider.IKeyGenOutput;
import com.android.sdklib.internal.build.DebugKeyProvider.KeytoolException;
import com.android.sdklib.internal.build.SignedJarBuilder;
import com.android.sdklib.internal.build.SignedJarBuilder.IZipEntryFilter;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.security.PrivateKey;
import java.security.cert.X509Certificate;
import java.text.DateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Class making the final apk packaging.
 * The inputs are:
 * - packaged resources (output of aapt)
 * - code file (output of dx)
 * - Java resources coming from the project, its libraries, and its jar files
 * - Native libraries from the project or its library.
 *
 */
public final class ApkBuilder implements IArchiveBuilder {

    private static final Pattern PATTERN_NATIVELIB_EXT = Pattern.compile("^.+\\.so$",
            Pattern.CASE_INSENSITIVE);
    private static final Pattern PATTERN_BITCODELIB_EXT = Pattern.compile("^.+\\.bc$",
            Pattern.CASE_INSENSITIVE);

    /**
     * A No-op zip filter. It's used to detect conflicts.
     *
     */
    private final class NullZipFilter implements IZipEntryFilter {
        private File mInputFile;

        void reset(File inputFile) {
            mInputFile = inputFile;
        }

        @Override
        public boolean checkEntry(String archivePath) throws ZipAbortException {
            verbosePrintln("=> %s", archivePath);

            File duplicate = checkFileForDuplicate(archivePath);
            if (duplicate != null) {
                throw new DuplicateFileException(archivePath, duplicate, mInputFile);
            } else {
                mAddedFiles.put(archivePath, mInputFile);
            }

            return true;
        }
    }

    /**
     * Custom {@link IZipEntryFilter} to filter out everything that is not a standard java
     * resources, and also record whether the zip file contains native libraries.
     * <p/>Used in {@link SignedJarBuilder#writeZip(java.io.InputStream, IZipEntryFilter)} when
     * we only want the java resources from external jars.
     */
    private final class JavaAndNativeResourceFilter implements IZipEntryFilter {
        private final List<String> mNativeLibs = new ArrayList<String>();
        private boolean mNativeLibsConflict = false;
        private File mInputFile;

        @Override
        public boolean checkEntry(String archivePath) throws ZipAbortException {
            // split the path into segments.
            String[] segments = archivePath.split("/");

            // empty path? skip to next entry.
            if (segments.length == 0) {
                return false;
            }

            // Check each folders to make sure they should be included.
            // Folders like CVS, .svn, etc.. should already have been excluded from the
            // jar file, but we need to exclude some other folder (like /META-INF) so
            // we check anyway.
            for (int i = 0 ; i < segments.length - 1; i++) {
                if (!checkFolderForPackaging(segments[i])) {
                    return false;
                }
            }

            // get the file name from the path
            String fileName = segments[segments.length-1];

            boolean check = checkFileForPackaging(fileName);

            // only do additional checks if the file passes the default checks.
            if (check) {
                verbosePrintln("=> %s", archivePath);

                File duplicate = checkFileForDuplicate(archivePath);
                if (duplicate != null) {
                    throw new DuplicateFileException(archivePath, duplicate, mInputFile);
                } else {
                    mAddedFiles.put(archivePath, mInputFile);
                }

                if (archivePath.endsWith(".so") || archivePath.endsWith(".bc")) {
                    mNativeLibs.add(archivePath);

                    // only .so located in lib/ will interfere with the installation
                    if (archivePath.startsWith(SdkConstants.FD_APK_NATIVE_LIBS + "/")) {
                        mNativeLibsConflict = true;
                    }
                } else if (archivePath.endsWith(".jnilib")) {
                    mNativeLibs.add(archivePath);
                }
            }

            return check;
        }

        List<String> getNativeLibs() {
            return mNativeLibs;
        }

        boolean getNativeLibsConflict() {
            return mNativeLibsConflict;
        }

        void reset(File inputFile) {
            mInputFile = inputFile;
            mNativeLibs.clear();
            mNativeLibsConflict = false;
        }
    }

    private File mApkFile;
    private File mResFile;
    private File mDexFile;
    private PrintStream mVerboseStream;
    private SignedJarBuilder mBuilder;
    private boolean mDebugMode = false;
    private boolean mIsSealed = false;

    private final NullZipFilter mNullFilter = new NullZipFilter();
    private final JavaAndNativeResourceFilter mFilter = new JavaAndNativeResourceFilter();
    private final HashMap<String, File> mAddedFiles = new HashMap<String, File>();

    /**
     * Status for the addition of a jar file resources into the APK.
     * This indicates possible issues with native library inside the jar file.
     */
    public interface JarStatus {
        /**
         * Returns the list of native libraries found in the jar file.
         */
        List<String> getNativeLibs();

        /**
         * Returns whether some of those libraries were located in the location that Android
         * expects its native libraries.
         */
        boolean hasNativeLibsConflicts();

    }

    /** Internal implementation of {@link JarStatus}. */
    private static final class JarStatusImpl implements JarStatus {
        public final List<String> mLibs;
        public final boolean mNativeLibsConflict;

        private JarStatusImpl(List<String> libs, boolean nativeLibsConflict) {
            mLibs = libs;
            mNativeLibsConflict = nativeLibsConflict;
        }

        @Override
        public List<String> getNativeLibs() {
            return mLibs;
        }

        @Override
        public boolean hasNativeLibsConflicts() {
            return mNativeLibsConflict;
        }
    }

    /**
     * Signing information.
     *
     * Both the {@link PrivateKey} and the {@link X509Certificate} are guaranteed to be non-null.
     *
     */
    public static final class SigningInfo {
        public final PrivateKey key;
        public final X509Certificate certificate;

        private SigningInfo(PrivateKey key, X509Certificate certificate) {
            if (key == null || certificate == null) {
                throw new IllegalArgumentException("key and certificate cannot be null");
            }
            this.key = key;
            this.certificate = certificate;
        }
    }

    /**
     * Returns the key and certificate from a given debug store.
     *
     * It is expected that the store password is 'android' and the key alias and password are
     * 'androiddebugkey' and 'android' respectively.
     *
     * @param storeOsPath the OS path to the debug store.
     * @param verboseStream an option {@link PrintStream} to display verbose information
     * @return they key and certificate in a {@link SigningInfo} object or null.
     * @throws ApkCreationException
     */
    public static SigningInfo getDebugKey(String storeOsPath, final PrintStream verboseStream)
            throws ApkCreationException {
        try {
            if (storeOsPath != null) {
                File storeFile = new File(storeOsPath);
                try {
                    checkInputFile(storeFile);
                } catch (FileNotFoundException e) {
                    // ignore these since the debug store can be created on the fly anyway.
                }

                // get the debug key
                if (verboseStream != null) {
                    verboseStream.println(String.format("Using keystore: %s", storeOsPath));
                }

                IKeyGenOutput keygenOutput = null;
                if (verboseStream != null) {
                    keygenOutput = new IKeyGenOutput() {
                        @Override
                        public void out(String message) {
                            verboseStream.println(message);
                        }

                        @Override
                        public void err(String message) {
                            verboseStream.println(message);
                        }
                    };
                }

                DebugKeyProvider keyProvider = new DebugKeyProvider(
                        storeOsPath, null /*store type*/, keygenOutput);

                PrivateKey key = keyProvider.getDebugKey();
                X509Certificate certificate = (X509Certificate)keyProvider.getCertificate();

                if (key == null) {
                    throw new ApkCreationException("Unable to get debug signature key");
                }

                // compare the certificate expiration date
                if (certificate != null && certificate.getNotAfter().compareTo(new Date()) < 0) {
                    // TODO, regenerate a new one.
                    throw new ApkCreationException("Debug Certificate expired on " +
                            DateFormat.getInstance().format(certificate.getNotAfter()));
                }

                return new SigningInfo(key, certificate);
            } else {
                return null;
            }
        } catch (KeytoolException e) {
            if (e.getJavaHome() == null) {
                throw new ApkCreationException(e.getMessage() +
                        "\nJAVA_HOME seems undefined, setting it will help locating keytool automatically\n" +
                        "You can also manually execute the following command\n:" +
                        e.getCommandLine(), e);
            } else {
                throw new ApkCreationException(e.getMessage() +
                        "\nJAVA_HOME is set to: " + e.getJavaHome() +
                        "\nUpdate it if necessary, or manually execute the following command:\n" +
                        e.getCommandLine(), e);
            }
        } catch (ApkCreationException e) {
            throw e;
        } catch (Exception e) {
            throw new ApkCreationException(e);
        }
    }

    /**
     * Creates a new instance.
     *
     * This creates a new builder that will create the specified output file, using the two
     * mandatory given input files.
     *
     * An optional debug keystore can be provided. If set, it is expected that the store password
     * is 'android' and the key alias and password are 'androiddebugkey' and 'android'.
     *
     * An optional {@link PrintStream} can also be provided for verbose output. If null, there will
     * be no output.
     *
     * @param apkOsPath the OS path of the file to create.
     * @param resOsPath the OS path of the packaged resource file.
     * @param dexOsPath the OS path of the dex file. This can be null for apk with no code.
     * @param verboseStream the stream to which verbose output should go. If null, verbose mode
     *                      is not enabled.
     * @throws ApkCreationException
     */
    public ApkBuilder(String apkOsPath, String resOsPath, String dexOsPath, String storeOsPath,
            PrintStream verboseStream) throws ApkCreationException {
        this(new File(apkOsPath),
             new File(resOsPath),
             dexOsPath != null ? new File(dexOsPath) : null,
             storeOsPath,
             verboseStream);
    }

    /**
     * Creates a new instance.
     *
     * This creates a new builder that will create the specified output file, using the two
     * mandatory given input files.
     *
     * Optional {@link PrivateKey} and {@link X509Certificate} can be provided to sign the APK.
     *
     * An optional {@link PrintStream} can also be provided for verbose output. If null, there will
     * be no output.
     *
     * @param apkOsPath the OS path of the file to create.
     * @param resOsPath the OS path of the packaged resource file.
     * @param dexOsPath the OS path of the dex file. This can be null for apk with no code.
     * @param key the private key used to sign the package. Can be null.
     * @param certificate the certificate used to sign the package. Can be null.
     * @param verboseStream the stream to which verbose output should go. If null, verbose mode
     *                      is not enabled.
     * @throws ApkCreationException
     */
    public ApkBuilder(String apkOsPath, String resOsPath, String dexOsPath, PrivateKey key,
            X509Certificate certificate, PrintStream verboseStream) throws ApkCreationException {
        this(new File(apkOsPath),
             new File(resOsPath),
             dexOsPath != null ? new File(dexOsPath) : null,
             key, certificate,
             verboseStream);
    }

    /**
     * Creates a new instance.
     *
     * This creates a new builder that will create the specified output file, using the two
     * mandatory given input files.
     *
     * An optional debug keystore can be provided. If set, it is expected that the store password
     * is 'android' and the key alias and password are 'androiddebugkey' and 'android'.
     *
     * An optional {@link PrintStream} can also be provided for verbose output. If null, there will
     * be no output.
     *
     * @param apkFile the file to create
     * @param resFile the file representing the packaged resource file.
     * @param dexFile the file representing the dex file. This can be null for apk with no code.
     * @param debugStoreOsPath the OS path to the debug keystore, if needed or null.
     * @param verboseStream the stream to which verbose output should go. If null, verbose mode
     *                      is not enabled.
     * @throws ApkCreationException
     */
    public ApkBuilder(File apkFile, File resFile, File dexFile, String debugStoreOsPath,
            final PrintStream verboseStream) throws ApkCreationException {

        SigningInfo info = getDebugKey(debugStoreOsPath, verboseStream);
        if (info != null) {
            init(apkFile, resFile, dexFile, info.key, info.certificate, verboseStream);
        } else {
            init(apkFile, resFile, dexFile, null /*key*/, null/*certificate*/, verboseStream);
        }
    }

    /**
     * Creates a new instance.
     *
     * This creates a new builder that will create the specified output file, using the two
     * mandatory given input files.
     *
     * Optional {@link PrivateKey} and {@link X509Certificate} can be provided to sign the APK.
     *
     * An optional {@link PrintStream} can also be provided for verbose output. If null, there will
     * be no output.
     *
     * @param apkFile the file to create
     * @param resFile the file representing the packaged resource file.
     * @param dexFile the file representing the dex file. This can be null for apk with no code.
     * @param key the private key used to sign the package. Can be null.
     * @param certificate the certificate used to sign the package. Can be null.
     * @param verboseStream the stream to which verbose output should go. If null, verbose mode
     *                      is not enabled.
     * @throws ApkCreationException
     */
    public ApkBuilder(File apkFile, File resFile, File dexFile, PrivateKey key,
            X509Certificate certificate, PrintStream verboseStream) throws ApkCreationException {
        init(apkFile, resFile, dexFile, key, certificate, verboseStream);
    }


    /**
     * Constructor init method.
     *
     * @see #ApkBuilder(File, File, File, String, PrintStream)
     * @see #ApkBuilder(String, String, String, String, PrintStream)
     * @see #ApkBuilder(File, File, File, PrivateKey, X509Certificate, PrintStream)
     */
    private void init(File apkFile, File resFile, File dexFile, PrivateKey key,
            X509Certificate certificate, PrintStream verboseStream) throws ApkCreationException {

        try {
            checkOutputFile(mApkFile = apkFile);
            checkInputFile(mResFile = resFile);
            if (dexFile != null) {
                checkInputFile(mDexFile = dexFile);
            } else {
                mDexFile = null;
            }
            mVerboseStream = verboseStream;

            mBuilder = new SignedJarBuilder(
                    new FileOutputStream(mApkFile, false /* append */), key,
                    certificate);

            verbosePrintln("Packaging %s", mApkFile.getName());

            // add the resources
            addZipFile(mResFile);

            // add the class dex file at the root of the apk
            if (mDexFile != null) {
                addFile(mDexFile, SdkConstants.FN_APK_CLASSES_DEX);
            }

        } catch (ApkCreationException e) {
            if (mBuilder != null) {
                mBuilder.cleanUp();
            }
            throw e;
        } catch (Exception e) {
            if (mBuilder != null) {
                mBuilder.cleanUp();
            }
            throw new ApkCreationException(e);
        }
    }

    /**
     * Sets the debug mode. In debug mode, when native libraries are present, the packaging
     * will also include one or more copies of gdbserver in the final APK file.
     *
     * These are used for debugging native code, to ensure that gdbserver is accessible to the
     * application.
     *
     * There will be one version of gdbserver for each ABI supported by the application.
     *
     * the gbdserver files are placed in the libs/abi/ folders automatically by the NDK.
     *
     * @param debugMode the debug mode flag.
     */
    public void setDebugMode(boolean debugMode) {
        mDebugMode = debugMode;
    }

    /**
     * Adds a file to the APK at a given path
     * @param file the file to add
     * @param archivePath the path of the file inside the APK archive.
     * @throws ApkCreationException if an error occurred
     * @throws SealedApkException if the APK is already sealed.
     * @throws DuplicateFileException if a file conflicts with another already added to the APK
     *                                   at the same location inside the APK archive.
     */
    @Override
    public void addFile(File file, String archivePath) throws ApkCreationException,
            SealedApkException, DuplicateFileException {
        if (mIsSealed) {
            throw new SealedApkException("APK is already sealed");
        }

        try {
            doAddFile(file, archivePath);
        } catch (DuplicateFileException e) {
            mBuilder.cleanUp();
            throw e;
        } catch (Exception e) {
            mBuilder.cleanUp();
            throw new ApkCreationException(e, "Failed to add %s", file);
        }
    }

    /**
     * Adds the content from a zip file.
     * All file keep the same path inside the archive.
     * @param zipFile the zip File.
     * @throws ApkCreationException if an error occurred
     * @throws SealedApkException if the APK is already sealed.
     * @throws DuplicateFileException if a file conflicts with another already added to the APK
     *                                   at the same location inside the APK archive.
     */
    public void addZipFile(File zipFile) throws ApkCreationException, SealedApkException,
            DuplicateFileException {
        if (mIsSealed) {
            throw new SealedApkException("APK is already sealed");
        }

        try {
            verbosePrintln("%s:", zipFile);

            // reset the filter with this input.
            mNullFilter.reset(zipFile);

            // ask the builder to add the content of the file.
            FileInputStream fis = new FileInputStream(zipFile);
            mBuilder.writeZip(fis, mNullFilter);
            fis.close();
        } catch (DuplicateFileException e) {
            mBuilder.cleanUp();
            throw e;
        } catch (Exception e) {
            mBuilder.cleanUp();
            throw new ApkCreationException(e, "Failed to add %s", zipFile);
        }
    }

    /**
     * Adds the resources from a jar file.
     * @param jarFile the jar File.
     * @return a {@link JarStatus} object indicating if native libraries where found in
     *         the jar file.
     * @throws ApkCreationException if an error occurred
     * @throws SealedApkException if the APK is already sealed.
     * @throws DuplicateFileException if a file conflicts with another already added to the APK
     *                                   at the same location inside the APK archive.
     */
    public JarStatus addResourcesFromJar(File jarFile) throws ApkCreationException,
            SealedApkException, DuplicateFileException {
        if (mIsSealed) {
            throw new SealedApkException("APK is already sealed");
        }

        try {
            verbosePrintln("%s:", jarFile);

            // reset the filter with this input.
            mFilter.reset(jarFile);

            // ask the builder to add the content of the file, filtered to only let through
            // the java resources.
            FileInputStream fis = new FileInputStream(jarFile);
            mBuilder.writeZip(fis, mFilter);
            fis.close();

            // check if native libraries were found in the external library. This should
            // constitutes an error or warning depending on if they are in lib/
            return new JarStatusImpl(mFilter.getNativeLibs(), mFilter.getNativeLibsConflict());
        } catch (DuplicateFileException e) {
            mBuilder.cleanUp();
            throw e;
        } catch (Exception e) {
            mBuilder.cleanUp();
            throw new ApkCreationException(e, "Failed to add %s", jarFile);
        }
    }

    /**
     * Adds the resources from a source folder.
     * @param sourceFolder the source folder.
     * @throws ApkCreationException if an error occurred
     * @throws SealedApkException if the APK is already sealed.
     * @throws DuplicateFileException if a file conflicts with another already added to the APK
     *                                   at the same location inside the APK archive.
     */
    public void addSourceFolder(File sourceFolder) throws ApkCreationException, SealedApkException,
            DuplicateFileException {
        if (mIsSealed) {
            throw new SealedApkException("APK is already sealed");
        }

        addSourceFolder(this, sourceFolder);
    }

    /**
     * Adds the resources from a source folder to a given {@link IArchiveBuilder}
     * @param sourceFolder the source folder.
     * @throws ApkCreationException if an error occurred
     * @throws DuplicateFileException if a file conflicts with another already added to the APK
     *                                   at the same location inside the APK archive.
     */
    public static void addSourceFolder(IArchiveBuilder builder, File sourceFolder)
            throws ApkCreationException, DuplicateFileException {
        if (sourceFolder.isDirectory()) {
            try {
                // file is a directory, process its content.
                File[] files = sourceFolder.listFiles();
                if (files != null) {
                    for (File file : files) {
                        processFileForResource(builder, file, null);
                    }
                }
            } catch (DuplicateFileException e) {
                throw e;
            } catch (Exception e) {
                throw new ApkCreationException(e, "Failed to add %s", sourceFolder);
            }
        } else {
            // not a directory? check if it's a file or doesn't exist
            if (sourceFolder.exists()) {
                throw new ApkCreationException("%s is not a folder", sourceFolder);
            } else {
                throw new ApkCreationException("%s does not exist", sourceFolder);
            }
        }
    }

    /**
     * Adds the native libraries from the top native folder.
     * The content of this folder must be the various ABI folders.
     *
     * This may or may not copy gdbserver into the apk based on whether the debug mode is set.
     *
     * @param nativeFolder the native folder.
     *
     * @throws ApkCreationException if an error occurred
     * @throws SealedApkException if the APK is already sealed.
     * @throws DuplicateFileException if a file conflicts with another already added to the APK
     *                                   at the same location inside the APK archive.
     *
     * @see #setDebugMode(boolean)
     */
    public void addNativeLibraries(File nativeFolder)
            throws ApkCreationException, SealedApkException, DuplicateFileException {
        if (mIsSealed) {
            throw new SealedApkException("APK is already sealed");
        }

        if (!nativeFolder.isDirectory()) {
            // not a directory? check if it's a file or doesn't exist
            if (nativeFolder.exists()) {
                throw new ApkCreationException("%s is not a folder", nativeFolder);
            } else {
                throw new ApkCreationException("%s does not exist", nativeFolder);
            }
        }

        File[] abiList = nativeFolder.listFiles();

        verbosePrintln("Native folder: %s", nativeFolder);

        if (abiList != null) {
            for (File abi : abiList) {
                if (abi.isDirectory()) { // ignore files

                    File[] libs = abi.listFiles();
                    if (libs != null) {
                        for (File lib : libs) {
                            // only consider files that are .so or, if in debug mode, that
                            // are gdbserver executables
                            if (lib.isFile() &&
                                    (PATTERN_NATIVELIB_EXT.matcher(lib.getName()).matches() ||
                                    PATTERN_BITCODELIB_EXT.matcher(lib.getName()).matches() ||
                                            (mDebugMode &&
                                                    SdkConstants.FN_GDBSERVER.equals(
                                                            lib.getName())))) {
                                String path =
                                    SdkConstants.FD_APK_NATIVE_LIBS + "/" +
                                    abi.getName() + "/" + lib.getName();

                                try {
                                    doAddFile(lib, path);
                                } catch (IOException e) {
                                    mBuilder.cleanUp();
                                    throw new ApkCreationException(e, "Failed to add %s", lib);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    public void addNativeLibraries(List<FileEntry> entries) throws SealedApkException,
            DuplicateFileException, ApkCreationException {
        if (mIsSealed) {
            throw new SealedApkException("APK is already sealed");
        }

        for (FileEntry entry : entries) {
            try {
                doAddFile(entry.mFile, entry.mPath);
            } catch (IOException e) {
                mBuilder.cleanUp();
                throw new ApkCreationException(e, "Failed to add %s", entry.mFile);
            }
        }
    }

    public static final class FileEntry {
        public final File mFile;
        public final String mPath;

        FileEntry(File file, String path) {
            mFile = file;
            mPath = path;
        }
    }

    public static List<FileEntry> getNativeFiles(File nativeFolder, boolean debugMode)
            throws ApkCreationException  {

        if (!nativeFolder.isDirectory()) {
            // not a directory? check if it's a file or doesn't exist
            if (nativeFolder.exists()) {
                throw new ApkCreationException("%s is not a folder", nativeFolder);
            } else {
                throw new ApkCreationException("%s does not exist", nativeFolder);
            }
        }

        List<FileEntry> files = new ArrayList<FileEntry>();

        File[] abiList = nativeFolder.listFiles();

        if (abiList != null) {
            for (File abi : abiList) {
                if (abi.isDirectory()) { // ignore files

                    File[] libs = abi.listFiles();
                    if (libs != null) {
                        for (File lib : libs) {
                            // only consider files that are .so or, if in debug mode, that
                            // are gdbserver executables
                            if (lib.isFile() &&
                                    (PATTERN_NATIVELIB_EXT.matcher(lib.getName()).matches() ||
                                    PATTERN_BITCODELIB_EXT.matcher(lib.getName()).matches() ||
                                            (debugMode &&
                                                    SdkConstants.FN_GDBSERVER.equals(
                                                            lib.getName())))) {
                                String path =
                                    SdkConstants.FD_APK_NATIVE_LIBS + "/" +
                                    abi.getName() + "/" + lib.getName();

                                files.add(new FileEntry(lib, path));
                            }
                        }
                    }
                }
            }
        }

        return files;
    }



    /**
     * Seals the APK, and signs it if necessary.
     * @throws ApkCreationException
     * @throws ApkCreationException if an error occurred
     * @throws SealedApkException if the APK is already sealed.
     */
    public void sealApk() throws ApkCreationException, SealedApkException {
        if (mIsSealed) {
            throw new SealedApkException("APK is already sealed");
        }

        // close and sign the application package.
        try {
            mBuilder.close();
            mIsSealed = true;
        } catch (Exception e) {
            throw new ApkCreationException(e, "Failed to seal APK");
        } finally {
            mBuilder.cleanUp();
        }
    }

    /**
     * Output a given message if the verbose mode is enabled.
     * @param format the format string for {@link String#format(String, Object...)}
     * @param args the string arguments
     */
    private void verbosePrintln(String format, Object... args) {
        if (mVerboseStream != null) {
            mVerboseStream.println(String.format(format, args));
        }
    }

    private void doAddFile(File file, String archivePath) throws DuplicateFileException,
            IOException {
        verbosePrintln("%1$s => %2$s", file, archivePath);

        File duplicate = checkFileForDuplicate(archivePath);
        if (duplicate != null) {
            throw new DuplicateFileException(archivePath, duplicate, file);
        }

        mAddedFiles.put(archivePath, file);
        mBuilder.writeFile(file, archivePath);
    }

    /**
     * Processes a {@link File} that could be an APK {@link File}, or a folder containing
     * java resources.
     *
     * @param file the {@link File} to process.
     * @param path the relative path of this file to the source folder.
     *          Can be <code>null</code> to identify a root file.
     * @throws IOException
     * @throws DuplicateFileException if a file conflicts with another already added
     *          to the APK at the same location inside the APK archive.
     * @throws SealedApkException if the APK is already sealed.
     * @throws ApkCreationException if an error occurred
     */
    private static void processFileForResource(IArchiveBuilder builder, File file, String path)
            throws IOException, DuplicateFileException, ApkCreationException, SealedApkException {
        if (file.isDirectory()) {
            // a directory? we check it
            if (checkFolderForPackaging(file.getName())) {
                // if it's valid, we append its name to the current path.
                if (path == null) {
                    path = file.getName();
                } else {
                    path = path + "/" + file.getName();
                }

                // and process its content.
                File[] files = file.listFiles();
                if (files != null) {
                    for (File contentFile : files) {
                        processFileForResource(builder, contentFile, path);
                    }
                }
            }
        } else {
            // a file? we check it to make sure it should be added
            if (checkFileForPackaging(file.getName())) {
                // we append its name to the current path
                if (path == null) {
                    path = file.getName();
                } else {
                    path = path + "/" + file.getName();
                }

                // and add it to the apk
                builder.addFile(file, path);
            }
        }
    }

    /**
     * Checks if the given path in the APK archive has not already been used and if it has been,
     * then returns a {@link File} object for the source of the duplicate
     * @param archivePath the archive path to test.
     * @return A File object of either a file at the same location or an archive that contains a
     * file that was put at the same location.
     */
    private File checkFileForDuplicate(String archivePath) {
        return mAddedFiles.get(archivePath);
    }

    /**
     * Checks an output {@link File} object.
     * This checks the following:
     * - the file is not an existing directory.
     * - if the file exists, that it can be modified.
     * - if it doesn't exists, that a new file can be created.
     * @param file the File to check
     * @throws ApkCreationException If the check fails
     */
    private void checkOutputFile(File file) throws ApkCreationException {
        if (file.isDirectory()) {
            throw new ApkCreationException("%s is a directory!", file);
        }

        if (file.exists()) { // will be a file in this case.
            if (!file.canWrite()) {
                throw new ApkCreationException("Cannot write %s", file);
            }
        } else {
            try {
                if (!file.createNewFile()) {
                    throw new ApkCreationException("Failed to create %s", file);
                }
            } catch (IOException e) {
                throw new ApkCreationException(
                        "Failed to create '%1$ss': %2$s", file, e.getMessage());
            }
        }
    }

    /**
     * Checks an input {@link File} object.
     * This checks the following:
     * - the file is not an existing directory.
     * - that the file exists (if <var>throwIfDoesntExist</var> is <code>false</code>) and can
     *    be read.
     * @param file the File to check
     * @throws FileNotFoundException if the file is not here.
     * @throws ApkCreationException If the file is a folder or a file that cannot be read.
     */
    private static void checkInputFile(File file) throws FileNotFoundException, ApkCreationException {
        if (file.isDirectory()) {
            throw new ApkCreationException("%s is a directory!", file);
        }

        if (file.exists()) {
            if (!file.canRead()) {
                throw new ApkCreationException("Cannot read %s", file);
            }
        } else {
            throw new FileNotFoundException(String.format("%s does not exist", file));
        }
    }

    public static String getDebugKeystore() throws ApkCreationException {
        try {
            return DebugKeyProvider.getDefaultKeyStoreOsPath();
        } catch (Exception e) {
            throw new ApkCreationException(e, e.getMessage());
        }
    }

    /**
     * Checks whether a folder and its content is valid for packaging into the .apk as
     * standard Java resource.
     * @param folderName the name of the folder.
     */
    public static boolean checkFolderForPackaging(String folderName) {
        return !folderName.equalsIgnoreCase("CVS") &&
                !folderName.equalsIgnoreCase(".svn") &&
                !folderName.equalsIgnoreCase("SCCS") &&
                !folderName.equalsIgnoreCase("META-INF") &&
                !folderName.startsWith("_");
    }

    /**
     * Checks a file to make sure it should be packaged as standard resources.
     * @param fileName the name of the file (including extension)
     * @return true if the file should be packaged as standard java resources.
     */
    public static boolean checkFileForPackaging(String fileName) {
        String[] fileSegments = fileName.split("\\.");
        String fileExt = "";
        if (fileSegments.length > 1) {
            fileExt = fileSegments[fileSegments.length-1];
        }

        return checkFileForPackaging(fileName, fileExt);
    }

    /**
     * Checks a file to make sure it should be packaged as standard resources.
     * @param fileName the name of the file (including extension)
     * @param extension the extension of the file (excluding '.')
     * @return true if the file should be packaged as standard java resources.
     */
    public static boolean checkFileForPackaging(String fileName, String extension) {
        // ignore hidden files and backup files
        if (fileName.charAt(0) == '.' || fileName.charAt(fileName.length()-1) == '~') {
            return false;
        }

        return !"aidl".equalsIgnoreCase(extension) &&           // Aidl files
                !"rs".equalsIgnoreCase(extension) &&            // RenderScript files
                !"fs".equalsIgnoreCase(extension) &&            // FilterScript files
                !"rsh".equalsIgnoreCase(extension) &&           // RenderScript header files
                !"d".equalsIgnoreCase(extension) &&             // Dependency files
                !"java".equalsIgnoreCase(extension) &&          // Java files
                !"scala".equalsIgnoreCase(extension) &&         // Scala files
                !"class".equalsIgnoreCase(extension) &&         // Java class files
                !"scc".equalsIgnoreCase(extension) &&           // VisualSourceSafe
                !"swp".equalsIgnoreCase(extension) &&           // vi swap file
                !"thumbs.db".equalsIgnoreCase(fileName) &&      // image index file
                !"picasa.ini".equalsIgnoreCase(fileName) &&     // image index file
                !"package.html".equalsIgnoreCase(fileName) &&   // Javadoc
                !"overview.html".equalsIgnoreCase(fileName);    // Javadoc
    }
}
