/*
 * Copyright 2014-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * A copy of the License is located at
 *
 *  http://aws.amazon.com/apache2.0
 *
 * or in the "license" file accompanying this file. This file is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */
package com.amazonaws.auth.profile;

import com.amazonaws.SdkClientException;
import com.amazonaws.auth.profile.internal.AbstractProfilesConfigFileScanner;
import com.amazonaws.auth.profile.internal.Profile;
import com.amazonaws.auth.profile.internal.ProfileKeyConstants;
import com.amazonaws.util.StringUtils;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Scanner;
import java.util.Set;
import java.util.UUID;

/**
 * The class for creating and modifying the credential profiles file.
 */
public class ProfilesConfigFileWriter {

    private static final Log LOG = LogFactory.getLog(ProfilesConfigFileWriter.class);

    /**
     * Write all the credential profiles to a file. Note that this method will
     * clobber the existing content in the destination file if it's in the
     * overwrite mode. Use {@link #modifyOrInsertProfiles(File, Profile...)}
     * instead, if you want to perform in-place modification on your existing
     * credentials file.
     *
     * @param destination
     *            The destination file where the credentials will be written to.
     * @param overwrite
     *            If true, this method If false, this method will throw
     *            exception if the file already exists.
     * @param profiles
     *            All the credential profiles to be written.
     */
    public static void dumpToFile(File destination, boolean overwrite, Profile... profiles) {
        if (destination.exists() && !overwrite) {
            throw new SdkClientException(
                    "The destination file already exists. " +
                    "Set overwrite=true if you want to clobber the existing " +
                    "content and completely re-write the file.");
        }

        OutputStreamWriter writer;
        try {
            writer = new OutputStreamWriter(new FileOutputStream(destination, false), StringUtils.UTF8);

        } catch (IOException ioe) {
            throw new SdkClientException(
                    "Unable to open the destination file.", ioe);
        }

        try {
            final Map<String, Profile> modifications = new LinkedHashMap<String, Profile>();
            for (Profile profile : profiles) {
                modifications.put(profile.getProfileName(), profile);
            }
            ProfilesConfigFileWriterHelper writerHelper = new ProfilesConfigFileWriterHelper(writer, modifications);

            writerHelper.writeWithoutExistingContent();
        } finally {
            try { writer.close(); } catch (IOException ioe) {}
        }

    }

    /**
     * Modify or insert new profiles into an existing credentials file by
     * in-place modification. Only the properties of the affected profiles will
     * be modified; all the unaffected profiles and comment lines will remain
     * the same. This method does not support renaming a profile.
     *
     * @param destination
     *            The destination file to modify
     * @param profiles
     *            All the credential profiles to be written.
     */
    public static void modifyOrInsertProfiles(File destination, Profile... profiles) {
        final Map<String, Profile> modifications = new LinkedHashMap<String, Profile>();
        for (Profile profile : profiles) {
            modifications.put(profile.getProfileName(), profile);
        }

        modifyProfiles(destination, modifications);
    }

    /**
     * Modify one profile in the existing credentials file by in-place
     * modification. This method will rename the existing profile if the
     * specified Profile has a different name.
     *
     * @param destination
     *            The destination file to modify
     * @param profileName
     *            The name of the existing profile to be modified
     * @param newProfile
     *            The new Profile object.
     */
    public static void modifyOneProfile(File destination, String profileName, Profile newProfile) {
        final Map<String, Profile> modifications = Collections.singletonMap(profileName, newProfile);

        modifyProfiles(destination, modifications);
    }

    /**
     * Remove one or more profiles from the existing credentials file by
     * in-place modification.
     *
     * @param destination
     *            The destination file to modify
     * @param profileNames
     *            The names of all the profiles to be deleted.
     */
    public static void deleteProfiles(File destination, String... profileNames) {
        final Map<String, Profile> modifications = new LinkedHashMap<String, Profile>();
        for (String profileName : profileNames) {
            modifications.put(profileName, null); // null value indicates a deletion
        }

        modifyProfiles(destination, modifications);
    }

    /**
     * A package-private method that supports all kinds of profile modification,
     * including renaming or deleting one or more profiles.
     *
     * @param modifications
     *            Use null key value to indicate a profile that is to be
     *            deleted.
     */
    static void modifyProfiles(File destination, Map<String, Profile> modifications) {
        final boolean inPlaceModify = destination.exists();
        File stashLocation = null;

        // Stash the original file, before we apply the changes
        if (inPlaceModify) {
            boolean stashed = false;

            try {
                // We can't use File.createTempFile, since it will always create
                // that file no matter what, and File.reNameTo does not allow
                // the destination to be an existing file
                stashLocation = new File(destination.getParentFile(),
                        destination.getName() + ".bak."
                                + UUID.randomUUID().toString());
                stashed = destination.renameTo(stashLocation);

                if (LOG.isDebugEnabled()) {
                    LOG.debug(String
                            .format("The original credentials file is stashed to loaction (%s).",
                                    stashLocation.getAbsolutePath()));
                }

            } finally {
                if (!stashed) {
                    throw new SdkClientException(
                            "Failed to stash the existing credentials file " +
                            "before applying the changes.");
                }
            }
        }

        OutputStreamWriter writer = null;
        try {
            writer = new OutputStreamWriter(new FileOutputStream(destination), StringUtils.UTF8);
            ProfilesConfigFileWriterHelper writerHelper = new ProfilesConfigFileWriterHelper(writer, modifications);

            if (inPlaceModify) {
                Scanner existingContent = new Scanner(stashLocation, StringUtils.UTF8.name());
                writerHelper.writeWithExistingContent(existingContent);
            } else {
                writerHelper.writeWithoutExistingContent();
            }

            // Make sure the output is valid and can be loaded by the loader
            new ProfilesConfigFile(destination);

            if ( inPlaceModify && !stashLocation.delete() ) {
                if (LOG.isDebugEnabled()) {
                    LOG.debug(String
                            .format("Successfully modified the credentials file. But failed to " +
                                    "delete the stashed copy of the original file (%s).",
                                    stashLocation.getAbsolutePath()));
                }
            }

        } catch (Exception e) {
            // Restore the stashed file
            if (inPlaceModify) {
                boolean restored = false;

                try {
                    // We don't really care about what destination.delete()
                    // returns, since the file might not have been created when
                    // the error occurred.
                    if ( !destination.delete() ) {
                        LOG.debug("Unable to remove the credentials file "
                                + "before restoring the original one.");
                    }
                    restored = stashLocation.renameTo(destination);
                } finally {
                    if (!restored) {
                        throw new SdkClientException(
                                "Unable to restore the original credentials file. " +
                                "File content stashed in " + stashLocation.getAbsolutePath());
                    }
                }
            }

            throw new SdkClientException(
                    "Unable to modify the credentials file. " +
                    "(The original file has been restored.)",
                    e);

        } finally {
            try {
                if (writer != null) writer.close();
            } catch (IOException e) {}
        }
    }

    /**
     * Implementation of AbstractProfilesConfigFileScanner, which reads the
     * content from an existing credentials file (if any) and then modifies some
     * of the profile properties in place.
     */
    private static class ProfilesConfigFileWriterHelper extends AbstractProfilesConfigFileScanner {

        /** The writer where the modified profiles will be output to */
        private final Writer writer;

        /** Map of all the profiles to be modified, keyed by profile names */
        private final Map<String, Profile> newProfiles = new LinkedHashMap<String, Profile>();

        /** Map of the names of all the profiles to be deleted */
        private final Set<String> deletedProfiles= new HashSet<String>();

        private final StringBuilder buffer = new StringBuilder();
        private final Map<String, Set<String>> existingProfileProperties = new HashMap<String, Set<String>>();

        /**
         * Creates ProfilesConfigFileWriterHelper with the specified new
         * profiles.
         *
         * @param writer
         *            The writer where the modified content is output to.
         * @param modifications
         *            A map of all the new profiles, keyed by the profile name.
         *            If a profile name is associated with a null value, it's
         *            profile content will be removed.
         */
        public ProfilesConfigFileWriterHelper(Writer writer, Map<String, Profile> modifications) {
            this.writer = writer;

            for (Entry<String, Profile> entry : modifications.entrySet()) {
                String profileName = entry.getKey();
                Profile profile    = entry.getValue();

                if (profile == null) {
                    deletedProfiles.add(profileName);
                } else {
                    newProfiles.put(profileName, profile);
                }
            }
        }

        /**
         * Append the new profiles to the writer, by reading from empty content.
         */
        public void writeWithoutExistingContent() {
            buffer.setLength(0);
            existingProfileProperties.clear();

            // Use empty String as input, since we are bootstrapping a new file.
            run(new Scanner(""));
        }

        /**
         * Read the existing content of a credentials file, and then make
         * in-place modification according to the new profiles specified in this
         * class.
         */
        public void writeWithExistingContent(Scanner existingContent) {
            buffer.setLength(0);
            existingProfileProperties.clear();

            run(existingContent);
        }

        @Override
        protected void onEmptyOrCommentLine(String profileName, String line) {
            /*
             * Buffer the line until we reach the next property line or the end
             * of the profile. We do this so that new properties could be
             * inserted at more appropriate location. For example:
             *
             * [default]
             * # access key
             * aws_access_key_id=aaa
             * # secret key
             * aws_secret_access_key=sss
             * # We want new properties to be inserted before this line
             * # instead of after the following empty line
             *
             * [next profile]
             * ...
             */
            if (profileName == null || !deletedProfiles.contains(profileName)) {
                buffer(line);
            }
        }

        @Override
        protected void onProfileStartingLine(String profileName, String line) {
            existingProfileProperties.put(profileName, new HashSet<String>());

            // Copy the line after flush the buffer
            flush();

            if (deletedProfiles.contains(profileName))
                return;

            // If the profile name is changed
            if (newProfiles.get(profileName) != null) {
                String newProfileName = newProfiles.get(profileName).getProfileName();
                if ( !newProfileName.equals(profileName) ) {
                    line = "[" + newProfileName + "]";
                }
            }

            writeLine(line);
        }

        @Override
        protected void onProfileEndingLine(String prevProfileName) {
            // Check whether we need to insert new properties into this profile
            Profile modifiedProfile = newProfiles.get(prevProfileName);
            if (modifiedProfile != null) {
                for (Entry<String, String> entry : modifiedProfile.getProperties().entrySet()) {
                    String propertyKey   = entry.getKey();
                    String propertyValue = entry.getValue();
                    if ( !existingProfileProperties.get(prevProfileName).contains(propertyKey) ) {
                        writeProperty(propertyKey, propertyValue);
                    }
                }
            }

            // flush all the buffered comments and empty lines
            flush();
        }

        @Override
        protected void onProfileProperty(String profileName,
                String propertyKey, String propertyValue,
                boolean isSupportedProperty, String line) {
            // Record that this property key has been declared for this profile
            if (existingProfileProperties.get(profileName) == null) {
                existingProfileProperties.put(profileName, new HashSet<String>());
            }
            existingProfileProperties.get(profileName).add(propertyKey);

            if (deletedProfiles.contains(profileName))
                return;

            // Keep the unsupported properties
            if ( !isSupportedProperty ) {
                writeLine(line);
                return;
            }

            // flush all the buffered comments and empty lines before this property line
            flush();

            // Modify the property value
            if (newProfiles.containsKey(profileName)) {
                String newValue = newProfiles.get(profileName)
                        .getPropertyValue(propertyKey);
                if (newValue != null) {
                    writeProperty(propertyKey, newValue);
                }
                // else remove that line
            } else {
                writeLine(line);
            }

        }

        @Override
        protected void onEndOfFile() {
            // Append profiles that don't exist in the original file
            for (Entry<String, Profile> entry : newProfiles.entrySet()) {
                String profileName = entry.getKey();
                Profile profile    = entry.getValue();

                if ( !existingProfileProperties.containsKey(profileName) ) {
                    // The profile name is not found in the file
                    // Append the profile properties
                    writeProfile(profile);
                    writeLine("");
                }
            }

            // Flush the "real" writer
            try {
                writer.flush();
            } catch (IOException ioe) {
                throw new SdkClientException(
                        "Unable to write to the target file to persist the profile credentials.",
                        ioe);
            }
        }

        /**
         * ProfilesConfigFileWriter still deals with legacy {@link Profile} interface so it can only
         * modify credential related properties. All other properties should be preserved when
         * modifying profiles.
         */
        @Override
        protected boolean isSupportedProperty(String propertyName) {
            return ProfileKeyConstants.AWS_ACCESS_KEY_ID.equals(propertyName) ||
                   ProfileKeyConstants.AWS_SECRET_ACCESS_KEY.equals(propertyName) ||
                   ProfileKeyConstants.AWS_SESSION_TOKEN.equals(propertyName) ||
                   ProfileKeyConstants.EXTERNAL_ID.equals(propertyName) ||
                   ProfileKeyConstants.ROLE_ARN.equals(propertyName) ||
                   ProfileKeyConstants.ROLE_SESSION_NAME.equals(propertyName) ||
                   ProfileKeyConstants.SOURCE_PROFILE.equals(propertyName);
        }

        /* Private interface */

        private void writeProfile(Profile profile) {
            writeProfileName(profile.getProfileName());

            for (Entry<String, String> entry : profile.getProperties().entrySet()) {
                writeProperty(entry.getKey(), entry.getValue());
            }
        }

        private void writeProfileName(String profileName) {
            writeLine(String.format("[%s]", profileName));
        }

        private void writeProperty(String propertyKey, String propertyValue) {
            writeLine(String.format("%s=%s", propertyKey, propertyValue));
        }

        private void writeLine(String line) {
            append(String.format("%s%n", line));
        }

        /**
         * This method handles IOException that occurs when calling the append
         * method on the writer.
         */
        private void append(String str) {
            try {
                writer.append(str);
            } catch (IOException ioe) {
                throw new SdkClientException(
                        "Unable to write to the target file to persist the profile credentials.",
                        ioe);
            }
        }

        private void flush() {
            if (buffer.length() != 0) {
                append(buffer.toString());
                buffer.setLength(0);
            }
        }

        private void buffer(String line) {
            buffer.append(String.format("%s%n", line));
        }
    }
}
