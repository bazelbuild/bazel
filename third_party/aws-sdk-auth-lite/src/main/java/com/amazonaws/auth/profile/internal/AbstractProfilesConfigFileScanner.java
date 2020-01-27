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
package com.amazonaws.auth.profile.internal;

import java.util.AbstractMap;
import java.util.Map.Entry;
import java.util.Scanner;

/**
 * An abstract template class for the generic operation that involves scanning
 * through a profile configuration file. Subclass should implement the abstract
 * methods to define the actions when different components of the profiles file
 * are detected.
 */
public abstract class AbstractProfilesConfigFileScanner {

    /**
     * Action to be performed when an empty or comment line is detected
     */
    protected abstract void onEmptyOrCommentLine(String profileName, String line);

    /**
     * Action to be performed when the starting line of a new profile is detected
     */
    protected abstract void onProfileStartingLine(String newProfileName, String line);

    /**
     * Action to be performed when the scanner reaches the end of a profile
     * section. This method is invoked either at the start of a new profile
     * section, or at the end of the file.
     */
    protected abstract void onProfileEndingLine(String prevProfileName);

    /**
     * Action to be performed when the scanner reaches the end of the
     * credentials file.
     */
    protected abstract void onEndOfFile();

    /**
     * Action to be performed when a property declaration is detected inside a
     * profile section.
     *
     * @param profileName
     *            The name of the profile where this property is declared.
     * @param propertyName
     *            The name of the property.
     * @param propertyValue
     *            The value of the property.
     * @param isSupportedProperty
     *            Whether this is a supported property according to the
     *            specification of credential profiles file.
     * @param line
     *            The original line of text where the property is declared.
     */
    protected abstract void onProfileProperty(String profileName,
                                              String propertyName,
                                              String propertyValue,
                                              boolean isSupportedProperty,
                                              String line);

    /**
     * Hook to allow subclasses to determine which properties are supported and which aren't.
     *
     * @return True if property is supported by scanner implementation, false otherwise.
     */
    protected boolean isSupportedProperty(String propertyName) {
        return true;
    }

    /**
     * Scan through the given input, and perform the defined actions.
     *
     * @param scanner
     *            The scanner for the credentials file input.
     */
    protected void run(Scanner scanner) {
        String currentProfileName = null;

        try {
            while(scanner.hasNextLine()) {
                String line = scanner.nextLine().trim();

                // Empty or comment lines
                if (line.isEmpty() || line.startsWith("#")) {
                    onEmptyOrCommentLine(currentProfileName, line);
                    continue;
                }

                // parseGroupName returns null if this line does not
                // indicate a new property group.
                String newProfileName = parseProfileName(line);
                boolean atNewProfileStartingLine = newProfileName != null;

                if (atNewProfileStartingLine) {
                    if (currentProfileName != null) {
                        onProfileEndingLine(currentProfileName);
                    }
                    onProfileStartingLine(newProfileName, line);

                    // Start the new profile
                    currentProfileName = newProfileName;
                } else {
                    // Parse the property line
                    Entry<String, String> property = parsePropertyLine(line);

                    if (currentProfileName == null) {
                        throw new IllegalArgumentException(
                                "Property is defined without a preceding profile name. "
                                + "Current line: " + line);
                    }

                    onProfileProperty(currentProfileName,
                                      property.getKey(),
                                      property.getValue(),
                                      isSupportedProperty(property.getKey()),
                                      line);
                }
            }

            // EOF
            if (currentProfileName != null) {
                onProfileEndingLine(currentProfileName);
            }

            onEndOfFile();

        } finally {
            scanner.close();
        }
    }


    /**
     * Returns the profile name if this line indicates the beginning of a new
     * profile section. Otherwise, returns null.
     */
    private static String parseProfileName(String trimmedLine) {
        if (trimmedLine.startsWith("[") && trimmedLine.endsWith("]")) {
            String profileName = trimmedLine.substring(1, trimmedLine.length() - 1);
            return profileName.trim();
        }
        return null;
    }

    private static Entry<String, String> parsePropertyLine(String propertyLine) {
        String[] pair = propertyLine.split("=", 2);
        if (pair.length != 2) {
            throw new IllegalArgumentException(
                    "Invalid property format: no '=' character is found in the line ["
                    + propertyLine + "].");
        }

        String propertyKey   = pair[0].trim();
        String propertyValue = pair[1].trim();

        return new AbstractMap.SimpleImmutableEntry<String, String>(propertyKey, propertyValue);
    }

}
