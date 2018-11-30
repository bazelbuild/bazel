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

import com.amazonaws.auth.AWSCredentials;
import com.amazonaws.auth.AWSCredentialsProvider;
import com.amazonaws.auth.profile.internal.AwsProfileNameLoader;

import java.util.concurrent.Semaphore;

/**
 * Credentials provider based on AWS configuration profiles. This provider vends AWSCredentials from
 * the profile configuration file for the default profile, or for a specific, named profile. <p> AWS
 * credential profiles allow you to share multiple sets of AWS security credentials between
 * different tools like the AWS SDK for Java and the AWS CLI. <p> See
 * http://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html
 *
 * @see ProfilesConfigFile
 */
public class ProfileCredentialsProvider implements AWSCredentialsProvider {

    /**
     * Default refresh interval
     */
    private static final long DEFAULT_REFRESH_INTERVAL_NANOS = 5 * 60 * 1000 * 1000 * 1000L;

    /**
     * Default force reload interval
     */
    private static final long DEFAULT_FORCE_RELOAD_INTERVAL_NANOS =
            2 * DEFAULT_REFRESH_INTERVAL_NANOS;

    /**
     * The credential profiles file from which this provider loads the security credentials. Lazily
     * loaded by the double-check idiom.
     */
    private volatile ProfilesConfigFile profilesConfigFile;

    /**
     * When the profiles file was last refreshed.
     */
    private volatile long lastRefreshed;

    /**
     * The name of the credential profile
     */
    private final String profileName;

    /**
     * Used to have only one thread block on refresh, for applications making at least one call
     * every REFRESH_INTERVAL_NANOS.
     */
    private final Semaphore refreshSemaphore = new Semaphore(1);

    /**
     * Refresh interval. Defaults to {@link #DEFAULT_REFRESH_INTERVAL_NANOS}
     */
    private long refreshIntervalNanos = DEFAULT_REFRESH_INTERVAL_NANOS;

    /**
     * Force reload interval. Defaults to {@link #DEFAULT_FORCE_RELOAD_INTERVAL_NANOS}
     */
    private long refreshForceIntervalNanos = DEFAULT_FORCE_RELOAD_INTERVAL_NANOS;

    /**
     * Creates a new profile credentials provider that returns the AWS security credentials
     * configured for the default profile. Loading the credential file is deferred until the
     * getCredentials() method is called.
     */
    public ProfileCredentialsProvider() {
        this(null);
    }

    /**
     * Creates a new profile credentials provider that returns the AWS security credentials
     * configured for the named profile. Loading the credential file is deferred until the
     * getCredentials() method is called.
     *
     * @param profileName The name of a local configuration profile.
     */
    public ProfileCredentialsProvider(String profileName) {
        this((ProfilesConfigFile) null, profileName);
    }

    /**
     * Creates a new profile credentials provider that returns the AWS security credentials for the
     * specified profiles configuration file and profile name.
     *
     * @param profilesConfigFilePath The file path where the profile configuration file is located.
     * @param profileName            The name of a configuration profile in the specified
     *                               configuration file.
     */
    public ProfileCredentialsProvider(String profilesConfigFilePath, String profileName) {
        this(new ProfilesConfigFile(profilesConfigFilePath), profileName);
    }

    /**
     * Creates a new profile credentials provider that returns the AWS security credentials for the
     * specified profiles configuration file and profile name.
     *
     * @param profilesConfigFile The profile configuration file containing the profiles used by this
     *                           credentials provider or null to defer load to first use.
     * @param profileName        The name of a configuration profile in the specified configuration
     *                           file.
     */
    public ProfileCredentialsProvider(ProfilesConfigFile profilesConfigFile, String profileName) {
        this.profilesConfigFile = profilesConfigFile;
        if (this.profilesConfigFile != null) {
            this.lastRefreshed = System.nanoTime();
        }
        if (profileName == null) {
            this.profileName = AwsProfileNameLoader.INSTANCE.loadProfileName();
        } else {
            this.profileName = profileName;
        }
    }

    @Override
    public AWSCredentials getCredentials() {
        if (profilesConfigFile == null) {
            synchronized (this) {
                if (profilesConfigFile == null) {
                    profilesConfigFile = new ProfilesConfigFile();
                    lastRefreshed = System.nanoTime();
                }
            }
        }

        // Periodically check if the file on disk has been modified
        // since we last read it.
        //
        // For active applications, only have one thread block.
        // For applications that use this method in bursts, ensure the
        // credentials are never too stale.
        long now = System.nanoTime();
        long age = now - lastRefreshed;
        if (age > refreshForceIntervalNanos) {
            refresh();
        } else if (age > refreshIntervalNanos) {
            if (refreshSemaphore.tryAcquire()) {
                try {
                    refresh();
                } finally {
                    refreshSemaphore.release();
                }
            }
        }

        return profilesConfigFile.getCredentials(profileName);
    }

    @Override
    public void refresh() {
        if (profilesConfigFile != null) {
            profilesConfigFile.refresh();
            lastRefreshed = System.nanoTime();
        }
    }

    /**
     * Gets the refresh interval in nanoseconds.
     *
     * @return nanoseconds
     */
    public long getRefreshIntervalNanos() {
        return refreshIntervalNanos;
    }

    /**
     * Sets the refresh interval in nanoseconds.
     *
     * @param refreshIntervalNanos nanoseconds
     */
    public void setRefreshIntervalNanos(long refreshIntervalNanos) {
        this.refreshIntervalNanos = refreshIntervalNanos;
    }

    /**
     * Gets the forced refresh interval in nanoseconds.
     *
     * @return nanoseconds
     */
    public long getRefreshForceIntervalNanos() {
        return refreshForceIntervalNanos;
    }

    /**
     * Sets the forced refresh interval in nanoseconds.
     */
    public void setRefreshForceIntervalNanos(long refreshForceIntervalNanos) {
        this.refreshForceIntervalNanos = refreshForceIntervalNanos;
    }
}
