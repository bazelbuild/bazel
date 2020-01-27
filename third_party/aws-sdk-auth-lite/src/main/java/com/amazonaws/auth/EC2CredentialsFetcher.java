/*
 * Copyright 2011-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *    http://aws.amazon.com/apache2.0
 *
 * This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and
 * limitations under the License.
 */
package com.amazonaws.auth;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Date;

import com.google.gson.*;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.amazonaws.SdkClientException;
import com.amazonaws.annotation.SdkInternalApi;
import com.amazonaws.internal.CredentialsEndpointProvider;
import com.amazonaws.internal.EC2CredentialsUtils;
import com.amazonaws.util.DateUtils;

/**
 * Helper class that contains the common behavior of the
 * CredentialsProviders that loads the credentials from a
 * local endpoint on an EC2 instance.
 */
@SdkInternalApi
class EC2CredentialsFetcher {

    private static final Log LOG = LogFactory.getLog(EC2CredentialsFetcher.class);

    /**
     * The threshold after the last attempt to load credentials (in
     * milliseconds) at which credentials are attempted to be refreshed.
     */
    private static final int REFRESH_THRESHOLD = 1000 * 60 * 60;

    /**
     * The threshold before credentials expire (in milliseconds) at which
     * this class will attempt to load new credentials.
     */
    private static final int EXPIRATION_THRESHOLD = 1000 * 60 * 15;

    /** The name of the Json Object that contains the access key.*/
    private static final String ACCESS_KEY_ID = "AccessKeyId";

    /** The name of the Json Object that contains the secret access key.*/
    private static final String SECRET_ACCESS_KEY = "SecretAccessKey";

    /** The name of the Json Object that contains the token.*/
    private static final String TOKEN = "Token";

    /** The current instance profile credentials */
    private volatile AWSCredentials credentials;

    /** The expiration for the current instance profile credentials */
    private volatile Date credentialsExpiration;

    /** The time of the last attempt to check for new credentials */
    protected volatile Date lastInstanceProfileCheck;

    /** Used to load the endpoint where the credentials are stored. */
    private final CredentialsEndpointProvider credentialsEndpointProvider;

    public EC2CredentialsFetcher(CredentialsEndpointProvider credentialsEndpointProvider) {
        this.credentialsEndpointProvider = credentialsEndpointProvider;
    }

    public AWSCredentials getCredentials() {
        if (needsToLoadCredentials())
            fetchCredentials();
        if (expired()) {
            throw new SdkClientException(
                    "The credentials received have been expired");
        }
        return credentials;
    }

    /**
     * Returns true if credentials are null, credentials are within expiration or
     * if the last attempt to refresh credentials is beyond the refresh threshold.
     */
     protected boolean needsToLoadCredentials() {
        if (credentials == null) return true;

        if (credentialsExpiration != null) {
            if (isWithinExpirationThreshold()) return true;
        }

        if (lastInstanceProfileCheck != null) {
            if (isPastRefreshThreshold()) return true;
        }

        return false;
    }

    /**
     * Fetches the credentials from the endpoint.
     */
    private synchronized void fetchCredentials() {
        if (!needsToLoadCredentials()) return;

        JsonElement accessKey;
        JsonElement secretKey;
        JsonObject node;
        JsonElement token;
        try {
            lastInstanceProfileCheck = new Date();

            String credentialsResponse = EC2CredentialsUtils.getInstance().readResource(
                credentialsEndpointProvider.getCredentialsEndpoint(),
                credentialsEndpointProvider.getRetryPolicy(),
                credentialsEndpointProvider.getHeaders());

            JsonParser parser = new JsonParser();

            node = parser.parse(credentialsResponse).getAsJsonObject();
            accessKey = node.get(ACCESS_KEY_ID);
            secretKey = node.get(SECRET_ACCESS_KEY);
            token = node.get(TOKEN);

            if (null == accessKey || null == secretKey) {
                throw new SdkClientException("Unable to load credentials.");
            }

            if (null != token) {
                credentials = new BasicSessionCredentials(accessKey.getAsString(),
                        secretKey.getAsString(), token.getAsString());
            } else {
                credentials = new BasicAWSCredentials(accessKey.getAsString(),
                        secretKey.getAsString());
            }

            JsonElement expirationJsonNode = node.get("Expiration");
            if (null != expirationJsonNode) {
                /*
                 * TODO: The expiration string comes in a different format
                 * than what we deal with in other parts of the SDK, so we
                 * have to convert it to the ISO8601 syntax we expect.
                 */
                String expiration = expirationJsonNode.getAsString();
                expiration = expiration.replaceAll("\\+0000$", "Z");

                try {
                    credentialsExpiration = DateUtils.parseISO8601Date(expiration);
                } catch(Exception ex) {
                    handleError("Unable to parse credentials expiration date from Amazon EC2 instance", ex);
                }
            }
        } catch (JsonSyntaxException e) {
            handleError("Unable to parse response returned from service endpoint", e);
        } catch (IOException e) {
            handleError("Unable to load credentials from service endpoint", e);
        } catch (URISyntaxException e) {
            handleError("Unable to load credentials from service endpoint", e);
        }
    }

    /**
     * Handles reporting or throwing an error encountered while requesting
     * credentials from the Amazon EC2 endpoint. The Service could be
     * briefly unavailable for a number of reasons, so
     * we need to gracefully handle falling back to valid credentials if they're
     * available, and only throw exceptions if we really can't recover.
     *
     * @param errorMessage
     *            A human readable description of the error.
     * @param e
     *            The error that occurred.
     */
    private void handleError(String errorMessage, Exception e) {
        // If we don't have any valid credentials to fall back on, then throw an exception
        if (credentials == null || expired())
            throw new SdkClientException(errorMessage, e);

        // Otherwise, just log the error and continuing using the current credentials
        LOG.debug(errorMessage, e);
    }

    public void refresh() {
        credentials = null;
    }

    /**
     * Returns true if the current credentials are within the expiration
     * threshold, and therefore, should be refreshed.
     */
    private boolean isWithinExpirationThreshold() {
        return (credentialsExpiration.getTime() - System.currentTimeMillis()) < EXPIRATION_THRESHOLD;
    }

    /**
     * Returns true if the last attempt to refresh credentials is beyond the
     * refresh threshold, and therefore the credentials should attempt to be
     * refreshed.
     */
    private boolean isPastRefreshThreshold() {
        return (System.currentTimeMillis() - lastInstanceProfileCheck.getTime()) > REFRESH_THRESHOLD;
    }

    private boolean expired() {
        if (credentialsExpiration != null) {
            if (credentialsExpiration.getTime() < System.currentTimeMillis()) {
                return true;
            }
        }

        return false;
    }

    public Date getCredentialsExpiration() {
        return credentialsExpiration;
    }

    @Override
    public String toString() {
        return getClass().getSimpleName();
    }
}
