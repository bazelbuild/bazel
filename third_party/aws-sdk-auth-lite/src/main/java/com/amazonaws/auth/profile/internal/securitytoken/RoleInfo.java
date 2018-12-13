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
package com.amazonaws.auth.profile.internal.securitytoken;

import com.amazonaws.annotation.SdkInternalApi;
import com.amazonaws.annotation.SdkProtectedApi;
import com.amazonaws.auth.AWSCredentials;
import com.amazonaws.auth.AWSCredentialsProvider;
import com.amazonaws.internal.StaticCredentialsProvider;

@SdkProtectedApi
public class RoleInfo implements Cloneable {
    /**
     * <p>
     * The Amazon Resource Name (ARN) of the role to assume.
     * </p>
     */
    private String roleArn;

    /**
     * <p>
     * An identifier for the assumed role session.
     * </p>
     * <p>
     * Use the role session name to uniquely identify a session when the same
     * role is assumed by different principals or for different reasons. In
     * cross-account scenarios, the role session name is visible to, and can be
     * logged by the account that owns the role. The role session name is also
     * used in the ARN of the assumed role principal. This means that subsequent
     * cross-account API requests using the temporary security credentials will
     * expose the role session name to the external account in their CloudTrail
     * logs.
     * </p>
     */
    private String roleSessionName;

    /**
     * <p>
     * A unique identifier that is used by third parties when assuming roles in
     * their customers' accounts. For each role that the third party can assume,
     * they should instruct their customers to ensure the role's trust policy
     * checks for the external ID that the third party generated. Each time the
     * third party assumes the role, they should pass the customer's external
     * ID. The external ID is useful in order to help third parties bind a role
     * to the customer who created it. For more information about the external
     * ID, see <a href=
     * "http://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create_for-user_externalid.html"
     * >How to Use an External ID When Granting Access to Your AWS Resources to
     * a Third Party</a> in the <i>Using IAM</i>.
     * </p>
     */
    private String externalId;

    /**
     * <p>
     * Provides the credentials that are used to assume the role.
     * </p>
     */
    private AWSCredentialsProvider longLivedCredentialsProvider;

    /**
     * Default constructor for RoleInfo object. Callers should use the setter
     * or fluent setter (with...) methods to initialize the object after
     * creating it.
     */
    public RoleInfo() {
    }

    /**
     * <p>
     * The Amazon Resource Name (ARN) of the role to assume.
     * </p>
     *
     * @param roleArn
     *        The Amazon Resource Name (ARN) of the role to assume.
     */
    public void setRoleArn(String roleArn) {
        this.roleArn = roleArn;
    }

    /**
     * <p>
     * The Amazon Resource Name (ARN) of the role to assume.
     * </p>
     *
     * @return The Amazon Resource Name (ARN) of the role to assume.
     */
    public String getRoleArn() {
        return this.roleArn;
    }

    /**
     * <p>
     * The Amazon Resource Name (ARN) of the role to assume.
     * </p>
     *
     * @param roleArn
     *        The Amazon Resource Name (ARN) of the role to assume.
     * @return Returns a reference to this object so that method calls can be
     *         chained together.
     */
    public RoleInfo withRoleArn(String roleArn) {
        setRoleArn(roleArn);
        return this;
    }

    /**
     * <p>
     * An identifier for the assumed role session.
     * </p>
     * <p>
     * Use the role session name to uniquely identify a session when the same
     * role is assumed by different principals or for different reasons. In
     * cross-account scenarios, the role session name is visible to, and can be
     * logged by the account that owns the role. The role session name is also
     * used in the ARN of the assumed role principal. This means that subsequent
     * cross-account API requests using the temporary security credentials will
     * expose the role session name to the external account in their CloudTrail
     * logs.
     * </p>
     *
     * @param roleSessionName
     *        An identifier for the assumed role session. </p>
     *        <p>
     *        Use the role session name to uniquely identify a session when the
     *        same role is assumed by different principals or for different
     *        reasons. In cross-account scenarios, the role session name is
     *        visible to, and can be logged by the account that owns the role.
     *        The role session name is also used in the ARN of the assumed role
     *        principal. This means that subsequent cross-account API requests
     *        using the temporary security credentials will expose the role
     *        session name to the external account in their CloudTrail logs.
     */
    public void setRoleSessionName(String roleSessionName) {
        this.roleSessionName = roleSessionName;
    }

    /**
     * <p>
     * An identifier for the assumed role session.
     * </p>
     * <p>
     * Use the role session name to uniquely identify a session when the same
     * role is assumed by different principals or for different reasons. In
     * cross-account scenarios, the role session name is visible to, and can be
     * logged by the account that owns the role. The role session name is also
     * used in the ARN of the assumed role principal. This means that subsequent
     * cross-account API requests using the temporary security credentials will
     * expose the role session name to the external account in their CloudTrail
     * logs.
     * </p>
     *
     * @return An identifier for the assumed role session. </p>
     *         <p>
     *         Use the role session name to uniquely identify a session when the
     *         same role is assumed by different principals or for different
     *         reasons. In cross-account scenarios, the role session name is
     *         visible to, and can be logged by the account that owns the role.
     *         The role session name is also used in the ARN of the assumed role
     *         principal. This means that subsequent cross-account API requests
     *         using the temporary security credentials will expose the role
     *         session name to the external account in their CloudTrail logs.
     */
    public String getRoleSessionName() {
        return this.roleSessionName;
    }

    /**
     * <p>
     * An identifier for the assumed role session.
     * </p>
     * <p>
     * Use the role session name to uniquely identify a session when the same
     * role is assumed by different principals or for different reasons. In
     * cross-account scenarios, the role session name is visible to, and can be
     * logged by the account that owns the role. The role session name is also
     * used in the ARN of the assumed role principal. This means that subsequent
     * cross-account API requests using the temporary security credentials will
     * expose the role session name to the external account in their CloudTrail
     * logs.
     * </p>
     *
     * @param roleSessionName
     *        An identifier for the assumed role session. </p>
     *        <p>
     *        Use the role session name to uniquely identify a session when the
     *        same role is assumed by different principals or for different
     *        reasons. In cross-account scenarios, the role session name is
     *        visible to, and can be logged by the account that owns the role.
     *        The role session name is also used in the ARN of the assumed role
     *        principal. This means that subsequent cross-account API requests
     *        using the temporary security credentials will expose the role
     *        session name to the external account in their CloudTrail logs.
     * @return Returns a reference to this object so that method calls can be
     *         chained together.
     */
    public RoleInfo withRoleSessionName(String roleSessionName) {
        setRoleSessionName(roleSessionName);
        return this;
    }

    /**
     * <p>
     * A unique identifier that is used by third parties when assuming roles in
     * their customers' accounts. For each role that the third party can assume,
     * they should instruct their customers to ensure the role's trust policy
     * checks for the external ID that the third party generated. Each time the
     * third party assumes the role, they should pass the customer's external
     * ID. The external ID is useful in order to help third parties bind a role
     * to the customer who created it. For more information about the external
     * ID, see <a href=
     * "http://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create_for-user_externalid.html"
     * >How to Use an External ID When Granting Access to Your AWS Resources to
     * a Third Party</a> in the <i>Using IAM</i>.
     * </p>
     *
     * @param externalId
     *        A unique identifier that is used by third parties when assuming
     *        roles in their customers' accounts. For each role that the third
     *        party can assume, they should instruct their customers to ensure
     *        the role's trust policy checks for the external ID that the third
     *        party generated. Each time the third party assumes the role, they
     *        should pass the customer's external ID. The external ID is useful
     *        in order to help third parties bind a role to the customer who
     *        created it. For more information about the external ID, see <a
     *        href=
     *        "http://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create_for-user_externalid.html"
     *        >How to Use an External ID When Granting Access to Your AWS
     *        Resources to a Third Party</a> in the <i>Using IAM</i>.
     */
    public void setExternalId(String externalId) {
        this.externalId = externalId;
    }

    /**
     * <p>
     * A unique identifier that is used by third parties when assuming roles in
     * their customers' accounts. For each role that the third party can assume,
     * they should instruct their customers to ensure the role's trust policy
     * checks for the external ID that the third party generated. Each time the
     * third party assumes the role, they should pass the customer's external
     * ID. The external ID is useful in order to help third parties bind a role
     * to the customer who created it. For more information about the external
     * ID, see <a href=
     * "http://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create_for-user_externalid.html"
     * >How to Use an External ID When Granting Access to Your AWS Resources to
     * a Third Party</a> in the <i>Using IAM</i>.
     * </p>
     *
     * @return A unique identifier that is used by third parties when assuming
     *         roles in their customers' accounts. For each role that the third
     *         party can assume, they should instruct their customers to ensure
     *         the role's trust policy checks for the external ID that the third
     *         party generated. Each time the third party assumes the role, they
     *         should pass the customer's external ID. The external ID is useful
     *         in order to help third parties bind a role to the customer who
     *         created it. For more information about the external ID, see <a
     *         href=
     *         "http://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create_for-user_externalid.html"
     *         >How to Use an External ID When Granting Access to Your AWS
     *         Resources to a Third Party</a> in the <i>Using IAM</i>.
     */
    public String getExternalId() {
        return this.externalId;
    }

    /**
     * <p>
     * A unique identifier that is used by third parties when assuming roles in
     * their customers' accounts. For each role that the third party can assume,
     * they should instruct their customers to ensure the role's trust policy
     * checks for the external ID that the third party generated. Each time the
     * third party assumes the role, they should pass the customer's external
     * ID. The external ID is useful in order to help third parties bind a role
     * to the customer who created it. For more information about the external
     * ID, see <a href=
     * "http://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create_for-user_externalid.html"
     * >How to Use an External ID When Granting Access to Your AWS Resources to
     * a Third Party</a> in the <i>Using IAM</i>.
     * </p>
     *
     * @param externalId
     *        A unique identifier that is used by third parties when assuming
     *        roles in their customers' accounts. For each role that the third
     *        party can assume, they should instruct their customers to ensure
     *        the role's trust policy checks for the external ID that the third
     *        party generated. Each time the third party assumes the role, they
     *        should pass the customer's external ID. The external ID is useful
     *        in order to help third parties bind a role to the customer who
     *        created it. For more information about the external ID, see <a
     *        href=
     *        "http://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create_for-user_externalid.html"
     *        >How to Use an External ID When Granting Access to Your AWS
     *        Resources to a Third Party</a> in the <i>Using IAM</i>.
     * @return Returns a reference to this object so that method calls can be
     *         chained together.
     */
    public RoleInfo withExternalId(String externalId) {
        setExternalId(externalId);
        return this;
    }

    /**
     * <p>
     * Provides the credentials that are used to assume the role.
     * </p>
     * @param longLivedCredentialsProvider long lived credentials provider
     */
    public void setLongLivedCredentialsProvider(AWSCredentialsProvider longLivedCredentialsProvider) {
        this.longLivedCredentialsProvider = longLivedCredentialsProvider;
    }

    /**
     * <p>
     * Provides the credentials that are used to assume the role.
     * </p>
     * @return long lived credentials provider
     */
    public AWSCredentialsProvider getLongLivedCredentialsProvider() {
        return this.longLivedCredentialsProvider;
    }

    /**
     * <p>
     * Provides the credentials that are used to assume the role.
     * </p>
     * @param longLivedCredentialsProvider long lived credentials provider
     * @return Returns a reference to this object so that method calls can be
     *         chained together.
     */
    public RoleInfo withLongLivedCredentialsProvider(AWSCredentialsProvider longLivedCredentialsProvider) {
        setLongLivedCredentialsProvider(longLivedCredentialsProvider);
        return this;
    }

    /**
     * <p>
     * Provides the credentials that are used to assume the role.
     * </p>
     * @param longLivedCredentials long lived credentials
     * @return Returns a reference to this object so that method calls can be
     *         chained together.
     */
    public RoleInfo withLongLivedCredentials(AWSCredentials longLivedCredentials) {
        setLongLivedCredentialsProvider(new StaticCredentialsProvider(longLivedCredentials));
        return this;
    }

    /**
     * Returns a string representation of this object; useful for testing and
     * debugging.
     *
     * @return A string representation of this object.
     * @see java.lang.Object#toString()
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        if (getRoleArn() != null)
            sb.append("RoleArn: " + getRoleArn() + ",");
        if (getRoleSessionName() != null)
            sb.append("RoleSessionName: " + getRoleSessionName() + ",");
        if (getExternalId() != null)
            sb.append("ExternalId: " + getExternalId() + ",");
        sb.append("}");
        return sb.toString();
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;

        if (obj instanceof RoleInfo == false)
            return false;

        RoleInfo other = (RoleInfo) obj;
        if (other.getRoleArn() == null ^ this.getRoleArn() == null)
            return false;
        if (other.getRoleArn() != null
            && other.getRoleArn().equals(this.getRoleArn()) == false)
            return false;
        if (other.getRoleSessionName() == null
            ^ this.getRoleSessionName() == null)
            return false;
        if (other.getRoleSessionName() != null
            && other.getRoleSessionName().equals(this.getRoleSessionName()) == false)
            return false;
        if (other.getExternalId() == null ^ this.getExternalId() == null)
            return false;
        if (other.getExternalId() != null
            && other.getExternalId().equals(this.getExternalId()) == false)
            return false;
        if (other.getLongLivedCredentialsProvider() != this.getLongLivedCredentialsProvider())
            return false;
        return true;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int hashCode = 1;

        hashCode = prime * hashCode
                   + ((getRoleArn() == null) ? 0 : getRoleArn().hashCode());
        hashCode = prime
                   * hashCode
                   + ((getRoleSessionName() == null) ? 0 : getRoleSessionName()
                .hashCode());
        hashCode = prime * hashCode
                   + ((getExternalId() == null) ? 0 : getExternalId().hashCode());
        hashCode = prime * hashCode
                   + ((getLongLivedCredentialsProvider() == null) ? 0 : getLongLivedCredentialsProvider().hashCode());
        return hashCode;
    }

    @Override
    public RoleInfo clone() {
        try {
            return (RoleInfo) super.clone();
        } catch (CloneNotSupportedException e) {
            throw new IllegalStateException(
                    "Got a CloneNotSupportedException from Object.clone() "
                    + "even though we're Cloneable!", e);
        }
    }
}
