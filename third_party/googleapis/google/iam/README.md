# Google Identity and Access Management (IAM) API

Documentation of the access control API that will be implemented by all
1st party services provided by the Google Cloud Platform (like Cloud Storage,
Compute Engine, App Engine).

Any implementation of an API that offers access control features
will implement the google.iam.v1.IAMPolicy interface.

## Data model

Access control is applied when a principal (user or service account), takes
some action on a resource exposed by a service. Resources, identified by
URI-like names, are the unit of access control specification. It is up to
the service implementations to choose what granularity of access control to
support and what set of actions (permissions) to support for the resources
they provide. For example one database service may allow access control to be
specified only at the Table level, whereas another might allow access control
to also be specified at the Column level.

This is intentionally not a CRUD style API because access control policies
are created and deleted implicitly with the resources to which they are
attached.

## Policy

A `Policy` consists of a list of bindings. A `Binding` binds a set of members
to a role, where the members can include user accounts, user groups, user
domains, and service accounts. A role is a named set of permissions, defined
by the IAM system. The definition of a role is outside the policy.

A permission check involves determining the roles that include the specified
permission, and then determining if the principal specified by the check is a
member of a binding to at least one of these roles. The membership check is
recursive when a group is bound to a role.