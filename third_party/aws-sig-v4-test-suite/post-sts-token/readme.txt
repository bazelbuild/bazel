A note about using temporary security credentials:

You can use temporary security credentials provided by the AWS Security Token Service (AWS STS) to sign a request. The process is the same as using long-term credentials but requires an additional HTTP header or query string parameter for the security token. The name of the header or query string parameter is X-Amz-Security-Token, and the value is the session token (the string that you received from AWS STS when you obtained temporary security credentials).

When you add X-Amz-Security-Token, some services require that you include this parameter in the canonical (signed) request. For other services, you add this parameter at the end, after you calculate the signature. For details see the API reference documentation for that service.

The test suite has 2 examples:

post-sts-header-before - The X-Amz-Security-Token header is part of the canonical request.

post-sts-header-after - The X-Amz-Security-Token header is added to the request after you calculate the signature.

The test suite uses this example value for X-Amz-Security-Token:

AQoDYXdzEPT//////////wEXAMPLEtc764bNrC9SAPBSM22wDOk4x4HIZ8j4FZTwdQWLWsKWHGBuFqwAeMicRXmxfpSPfIeoIYRqTflfKD8YUuwthAx7mSEI/qkPpKPi/kMcGdQrmGdeehM4IC1NtBmUpp2wUE8phUZampKsburEDy0KPkyQDYwT7WZ0wq5VSXDvp75YU9HFvlRd8Tx6q6fE8YQcHNVXAkiY9q6d+xo0rKwT38xVqr7ZD0u0iPPkUL64lIZbqBAz+scqKmlzm8FDrypNC9Yjc8fPOLn9FX9KSYvKTr4rvx3iSIlTJabIQwj2ICCR/oLxBA==