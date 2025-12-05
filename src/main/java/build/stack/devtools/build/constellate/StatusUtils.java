
package build.stack.devtools.build.constellate;

import com.google.protobuf.Any;
import com.google.rpc.BadRequest;
import com.google.rpc.BadRequest.FieldViolation;
import com.google.rpc.Code;
import com.google.rpc.Status;
import io.grpc.StatusException;
import io.grpc.protobuf.StatusProto;

/** Some utility methods to convert exceptions to Status results. */
final class StatusUtils {
  private StatusUtils() {}

  static StatusException internalError(Exception e) {
    return StatusProto.toStatusException(internalErrorStatus(e));
  }

  static Status internalErrorStatus(Exception e) {
    // StatusProto.fromThrowable returns null on non-status errors or errors with no trailers,
    // unlike Status.fromThrowable which returns the UNKNOWN code for these.
    Status st = StatusProto.fromThrowable(e);
    return st != null
        ? st
        : Status.newBuilder().setCode(Code.INTERNAL.getNumber()).setMessage(e.getMessage()).build();
  }

  static StatusException notFoundError(String reason) {
    return StatusProto.toStatusException(notFoundStatus(reason));
  }

  static com.google.rpc.Status notFoundStatus(String reason) {
    return Status.newBuilder()
        .setCode(Code.NOT_FOUND.getNumber())
        .setMessage("not found:" + reason)
        .build();
  }

  static StatusException interruptedError(String reason) {
    return StatusProto.toStatusException(interruptedStatus(reason));
  }

  static com.google.rpc.Status interruptedStatus(String reason) {
    return Status.newBuilder()
        .setCode(Code.CANCELLED.getNumber())
        .setMessage("Server operation was interrupted for " + reason)
        .build();
  }

  static StatusException invalidArgumentError(String field, String desc) {
    return StatusProto.toStatusException(invalidArgumentStatus(field, desc));
  }

  static com.google.rpc.Status invalidArgumentStatus(String field, String desc) {
    FieldViolation v = FieldViolation.newBuilder().setField(field).setDescription(desc).build();
    return Status.newBuilder()
        .setCode(Code.INVALID_ARGUMENT.getNumber())
        .setMessage("invalid argument(s): " + field + ": " + desc)
        .addDetails(Any.pack(BadRequest.newBuilder().addFieldViolations(v).build()))
        .build();
  }
}
