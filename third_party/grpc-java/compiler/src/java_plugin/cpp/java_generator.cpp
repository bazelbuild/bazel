/*
 * Copyright 2019 The gRPC Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "java_generator.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <vector>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/stubs/common.h>

// Protobuf 3.21 changed the name of this file.
#if GOOGLE_PROTOBUF_VERSION >= 3021000
  #include <google/protobuf/compiler/java/names.h>
#else
  #include <google/protobuf/compiler/java/java_names.h>
#endif

// Stringify helpers used solely to cast GRPC_VERSION
#ifndef STR
#define STR(s) #s
#endif

#ifndef XSTR
#define XSTR(s) STR(s)
#endif

#ifdef ABSL_FALLTHROUGH_INTENDED
#define FALLTHROUGH ABSL_FALLTHROUGH_INTENDED
#else
#define FALLTHROUGH
#endif

namespace java_grpc_generator {

namespace protobuf = google::protobuf;

using protobuf::Descriptor;
using protobuf::FileDescriptor;
using protobuf::MethodDescriptor;
using protobuf::ServiceDescriptor;
using protobuf::SourceLocation;
using protobuf::io::Printer;
using std::to_string;

// java keywords from: https://docs.oracle.com/javase/specs/jls/se8/html/jls-3.html#jls-3.9
static std::set<std::string> java_keywords = {
  "abstract",
  "assert",
  "boolean",
  "break",
  "byte",
  "case",
  "catch",
  "char",
  "class",
  "const",
  "continue",
  "default",
  "do",
  "double",
  "else",
  "enum",
  "extends",
  "final",
  "finally",
  "float",
  "for",
  "goto",
  "if",
  "implements",
  "import",
  "instanceof",
  "int",
  "interface",
  "long",
  "native",
  "new",
  "package",
  "private",
  "protected",
  "public",
  "return",
  "short",
  "static",
  "strictfp",
  "super",
  "switch",
  "synchronized",
  "this",
  "throw",
  "throws",
  "transient",
  "try",
  "void",
  "volatile",
  "while",
  // additional ones added by us
  "true",
  "false",
};

// Adjust a method name prefix identifier to follow the JavaBean spec:
//   - decapitalize the first letter
//   - remove embedded underscores & capitalize the following letter
//  Finally, if the result is a reserved java keyword, append an underscore.
static std::string MixedLower(const std::string& word) {
  std::string w;
  w += tolower(word[0]);
  bool after_underscore = false;
  for (size_t i = 1; i < word.length(); ++i) {
    if (word[i] == '_') {
      after_underscore = true;
    } else {
      w += after_underscore ? toupper(word[i]) : word[i];
      after_underscore = false;
    }
  }
  if (java_keywords.find(w) != java_keywords.end()) {
    return w + "_";
  }
  return w;
}

// Converts to the identifier to the ALL_UPPER_CASE format.
//   - An underscore is inserted where a lower case letter is followed by an
//     upper case letter.
//   - All letters are converted to upper case
static std::string ToAllUpperCase(const std::string& word) {
  std::string w;
  for (size_t i = 0; i < word.length(); ++i) {
    w += toupper(word[i]);
    if ((i < word.length() - 1) && islower(word[i]) && isupper(word[i + 1])) {
      w += '_';
    }
  }
  return w;
}

static inline std::string LowerMethodName(const MethodDescriptor* method) {
  return MixedLower(method->name());
}

static inline std::string MethodPropertiesFieldName(const MethodDescriptor* method) {
  return "METHOD_" + ToAllUpperCase(method->name());
}

static inline std::string MethodPropertiesGetterName(const MethodDescriptor* method) {
  return MixedLower("get_" + method->name() + "_method");
}

static inline std::string MethodIdFieldName(const MethodDescriptor* method) {
  return "METHODID_" + ToAllUpperCase(method->name());
}

static inline std::string MessageFullJavaName(const Descriptor* desc) {
  return protobuf::compiler::java::ClassName(desc);
}

// TODO(nmittler): Remove once protobuf includes javadoc methods in distribution.
template <typename ITR>
static void GrpcSplitStringToIteratorUsing(const std::string& full,
                                       const char* delim,
                                       ITR& result) {
  // Optimize the common case where delim is a single character.
  if (delim[0] != '\0' && delim[1] == '\0') {
    char c = delim[0];
    const char* p = full.data();
    const char* end = p + full.size();
    while (p != end) {
      if (*p == c) {
        ++p;
      } else {
        const char* start = p;
        while (++p != end && *p != c);
        *result++ = std::string(start, p - start);
      }
    }
    return;
  }

  std::string::size_type begin_index, end_index;
  begin_index = full.find_first_not_of(delim);
  while (begin_index != std::string::npos) {
    end_index = full.find_first_of(delim, begin_index);
    if (end_index == std::string::npos) {
      *result++ = full.substr(begin_index);
      return;
    }
    *result++ = full.substr(begin_index, (end_index - begin_index));
    begin_index = full.find_first_not_of(delim, end_index);
  }
}

// TODO(nmittler): Remove once protobuf includes javadoc methods in distribution.
static void GrpcSplitStringUsing(const std::string& full,
                             const char* delim,
                             std::vector<std::string>* result) {
  std::back_insert_iterator< std::vector<std::string> > it(*result);
  GrpcSplitStringToIteratorUsing(full, delim, it);
}

// TODO(nmittler): Remove once protobuf includes javadoc methods in distribution.
static std::vector<std::string> GrpcSplit(const std::string& full, const char* delim) {
  std::vector<std::string> result;
  GrpcSplitStringUsing(full, delim, &result);
  return result;
}

// TODO(nmittler): Remove once protobuf includes javadoc methods in distribution.
static std::string GrpcEscapeJavadoc(const std::string& input) {
  std::string result;
  result.reserve(input.size() * 2);

  char prev = '*';

  for (std::string::size_type i = 0; i < input.size(); i++) {
    char c = input[i];
    switch (c) {
      case '*':
        // Avoid "/*".
        if (prev == '/') {
          result.append("&#42;");
        } else {
          result.push_back(c);
        }
        break;
      case '/':
        // Avoid "*/".
        if (prev == '*') {
          result.append("&#47;");
        } else {
          result.push_back(c);
        }
        break;
      case '@':
        // '@' starts javadoc tags including the @deprecated tag, which will
        // cause a compile-time error if inserted before a declaration that
        // does not have a corresponding @Deprecated annotation.
        result.append("&#64;");
        break;
      case '<':
        // Avoid interpretation as HTML.
        result.append("&lt;");
        break;
      case '>':
        // Avoid interpretation as HTML.
        result.append("&gt;");
        break;
      case '&':
        // Avoid interpretation as HTML.
        result.append("&amp;");
        break;
      case '\\':
        // Java interprets Unicode escape sequences anywhere!
        result.append("&#92;");
        break;
      default:
        result.push_back(c);
        break;
    }

    prev = c;
  }

  return result;
}

// TODO(nmittler): Remove once protobuf includes javadoc methods in distribution.
template <typename DescriptorType>
static std::string GrpcGetCommentsForDescriptor(const DescriptorType* descriptor) {
  SourceLocation location;
  if (descriptor->GetSourceLocation(&location)) {
    return location.leading_comments.empty() ?
      location.trailing_comments : location.leading_comments;
  }
  return std::string();
}

// TODO(nmittler): Remove once protobuf includes javadoc methods in distribution.
static std::vector<std::string> GrpcGetDocLines(const std::string& comments) {
  if (!comments.empty()) {
    // TODO(kenton):  Ideally we should parse the comment text as Markdown and
    //   write it back as HTML, but this requires a Markdown parser.  For now
    //   we just use <pre> to get fixed-width text formatting.

    // If the comment itself contains block comment start or end markers,
    // HTML-escape them so that they don't accidentally close the doc comment.
    std::string escapedComments = GrpcEscapeJavadoc(comments);

    std::vector<std::string> lines = GrpcSplit(escapedComments, "\n");
    while (!lines.empty() && lines.back().empty()) {
      lines.pop_back();
    }
    return lines;
  }
  return std::vector<std::string>();
}

// TODO(nmittler): Remove once protobuf includes javadoc methods in distribution.
template <typename DescriptorType>
static std::vector<std::string> GrpcGetDocLinesForDescriptor(const DescriptorType* descriptor) {
  return GrpcGetDocLines(GrpcGetCommentsForDescriptor(descriptor));
}

// TODO(nmittler): Remove once protobuf includes javadoc methods in distribution.
static void GrpcWriteDocCommentBody(Printer* printer,
                                    const std::vector<std::string>& lines,
                                    bool surroundWithPreTag) {
  if (!lines.empty()) {
    if (surroundWithPreTag) {
      printer->Print(" * <pre>\n");
    }

    for (size_t i = 0; i < lines.size(); i++) {
      // Most lines should start with a space.  Watch out for lines that start
      // with a /, since putting that right after the leading asterisk will
      // close the comment.
      if (!lines[i].empty() && lines[i][0] == '/') {
        printer->Print(" * $line$\n", "line", lines[i]);
      } else {
        printer->Print(" *$line$\n", "line", lines[i]);
      }
    }

    if (surroundWithPreTag) {
      printer->Print(" * </pre>\n");
    }
  }
}

// TODO(nmittler): Remove once protobuf includes javadoc methods in distribution.
static void GrpcWriteDocComment(Printer* printer, const std::string& comments) {
  printer->Print("/**\n");
  std::vector<std::string> lines = GrpcGetDocLines(comments);
  GrpcWriteDocCommentBody(printer, lines, false);
  printer->Print(" */\n");
}

// TODO(nmittler): Remove once protobuf includes javadoc methods in distribution.
static void GrpcWriteServiceDocComment(Printer* printer,
                                       const ServiceDescriptor* service) {
  // Deviating from protobuf to avoid extraneous docs
  // (see https://github.com/google/protobuf/issues/1406);
  printer->Print("/**\n");
  std::vector<std::string> lines = GrpcGetDocLinesForDescriptor(service);
  GrpcWriteDocCommentBody(printer, lines, true);
  printer->Print(" */\n");
}

// TODO(nmittler): Remove once protobuf includes javadoc methods in distribution.
void GrpcWriteMethodDocComment(Printer* printer,
                           const MethodDescriptor* method) {
  // Deviating from protobuf to avoid extraneous docs
  // (see https://github.com/google/protobuf/issues/1406);
  printer->Print("/**\n");
  std::vector<std::string> lines = GrpcGetDocLinesForDescriptor(method);
  GrpcWriteDocCommentBody(printer, lines, true);
  printer->Print(" */\n");
}

static void PrintMethodFields(
    const ServiceDescriptor* service, std::map<std::string, std::string>* vars,
    Printer* p, ProtoFlavor flavor) {
  p->Print("// Static method descriptors that strictly reflect the proto.\n");
  (*vars)["service_name"] = service->name();
  for (int i = 0; i < service->method_count(); ++i) {
    const MethodDescriptor* method = service->method(i);
    (*vars)["arg_in_id"] = to_string(2 * i);
    (*vars)["arg_out_id"] = to_string(2 * i + 1);
    (*vars)["method_name"] = method->name();
    (*vars)["input_type"] = MessageFullJavaName(method->input_type());
    (*vars)["output_type"] = MessageFullJavaName(method->output_type());
    (*vars)["method_field_name"] = MethodPropertiesFieldName(method);
    (*vars)["method_new_field_name"] = MethodPropertiesGetterName(method);
    (*vars)["method_method_name"] = MethodPropertiesGetterName(method);
    bool client_streaming = method->client_streaming();
    bool server_streaming = method->server_streaming();
    if (client_streaming) {
      if (server_streaming) {
        (*vars)["method_type"] = "BIDI_STREAMING";
      } else {
        (*vars)["method_type"] = "CLIENT_STREAMING";
      }
    } else {
      if (server_streaming) {
        (*vars)["method_type"] = "SERVER_STREAMING";
      } else {
        (*vars)["method_type"] = "UNARY";
      }
    }

    if (flavor == ProtoFlavor::LITE) {
      (*vars)["ProtoUtils"] = "io.grpc.protobuf.lite.ProtoLiteUtils";
    } else {
      (*vars)["ProtoUtils"] = "io.grpc.protobuf.ProtoUtils";
    }
    p->Print(
        *vars,
        "private static volatile $MethodDescriptor$<$input_type$,\n"
        "    $output_type$> $method_new_field_name$;\n"
        "\n"
        "@$RpcMethod$(\n"
        "    fullMethodName = SERVICE_NAME + '/' + \"$method_name$\",\n"
        "    requestType = $input_type$.class,\n"
        "    responseType = $output_type$.class,\n"
        "    methodType = $MethodType$.$method_type$)\n"
        "public static $MethodDescriptor$<$input_type$,\n"
        "    $output_type$> $method_method_name$() {\n"
        "  $MethodDescriptor$<$input_type$, $output_type$> $method_new_field_name$;\n"
        "  if (($method_new_field_name$ = $service_class_name$.$method_new_field_name$) == null) {\n"
        "    synchronized ($service_class_name$.class) {\n"
        "      if (($method_new_field_name$ = $service_class_name$.$method_new_field_name$) == null) {\n"
        "        $service_class_name$.$method_new_field_name$ = $method_new_field_name$ =\n"
        "            $MethodDescriptor$.<$input_type$, $output_type$>newBuilder()\n"
        "            .setType($MethodType$.$method_type$)\n"
        "            .setFullMethodName(generateFullMethodName(SERVICE_NAME, \"$method_name$\"))\n");
        
    bool safe = method->options().idempotency_level()
        == protobuf::MethodOptions_IdempotencyLevel_NO_SIDE_EFFECTS;
    if (safe) {
      p->Print(*vars, "            .setSafe(true)\n");
    } else {
      bool idempotent = method->options().idempotency_level()
          == protobuf::MethodOptions_IdempotencyLevel_IDEMPOTENT;
      if (idempotent) {
        p->Print(*vars, "            .setIdempotent(true)\n");
      }
    }
        
    p->Print(
        *vars,
        "            .setSampledToLocalTracing(true)\n"
        "            .setRequestMarshaller($ProtoUtils$.marshaller(\n"
        "                $input_type$.getDefaultInstance()))\n"
        "            .setResponseMarshaller($ProtoUtils$.marshaller(\n"
        "                $output_type$.getDefaultInstance()))\n");

    (*vars)["proto_method_descriptor_supplier"] = service->name() + "MethodDescriptorSupplier";
    if (flavor == ProtoFlavor::NORMAL) {
      p->Print(
          *vars,
        "            .setSchemaDescriptor(new $proto_method_descriptor_supplier$(\"$method_name$\"))\n");
    }

    p->Print(
        *vars,
        "            .build();\n");
    p->Print(*vars,
        "      }\n"
        "    }\n"
        "  }\n"
        "  return $method_new_field_name$;\n"
        "}\n"
        "\n");
  }
}

enum StubType {
  ASYNC_INTERFACE = 0,
  BLOCKING_CLIENT_INTERFACE = 1,
  FUTURE_CLIENT_INTERFACE = 2,
  BLOCKING_SERVER_INTERFACE = 3,
  ASYNC_CLIENT_IMPL = 4,
  BLOCKING_CLIENT_IMPL = 5,
  FUTURE_CLIENT_IMPL = 6,
  ABSTRACT_CLASS = 7,
};

enum CallType {
  ASYNC_CALL = 0,
  BLOCKING_CALL = 1,
  FUTURE_CALL = 2
};

static void PrintBindServiceMethodBody(const ServiceDescriptor* service,
                                   std::map<std::string, std::string>* vars,
                                   Printer* p);

// Prints a StubFactory for given service / stub type.
static void PrintStubFactory(
    const ServiceDescriptor* service,
    std::map<std::string, std::string>* vars,
    Printer* p, StubType type) {
  std::string stub_type_name;
  switch (type) {
    case ASYNC_CLIENT_IMPL:
      stub_type_name = "";
      break;
    case FUTURE_CLIENT_IMPL:
      stub_type_name = "Future";
      break;
    case BLOCKING_CLIENT_IMPL:
      stub_type_name = "Blocking";
      break;
    default:
      GRPC_CODEGEN_FAIL << "Cannot generate StubFactory for StubType: " << type;
  }
  (*vars)["stub_full_name"] = (*vars)["service_name"] + stub_type_name + "Stub";
  p->Print(
    *vars,
    "$StubFactory$<$stub_full_name$> factory =\n"
    "  new $StubFactory$<$stub_full_name$>() {\n"
    "    @$Override$\n"
    "    public $stub_full_name$ newStub($Channel$ channel, $CallOptions$ callOptions) {\n"
    "      return new $stub_full_name$(channel, callOptions);\n"
    "    }\n"
    "  };\n");
}

// Prints a client interface or implementation class, or a server interface.
static void PrintStub(
    const ServiceDescriptor* service,
    std::map<std::string, std::string>* vars,
    Printer* p, StubType type) {
  const std::string service_name = service->name();
  (*vars)["service_name"] = service_name;
  (*vars)["abstract_name"] = service_name + "ImplBase";
  std::string stub_name = service_name;
  std::string client_name = service_name;
  std::string stub_base_class_name = "AbstractStub";
  CallType call_type;
  bool impl_base = false;
  bool interface = false;
  switch (type) {
    case ABSTRACT_CLASS:
      call_type = ASYNC_CALL;
      impl_base = true;
      break;
    case ASYNC_CLIENT_IMPL:
      call_type = ASYNC_CALL;
      stub_name += "Stub";
      stub_base_class_name = "AbstractAsyncStub";
      break;
    case BLOCKING_CLIENT_INTERFACE:
      interface = true;
      FALLTHROUGH;
    case BLOCKING_CLIENT_IMPL:
      call_type = BLOCKING_CALL;
      stub_name += "BlockingStub";
      client_name += "BlockingClient";
      stub_base_class_name = "AbstractBlockingStub";
      break;
    case FUTURE_CLIENT_INTERFACE:
      interface = true;
      FALLTHROUGH;
    case FUTURE_CLIENT_IMPL:
      call_type = FUTURE_CALL;
      stub_name += "FutureStub";
      client_name += "FutureClient";
      stub_base_class_name = "AbstractFutureStub";
      break;
    case ASYNC_INTERFACE:
      call_type = ASYNC_CALL;
      interface = true;
      stub_name += "Stub";
      stub_base_class_name = "AbstractAsyncStub";
      break;
    default:
      GRPC_CODEGEN_FAIL << "Cannot determine class name for StubType: " << type;
  }
  (*vars)["stub_name"] = stub_name;
  (*vars)["client_name"] = client_name;
  (*vars)["stub_base_class_name"] = (*vars)[stub_base_class_name];

  // Class head
  if (!interface) {
    GrpcWriteServiceDocComment(p, service);
  }

  if (service->options().deprecated()) {
    p->Print(*vars, "@$Deprecated$\n");
  }

  if (impl_base) {
    p->Print(
        *vars,
        "public static abstract class $abstract_name$"
        " implements $BindableService$ {\n");
  } else {
    p->Print(
        *vars,
        "public static final class $stub_name$"
        " extends $stub_base_class_name$<$stub_name$> {\n");
  }
  p->Indent();

  // Constructor and build() method
  if (!impl_base && !interface) {
    p->Print(
        *vars,
        "private $stub_name$(\n"
        "    $Channel$ channel, $CallOptions$ callOptions) {"
        "\n");
    p->Indent();
    p->Print("super(channel, callOptions);\n");
    p->Outdent();
    p->Print("}\n\n");
    p->Print(
        *vars,
        "@$Override$\n"
        "protected $stub_name$ build(\n"
        "    $Channel$ channel, $CallOptions$ callOptions) {"
        "\n");
    p->Indent();
    p->Print(
        *vars,
        "return new $stub_name$(channel, callOptions);\n");
    p->Outdent();
    p->Print("}\n");
  }

  // RPC methods
  for (int i = 0; i < service->method_count(); ++i) {
    const MethodDescriptor* method = service->method(i);
    (*vars)["input_type"] = MessageFullJavaName(method->input_type());
    (*vars)["output_type"] = MessageFullJavaName(method->output_type());
    (*vars)["lower_method_name"] = LowerMethodName(method);
    (*vars)["method_method_name"] = MethodPropertiesGetterName(method);
    bool client_streaming = method->client_streaming();
    bool server_streaming = method->server_streaming();

    if (call_type == BLOCKING_CALL && client_streaming) {
      // Blocking client interface with client streaming is not available
      continue;
    }

    if (call_type == FUTURE_CALL && (client_streaming || server_streaming)) {
      // Future interface doesn't support streaming.
      continue;
    }

    // Method signature
    p->Print("\n");
    // TODO(nmittler): Replace with WriteMethodDocComment once included by the protobuf distro.
    if (!interface) {
      GrpcWriteMethodDocComment(p, method);
    }

    if (method->options().deprecated()) {
      p->Print(*vars, "@$Deprecated$\n");
    }

    p->Print("public ");
    switch (call_type) {
      case BLOCKING_CALL:
        GRPC_CODEGEN_CHECK(!client_streaming)
            << "Blocking client interface with client streaming is unavailable";
        if (server_streaming) {
          // Server streaming
          p->Print(
              *vars,
              "$Iterator$<$output_type$> $lower_method_name$(\n"
              "    $input_type$ request)");
        } else {
          // Simple RPC
          p->Print(
              *vars,
              "$output_type$ $lower_method_name$($input_type$ request)");
        }
        break;
      case ASYNC_CALL:
        if (client_streaming) {
          // Bidirectional streaming or client streaming
          p->Print(
              *vars,
              "$StreamObserver$<$input_type$> $lower_method_name$(\n"
              "    $StreamObserver$<$output_type$> responseObserver)");
        } else {
          // Server streaming or simple RPC
          p->Print(
              *vars,
              "void $lower_method_name$($input_type$ request,\n"
              "    $StreamObserver$<$output_type$> responseObserver)");
        }
        break;
      case FUTURE_CALL:
        GRPC_CODEGEN_CHECK(!client_streaming && !server_streaming)
            << "Future interface doesn't support streaming. "
            << "client_streaming=" << client_streaming << ", "
            << "server_streaming=" << server_streaming;
        p->Print(
            *vars,
            "$ListenableFuture$<$output_type$> $lower_method_name$(\n"
            "    $input_type$ request)");
        break;
    }

    if (interface) {
      p->Print(";\n");
      continue;
    }
    // Method body.
    p->Print(" {\n");
    p->Indent();
    if (impl_base) {
      switch (call_type) {
        // NB: Skipping validation of service methods. If something is wrong, we wouldn't get to
        // this point as compiler would return errors when generating service interface.
        case ASYNC_CALL:
          if (client_streaming) {
            p->Print(
                *vars,
                "return io.grpc.stub.ServerCalls.asyncUnimplementedStreamingCall("
                "$method_method_name$(), responseObserver);\n");
          } else {
            p->Print(
                *vars,
                "io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall("
                "$method_method_name$(), responseObserver);\n");
          }
          break;
        default:
          break;
      }
    } else if (!interface) {
      switch (call_type) {
        case BLOCKING_CALL:
          GRPC_CODEGEN_CHECK(!client_streaming)
              << "Blocking client streaming interface is not available";
          if (server_streaming) {
            (*vars)["calls_method"] = "io.grpc.stub.ClientCalls.blockingServerStreamingCall";
            (*vars)["params"] = "request";
          } else {
            (*vars)["calls_method"] = "io.grpc.stub.ClientCalls.blockingUnaryCall";
            (*vars)["params"] = "request";
          }
          p->Print(
              *vars,
              "return $calls_method$(\n"
              "    getChannel(), $method_method_name$(), getCallOptions(), $params$);\n");
          break;
        case ASYNC_CALL:
          if (server_streaming) {
            if (client_streaming) {
              (*vars)["calls_method"] = "io.grpc.stub.ClientCalls.asyncBidiStreamingCall";
              (*vars)["params"] = "responseObserver";
            } else {
              (*vars)["calls_method"] = "io.grpc.stub.ClientCalls.asyncServerStreamingCall";
              (*vars)["params"] = "request, responseObserver";
            }
          } else {
            if (client_streaming) {
              (*vars)["calls_method"] = "io.grpc.stub.ClientCalls.asyncClientStreamingCall";
              (*vars)["params"] = "responseObserver";
            } else {
              (*vars)["calls_method"] = "io.grpc.stub.ClientCalls.asyncUnaryCall";
              (*vars)["params"] = "request, responseObserver";
            }
          }
          (*vars)["last_line_prefix"] = client_streaming ? "return " : "";
          p->Print(
              *vars,
              "$last_line_prefix$$calls_method$(\n"
              "    getChannel().newCall($method_method_name$(), getCallOptions()), $params$);\n");
          break;
        case FUTURE_CALL:
          GRPC_CODEGEN_CHECK(!client_streaming && !server_streaming)
              << "Future interface doesn't support streaming. "
              << "client_streaming=" << client_streaming << ", "
              << "server_streaming=" << server_streaming;
          (*vars)["calls_method"] = "io.grpc.stub.ClientCalls.futureUnaryCall";
          p->Print(
              *vars,
              "return $calls_method$(\n"
              "    getChannel().newCall($method_method_name$(), getCallOptions()), request);\n");
          break;
      }
    }
    p->Outdent();
    p->Print("}\n");
  }

  if (impl_base) {
    p->Print("\n");
    p->Print(
        *vars,
        "@$Override$ public final $ServerServiceDefinition$ bindService() {\n");
    (*vars)["instance"] = "this";
    PrintBindServiceMethodBody(service, vars, p);
    p->Print("}\n");
  }

  p->Outdent();
  p->Print("}\n\n");
}

static bool CompareMethodClientStreaming(const MethodDescriptor* method1,
                                         const MethodDescriptor* method2)
{
  return method1->client_streaming() < method2->client_streaming();
}

// Place all method invocations into a single class to reduce memory footprint
// on Android.
static void PrintMethodHandlerClass(const ServiceDescriptor* service,
                                   std::map<std::string, std::string>* vars,
                                   Printer* p) {
  // Sort method ids based on client_streaming() so switch tables are compact.
  std::vector<const MethodDescriptor*> sorted_methods(service->method_count());
  for (int i = 0; i < service->method_count(); ++i) {
    sorted_methods[i] = service->method(i);
  }
  stable_sort(sorted_methods.begin(), sorted_methods.end(),
              CompareMethodClientStreaming);
  for (size_t i = 0; i < sorted_methods.size(); i++) {
    const MethodDescriptor* method = sorted_methods[i];
    (*vars)["method_id"] = to_string(i);
    (*vars)["method_id_name"] = MethodIdFieldName(method);
    p->Print(
        *vars,
        "private static final int $method_id_name$ = $method_id$;\n");
  }
  p->Print("\n");
  (*vars)["service_name"] = service->name() + "ImplBase";
  p->Print(
      *vars,
      "private static final class MethodHandlers<Req, Resp> implements\n"
      "    io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,\n"
      "    io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,\n"
      "    io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,\n"
      "    io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {\n"
      "  private final $service_name$ serviceImpl;\n"
      "  private final int methodId;\n"
      "\n"
      "  MethodHandlers($service_name$ serviceImpl, int methodId) {\n"
      "    this.serviceImpl = serviceImpl;\n"
      "    this.methodId = methodId;\n"
      "  }\n\n");
  p->Indent();
  p->Print(
      *vars,
      "@$Override$\n"
      "@java.lang.SuppressWarnings(\"unchecked\")\n"
      "public void invoke(Req request, $StreamObserver$<Resp> responseObserver) {\n"
      "  switch (methodId) {\n");
  p->Indent();
  p->Indent();

  for (int i = 0; i < service->method_count(); ++i) {
    const MethodDescriptor* method = service->method(i);
    if (method->client_streaming()) {
      continue;
    }
    (*vars)["method_id_name"] = MethodIdFieldName(method);
    (*vars)["lower_method_name"] = LowerMethodName(method);
    (*vars)["input_type"] = MessageFullJavaName(method->input_type());
    (*vars)["output_type"] = MessageFullJavaName(method->output_type());
    p->Print(
        *vars,
        "case $method_id_name$:\n"
        "  serviceImpl.$lower_method_name$(($input_type$) request,\n"
        "      ($StreamObserver$<$output_type$>) responseObserver);\n"
        "  break;\n");
  }
  p->Print("default:\n"
           "  throw new AssertionError();\n");

  p->Outdent();
  p->Outdent();
  p->Print("  }\n"
           "}\n\n");

  p->Print(
      *vars,
      "@$Override$\n"
      "@java.lang.SuppressWarnings(\"unchecked\")\n"
      "public $StreamObserver$<Req> invoke(\n"
      "    $StreamObserver$<Resp> responseObserver) {\n"
      "  switch (methodId) {\n");
  p->Indent();
  p->Indent();

  for (int i = 0; i < service->method_count(); ++i) {
    const MethodDescriptor* method = service->method(i);
    if (!method->client_streaming()) {
      continue;
    }
    (*vars)["method_id_name"] = MethodIdFieldName(method);
    (*vars)["lower_method_name"] = LowerMethodName(method);
    (*vars)["input_type"] = MessageFullJavaName(method->input_type());
    (*vars)["output_type"] = MessageFullJavaName(method->output_type());
    p->Print(
        *vars,
        "case $method_id_name$:\n"
        "  return ($StreamObserver$<Req>) serviceImpl.$lower_method_name$(\n"
        "      ($StreamObserver$<$output_type$>) responseObserver);\n");
  }
  p->Print("default:\n"
           "  throw new AssertionError();\n");

  p->Outdent();
  p->Outdent();
  p->Print("  }\n"
           "}\n");


  p->Outdent();
  p->Print("}\n\n");
}

static void PrintGetServiceDescriptorMethod(const ServiceDescriptor* service,
                                   std::map<std::string, std::string>* vars,
                                   Printer* p,
                                   ProtoFlavor flavor) {
  (*vars)["service_name"] = service->name();


  if (flavor == ProtoFlavor::NORMAL) {
    (*vars)["proto_base_descriptor_supplier"] = service->name() + "BaseDescriptorSupplier";
    (*vars)["proto_file_descriptor_supplier"] = service->name() + "FileDescriptorSupplier";
    (*vars)["proto_method_descriptor_supplier"] = service->name() + "MethodDescriptorSupplier";
    (*vars)["proto_class_name"] = protobuf::compiler::java::ClassName(service->file());
    p->Print(
        *vars,
        "private static abstract class $proto_base_descriptor_supplier$\n"
        "    implements $ProtoFileDescriptorSupplier$, $ProtoServiceDescriptorSupplier$ {\n"
        "  $proto_base_descriptor_supplier$() {}\n"
        "\n"
        "  @$Override$\n"
        "  public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {\n"
        "    return $proto_class_name$.getDescriptor();\n"
        "  }\n"
        "\n"
        "  @$Override$\n"
        "  public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {\n"
        "    return getFileDescriptor().findServiceByName(\"$service_name$\");\n"
        "  }\n"
        "}\n"
        "\n"
        "private static final class $proto_file_descriptor_supplier$\n"
        "    extends $proto_base_descriptor_supplier$ {\n"
        "  $proto_file_descriptor_supplier$() {}\n"
        "}\n"
        "\n"
        "private static final class $proto_method_descriptor_supplier$\n"
        "    extends $proto_base_descriptor_supplier$\n"
        "    implements $ProtoMethodDescriptorSupplier$ {\n"
        "  private final String methodName;\n"
        "\n"
        "  $proto_method_descriptor_supplier$(String methodName) {\n"
        "    this.methodName = methodName;\n"
        "  }\n"
        "\n"
        "  @$Override$\n"
        "  public com.google.protobuf.Descriptors.MethodDescriptor getMethodDescriptor() {\n"
        "    return getServiceDescriptor().findMethodByName(methodName);\n"
        "  }\n"
        "}\n\n");
  }

  p->Print(
      *vars,
      "private static volatile $ServiceDescriptor$ serviceDescriptor;\n\n");

  p->Print(
      *vars,
      "public static $ServiceDescriptor$ getServiceDescriptor() {\n");
  p->Indent();
  p->Print(
      *vars,
      "$ServiceDescriptor$ result = serviceDescriptor;\n");
  p->Print("if (result == null) {\n");
  p->Indent();
  p->Print(
      *vars,
      "synchronized ($service_class_name$.class) {\n");
  p->Indent();
  p->Print("result = serviceDescriptor;\n");
  p->Print("if (result == null) {\n");
  p->Indent();

  p->Print(
      *vars,
      "serviceDescriptor = result = $ServiceDescriptor$.newBuilder(SERVICE_NAME)");
  p->Indent();
  p->Indent();
  if (flavor == ProtoFlavor::NORMAL) {
    p->Print(
        *vars,
        "\n.setSchemaDescriptor(new $proto_file_descriptor_supplier$())");
  }
  for (int i = 0; i < service->method_count(); ++i) {
    const MethodDescriptor* method = service->method(i);
    (*vars)["method_method_name"] = MethodPropertiesGetterName(method);
    p->Print(*vars, "\n.addMethod($method_method_name$())");
  }
  p->Print("\n.build();\n");
  p->Outdent();
  p->Outdent();

  p->Outdent();
  p->Print("}\n");
  p->Outdent();
  p->Print("}\n");
  p->Outdent();
  p->Print("}\n");
  p->Print("return result;\n");
  p->Outdent();
  p->Print("}\n");
}

static void PrintBindServiceMethodBody(const ServiceDescriptor* service,
                                   std::map<std::string, std::string>* vars,
                                   Printer* p) {
  (*vars)["service_name"] = service->name();
  p->Indent();
  p->Print(*vars,
           "return "
           "$ServerServiceDefinition$.builder(getServiceDescriptor())\n");
  p->Indent();
  p->Indent();
  for (int i = 0; i < service->method_count(); ++i) {
    const MethodDescriptor* method = service->method(i);
    (*vars)["lower_method_name"] = LowerMethodName(method);
    (*vars)["method_method_name"] = MethodPropertiesGetterName(method);
    (*vars)["input_type"] = MessageFullJavaName(method->input_type());
    (*vars)["output_type"] = MessageFullJavaName(method->output_type());
    (*vars)["method_id_name"] = MethodIdFieldName(method);
    bool client_streaming = method->client_streaming();
    bool server_streaming = method->server_streaming();
    if (client_streaming) {
      if (server_streaming) {
        (*vars)["calls_method"] = "io.grpc.stub.ServerCalls.asyncBidiStreamingCall";
      } else {
        (*vars)["calls_method"] = "io.grpc.stub.ServerCalls.asyncClientStreamingCall";
      }
    } else {
      if (server_streaming) {
        (*vars)["calls_method"] = "io.grpc.stub.ServerCalls.asyncServerStreamingCall";
      } else {
        (*vars)["calls_method"] = "io.grpc.stub.ServerCalls.asyncUnaryCall";
      }
    }
    p->Print(*vars, ".addMethod(\n");
    p->Indent();
    p->Print(
        *vars,
        "$method_method_name$(),\n"
        "$calls_method$(\n");
    p->Indent();
    p->Print(
        *vars,
        "new MethodHandlers<\n"
        "  $input_type$,\n"
        "  $output_type$>(\n"
        "    $instance$, $method_id_name$)))\n");
    p->Outdent();
    p->Outdent();
  }
  p->Print(".build();\n");
  p->Outdent();
  p->Outdent();
  p->Outdent();
}

static void PrintService(const ServiceDescriptor* service,
                         std::map<std::string, std::string>* vars,
                         Printer* p,
                         ProtoFlavor flavor,
                         bool disable_version) {
  (*vars)["service_name"] = service->name();
  (*vars)["file_name"] = service->file()->name();
  (*vars)["service_class_name"] = ServiceClassName(service);
  (*vars)["grpc_version"] = "";
  #ifdef GRPC_VERSION
  if (!disable_version) {
    (*vars)["grpc_version"] = " (version " XSTR(GRPC_VERSION) ")";
  }
  #endif
  // TODO(nmittler): Replace with WriteServiceDocComment once included by protobuf distro.
  GrpcWriteServiceDocComment(p, service);
  p->Print(
      *vars,
      "@$Generated$(\n"
      "    value = \"by gRPC proto compiler$grpc_version$\",\n"
      "    comments = \"Source: $file_name$\")\n"
      "@$GrpcGenerated$\n");

  if (service->options().deprecated()) {
    p->Print(*vars, "@$Deprecated$\n");
  }

  p->Print(
      *vars,
      "public final class $service_class_name$ {\n\n");
  p->Indent();
  p->Print(
      *vars,
      "private $service_class_name$() {}\n\n");

  p->Print(
      *vars,
      "public static final String SERVICE_NAME = "
      "\"$Package$$service_name$\";\n\n");

  PrintMethodFields(service, vars, p, flavor);

  // TODO(nmittler): Replace with WriteDocComment once included by protobuf distro.
  GrpcWriteDocComment(p, " Creates a new async stub that supports all call types for the service");
  p->Print(
      *vars,
      "public static $service_name$Stub newStub($Channel$ channel) {\n");
  p->Indent();
  PrintStubFactory(service, vars, p, ASYNC_CLIENT_IMPL);
  p->Print(*vars, "return $service_name$Stub.newStub(factory, channel);\n");
  p->Outdent();
  p->Print("}\n\n");

  // TODO(nmittler): Replace with WriteDocComment once included by protobuf distro.
  GrpcWriteDocComment(p, " Creates a new blocking-style stub that supports unary and streaming "
                         "output calls on the service");
  p->Print(
      *vars,
      "public static $service_name$BlockingStub newBlockingStub(\n"
      "    $Channel$ channel) {\n");
  p->Indent();
  PrintStubFactory(service, vars, p, BLOCKING_CLIENT_IMPL);
  p->Print(
      *vars,
      "return $service_name$BlockingStub.newStub(factory, channel);\n");
  p->Outdent();
  p->Print("}\n\n");

  // TODO(nmittler): Replace with WriteDocComment once included by protobuf distro.
  GrpcWriteDocComment(p, " Creates a new ListenableFuture-style stub that supports unary calls "
                         "on the service");
  p->Print(
      *vars,
      "public static $service_name$FutureStub newFutureStub(\n"
      "    $Channel$ channel) {\n");
  p->Indent();
  PrintStubFactory(service, vars, p, FUTURE_CLIENT_IMPL);
  p->Print(
      *vars,
      "return $service_name$FutureStub.newStub(factory, channel);\n");
  p->Outdent();
  p->Print("}\n\n");

  PrintStub(service, vars, p, ABSTRACT_CLASS);
  PrintStub(service, vars, p, ASYNC_CLIENT_IMPL);
  PrintStub(service, vars, p, BLOCKING_CLIENT_IMPL);
  PrintStub(service, vars, p, FUTURE_CLIENT_IMPL);

  PrintMethodHandlerClass(service, vars, p);
  PrintGetServiceDescriptorMethod(service, vars, p, flavor);
  p->Outdent();
  p->Print("}\n");
}

void PrintImports(Printer* p) {
  p->Print(
      "import static "
      "io.grpc.MethodDescriptor.generateFullMethodName;\n\n");
}

void GenerateService(const ServiceDescriptor* service,
                     protobuf::io::ZeroCopyOutputStream* out,
                     ProtoFlavor flavor,
                     bool disable_version) {
  // All non-generated classes must be referred by fully qualified names to
  // avoid collision with generated classes.
  std::map<std::string, std::string> vars;
  vars["String"] = "java.lang.String";
  vars["Deprecated"] = "java.lang.Deprecated";
  vars["Override"] = "java.lang.Override";
  vars["Channel"] = "io.grpc.Channel";
  vars["CallOptions"] = "io.grpc.CallOptions";
  vars["MethodType"] = "io.grpc.MethodDescriptor.MethodType";
  vars["ServerMethodDefinition"] =
      "io.grpc.ServerMethodDefinition";
  vars["BindableService"] = "io.grpc.BindableService";
  vars["ServerServiceDefinition"] =
      "io.grpc.ServerServiceDefinition";
  vars["ServiceDescriptor"] =
      "io.grpc.ServiceDescriptor";
  vars["ProtoFileDescriptorSupplier"] =
      "io.grpc.protobuf.ProtoFileDescriptorSupplier";
  vars["ProtoServiceDescriptorSupplier"] =
      "io.grpc.protobuf.ProtoServiceDescriptorSupplier";
  vars["ProtoMethodDescriptorSupplier"] =
      "io.grpc.protobuf.ProtoMethodDescriptorSupplier";
  vars["AbstractStub"] = "io.grpc.stub.AbstractStub";
  vars["AbstractAsyncStub"] = "io.grpc.stub.AbstractAsyncStub";
  vars["AbstractFutureStub"] = "io.grpc.stub.AbstractFutureStub";
  vars["AbstractBlockingStub"] = "io.grpc.stub.AbstractBlockingStub";
  vars["StubFactory"] = "io.grpc.stub.AbstractStub.StubFactory";
  vars["RpcMethod"] = "io.grpc.stub.annotations.RpcMethod";
  vars["MethodDescriptor"] = "io.grpc.MethodDescriptor";
  vars["StreamObserver"] = "io.grpc.stub.StreamObserver";
  vars["Iterator"] = "java.util.Iterator";
  vars["Generated"] = "javax.annotation.Generated";
  vars["GrpcGenerated"] = "io.grpc.stub.annotations.GrpcGenerated";
  vars["ListenableFuture"] =
      "com.google.common.util.concurrent.ListenableFuture";

  Printer printer(out, '$');
  std::string package_name = ServiceJavaPackage(service->file());
  if (!package_name.empty()) {
    printer.Print(
        "package $package_name$;\n\n",
        "package_name", package_name);
  }
  PrintImports(&printer);

  // Package string is used to fully qualify method names.
  vars["Package"] = service->file()->package();
  if (!vars["Package"].empty()) {
    vars["Package"].append(".");
  }
  PrintService(service, &vars, &printer, flavor, disable_version);
}

std::string ServiceJavaPackage(const FileDescriptor* file) {
  std::string result = protobuf::compiler::java::ClassName(file);
  size_t last_dot_pos = result.find_last_of('.');
  if (last_dot_pos != std::string::npos) {
    result.resize(last_dot_pos);
  } else {
    result = "";
  }
  return result;
}

std::string ServiceClassName(const ServiceDescriptor* service) {
  return service->name() + "Grpc";
}

}  // namespace java_grpc_generator
