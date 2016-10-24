#include "java_generator.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <vector>
#include <google/protobuf/compiler/java/java_names.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/io/zero_copy_stream.h>

// Stringify helpers used solely to cast GRPC_VERSION
#ifndef STR
#define STR(s) #s
#endif

#ifndef XSTR
#define XSTR(s) STR(s)
#endif

namespace java_grpc_generator {

using google::protobuf::FileDescriptor;
using google::protobuf::ServiceDescriptor;
using google::protobuf::MethodDescriptor;
using google::protobuf::Descriptor;
using google::protobuf::io::Printer;
using google::protobuf::SourceLocation;
using std::to_string;

// Adjust a method name prefix identifier to follow the JavaBean spec:
//   - decapitalize the first letter
//   - remove embedded underscores & capitalize the following letter
static string MixedLower(const string& word) {
  string w;
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
  return w;
}

// Converts to the identifier to the ALL_UPPER_CASE format.
//   - An underscore is inserted where a lower case letter is followed by an
//     upper case letter.
//   - All letters are converted to upper case
static string ToAllUpperCase(const string& word) {
  string w;
  for (size_t i = 0; i < word.length(); ++i) {
    w += toupper(word[i]);
    if ((i < word.length() - 1) && islower(word[i]) && isupper(word[i + 1])) {
      w += '_';
    }
  }
  return w;
}

static inline string LowerMethodName(const MethodDescriptor* method) {
  return MixedLower(method->name());
}

static inline string MethodPropertiesFieldName(const MethodDescriptor* method) {
  return "METHOD_" + ToAllUpperCase(method->name());
}

static inline string MethodIdFieldName(const MethodDescriptor* method) {
  return "METHODID_" + ToAllUpperCase(method->name());
}

static inline string MessageFullJavaName(bool nano, const Descriptor* desc) {
  string name = google::protobuf::compiler::java::ClassName(desc);
  if (nano) {
    // XXX: Add "nano" to the original package
    // (https://github.com/grpc/grpc-java/issues/900)
    if (isupper(name[0])) {
      // No java package specified.
      return "nano." + name;
    }
    for (int i = 0; i < name.size(); ++i) {
      if ((name[i] == '.') && (i < (name.size() - 1)) && isupper(name[i + 1])) {
        return name.substr(0, i + 1) + "nano." + name.substr(i + 1);
      }
    }
  }
  return name;
}

// TODO(nmittler): Remove once protobuf includes javadoc methods in distribution.
template <typename ITR>
static void GrpcSplitStringToIteratorUsing(const string& full,
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
        *result++ = string(start, p - start);
      }
    }
    return;
  }

  string::size_type begin_index, end_index;
  begin_index = full.find_first_not_of(delim);
  while (begin_index != string::npos) {
    end_index = full.find_first_of(delim, begin_index);
    if (end_index == string::npos) {
      *result++ = full.substr(begin_index);
      return;
    }
    *result++ = full.substr(begin_index, (end_index - begin_index));
    begin_index = full.find_first_not_of(delim, end_index);
  }
}

// TODO(nmittler): Remove once protobuf includes javadoc methods in distribution.
static void GrpcSplitStringUsing(const string& full,
                             const char* delim,
                             vector<string>* result) {
  back_insert_iterator< vector<string> > it(*result);
  GrpcSplitStringToIteratorUsing(full, delim, it);
}

// TODO(nmittler): Remove once protobuf includes javadoc methods in distribution.
static vector<string> GrpcSplit(const string& full, const char* delim) {
  vector<string> result;
  GrpcSplitStringUsing(full, delim, &result);
  return result;
}

// TODO(nmittler): Remove once protobuf includes javadoc methods in distribution.
static string GrpcEscapeJavadoc(const string& input) {
  string result;
  result.reserve(input.size() * 2);

  char prev = '*';

  for (string::size_type i = 0; i < input.size(); i++) {
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
static string GrpcGetCommentsForDescriptor(const DescriptorType* descriptor) {
  SourceLocation location;
  if (descriptor->GetSourceLocation(&location)) {
    return location.leading_comments.empty() ?
      location.trailing_comments : location.leading_comments;
  }
  return string();
}

// TODO(nmittler): Remove once protobuf includes javadoc methods in distribution.
static vector<string> GrpcGetDocLines(const string& comments) {
  if (!comments.empty()) {
    // TODO(kenton):  Ideally we should parse the comment text as Markdown and
    //   write it back as HTML, but this requires a Markdown parser.  For now
    //   we just use <pre> to get fixed-width text formatting.

    // If the comment itself contains block comment start or end markers,
    // HTML-escape them so that they don't accidentally close the doc comment.
    string escapedComments = GrpcEscapeJavadoc(comments);

    vector<string> lines = GrpcSplit(escapedComments, "\n");
    while (!lines.empty() && lines.back().empty()) {
      lines.pop_back();
    }
    return lines;
  }
  return vector<string>();
}

// TODO(nmittler): Remove once protobuf includes javadoc methods in distribution.
template <typename DescriptorType>
static vector<string> GrpcGetDocLinesForDescriptor(const DescriptorType* descriptor) {
  return GrpcGetDocLines(GrpcGetCommentsForDescriptor(descriptor));
}

// TODO(nmittler): Remove once protobuf includes javadoc methods in distribution.
static void GrpcWriteDocCommentBody(Printer* printer,
                                    const vector<string>& lines,
                                    bool surroundWithPreTag) {
  if (!lines.empty()) {
    if (surroundWithPreTag) {
      printer->Print(" * <pre>\n");
    }

    for (int i = 0; i < lines.size(); i++) {
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
static void GrpcWriteDocComment(Printer* printer, const string& comments) {
  printer->Print("/**\n");
  vector<string> lines = GrpcGetDocLines(comments);
  GrpcWriteDocCommentBody(printer, lines, false);
  printer->Print(" */\n");
}

// TODO(nmittler): Remove once protobuf includes javadoc methods in distribution.
static void GrpcWriteServiceDocComment(Printer* printer,
                                       const ServiceDescriptor* service) {
  // Deviating from protobuf to avoid extraneous docs
  // (see https://github.com/google/protobuf/issues/1406);
  printer->Print("/**\n");
  vector<string> lines = GrpcGetDocLinesForDescriptor(service);
  GrpcWriteDocCommentBody(printer, lines, true);
  printer->Print(" */\n");
}

// TODO(nmittler): Remove once protobuf includes javadoc methods in distribution.
void GrpcWriteMethodDocComment(Printer* printer,
                           const MethodDescriptor* method) {
  // Deviating from protobuf to avoid extraneous docs
  // (see https://github.com/google/protobuf/issues/1406);
  printer->Print("/**\n");
  vector<string> lines = GrpcGetDocLinesForDescriptor(method);
  GrpcWriteDocCommentBody(printer, lines, true);
  printer->Print(" */\n");
}

static void PrintMethodFields(
    const ServiceDescriptor* service, map<string, string>* vars, Printer* p,
    ProtoFlavor flavor) {
  p->Print("// Static method descriptors that strictly reflect the proto.\n");
  (*vars)["service_name"] = service->name();
  for (int i = 0; i < service->method_count(); ++i) {
    const MethodDescriptor* method = service->method(i);
    (*vars)["arg_in_id"] = to_string(2 * i);
    (*vars)["arg_out_id"] = to_string(2 * i + 1);
    (*vars)["method_name"] = method->name();
    (*vars)["input_type"] = MessageFullJavaName(flavor == ProtoFlavor::NANO,
                                                method->input_type());
    (*vars)["output_type"] = MessageFullJavaName(flavor == ProtoFlavor::NANO,
                                                 method->output_type());
    (*vars)["method_field_name"] = MethodPropertiesFieldName(method);
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

    if (flavor == ProtoFlavor::NANO) {
      // TODO(zsurocking): we're creating two NanoFactories for each method right now.
      // We could instead create static NanoFactories and reuse them if some methods
      // share the same request or response messages.
      p->Print(
          *vars,
          "private static final int ARG_IN_$method_field_name$ = $arg_in_id$;\n"
          "private static final int ARG_OUT_$method_field_name$ = $arg_out_id$;\n"
          "@$ExperimentalApi$(\"https://github.com/grpc/grpc-java/issues/1901\")\n"
          "public static final $MethodDescriptor$<$input_type$,\n"
          "    $output_type$> $method_field_name$ =\n"
          "    $MethodDescriptor$.create(\n"
          "        $MethodType$.$method_type$,\n"
          "        generateFullMethodName(\n"
          "            \"$Package$$service_name$\", \"$method_name$\"),\n"
          "        $NanoUtils$.<$input_type$>marshaller(\n"
          "            new NanoFactory<$input_type$>(ARG_IN_$method_field_name$)),\n"
          "        $NanoUtils$.<$output_type$>marshaller(\n"
          "            new NanoFactory<$output_type$>(ARG_OUT_$method_field_name$))\n"
          "        );\n");
    } else {
      if (flavor == ProtoFlavor::LITE) {
        (*vars)["ProtoUtils"] = "io.grpc.protobuf.lite.ProtoLiteUtils";
      } else {
        (*vars)["ProtoUtils"] = "io.grpc.protobuf.ProtoUtils";
      }
      p->Print(
          *vars,
          "@$ExperimentalApi$(\"https://github.com/grpc/grpc-java/issues/1901\")\n"
          "public static final $MethodDescriptor$<$input_type$,\n"
          "    $output_type$> $method_field_name$ =\n"
          "    $MethodDescriptor$.create(\n"
          "        $MethodType$.$method_type$,\n"
          "        generateFullMethodName(\n"
          "            \"$Package$$service_name$\", \"$method_name$\"),\n"
          "        $ProtoUtils$.marshaller($input_type$.getDefaultInstance()),\n"
          "        $ProtoUtils$.marshaller($output_type$.getDefaultInstance()));\n");
    }
  }
  p->Print("\n");

  if (flavor == ProtoFlavor::NANO) {
    p->Print(
        *vars,
        "private static final class NanoFactory<T extends com.google.protobuf.nano.MessageNano>\n"
        "    implements io.grpc.protobuf.nano.MessageNanoFactory<T> {\n"
        "  private final int id;\n"
        "\n"
        "  NanoFactory(int id) {\n"
        "    this.id = id;\n"
        "  }\n"
        "\n"
        "  @$Override$\n"
        "  public T newInstance() {\n"
        "    Object o;\n"
        "    switch (id) {\n");
    bool generate_nano = true;
    for (int i = 0; i < service->method_count(); ++i) {
      const MethodDescriptor* method = service->method(i);
      (*vars)["input_type"] = MessageFullJavaName(generate_nano,
                                                  method->input_type());
      (*vars)["output_type"] = MessageFullJavaName(generate_nano,
                                                   method->output_type());
      (*vars)["method_field_name"] = MethodPropertiesFieldName(method);
      p->Print(
          *vars,
          "    case ARG_IN_$method_field_name$:\n"
          "      o = new $input_type$();\n"
          "      break;\n"
          "    case ARG_OUT_$method_field_name$:\n"
          "      o = new $output_type$();\n"
          "      break;\n");
    }
    p->Print(
        "    default:\n"
        "      throw new AssertionError();\n"
        "    }\n"
        "    @java.lang.SuppressWarnings(\"unchecked\")\n"
        "    T t = (T) o;\n"
        "    return t;\n"
        "  }\n"
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

// Prints a client interface or implementation class, or a server interface.
static void PrintStub(
    const ServiceDescriptor* service,
    map<string, string>* vars,
    Printer* p, StubType type, bool generate_nano) {
  (*vars)["service_name"] = service->name();
  (*vars)["abstract_name"] = service->name() + "ImplBase";
  string interface_name = service->name();
  string impl_name = service->name();
  bool abstract = false;
  switch (type) {
    case ABSTRACT_CLASS:
      abstract = true;
      break;
    case ASYNC_INTERFACE:
    case ASYNC_CLIENT_IMPL:
      impl_name += "Stub";
      break;
    case BLOCKING_CLIENT_INTERFACE:
    case BLOCKING_CLIENT_IMPL:
      interface_name += "BlockingClient";
      impl_name += "BlockingStub";
      break;
    case FUTURE_CLIENT_INTERFACE:
    case FUTURE_CLIENT_IMPL:
      interface_name += "FutureClient";
      impl_name += "FutureStub";
      break;
    case BLOCKING_SERVER_INTERFACE:
      interface_name += "BlockingServer";
      break;
    default:
      GRPC_CODEGEN_FAIL << "Cannot determine class name for StubType: " << type;
  }
  CallType call_type;
  bool impl = false;
  switch (type) {
    case ABSTRACT_CLASS:
    case ASYNC_INTERFACE:
      call_type = ASYNC_CALL;
      impl = false;
      break;
    case BLOCKING_CLIENT_INTERFACE:
    case BLOCKING_SERVER_INTERFACE:
      call_type = BLOCKING_CALL;
      impl = false;
      break;
    case FUTURE_CLIENT_INTERFACE:
      call_type = FUTURE_CALL;
      impl = false;
      break;
    case ASYNC_CLIENT_IMPL:
      call_type = ASYNC_CALL;
      impl = true;
      break;
    case BLOCKING_CLIENT_IMPL:
      call_type = BLOCKING_CALL;
      impl = true;
      break;
    case FUTURE_CLIENT_IMPL:
      call_type = FUTURE_CALL;
      impl = true;
      break;
    default:
      GRPC_CODEGEN_FAIL << "Cannot determine call type for StubType: " << type;
  }
  (*vars)["interface_name"] = interface_name;
  (*vars)["impl_name"] = impl_name;

  bool interface = !abstract && !impl;
  // Class head
  if (abstract) {
    p->Print(
        *vars,
        "@$ExperimentalApi$(\"https://github.com/grpc/grpc-java/issues/1469\")\n"
        "public static abstract class $abstract_name$ implements $service_name$, "
        "$BindableService$ {\n");
  } else if (interface) {
    // TODO(nmittler): Replace with WriteServiceDocComment when included in protobuf distribution.
    // Print the service-level javadoc when we define the interface.
    GrpcWriteServiceDocComment(p, service);
    p->Print(
        *vars,
        "@$Deprecated$ public static interface $interface_name$ {\n");
  } else {
    p->Print(
        *vars,
        "public static class $impl_name$ extends $AbstractStub$<$impl_name$>\n"
        "    implements $interface_name$ {\n");
  }
  p->Indent();

  // Constructor and build() method
  if (impl) {
    p->Print(
        *vars,
        "private $impl_name$($Channel$ channel) {\n");
    p->Indent();
    p->Print("super(channel);\n");
    p->Outdent();
    p->Print("}\n\n");
    p->Print(
        *vars,
        "private $impl_name$($Channel$ channel,\n"
        "    $CallOptions$ callOptions) {\n");
    p->Indent();
    p->Print("super(channel, callOptions);\n");
    p->Outdent();
    p->Print("}\n\n");
    p->Print(
        *vars,
        "@$Override$\n"
        "protected $impl_name$ build($Channel$ channel,\n"
        "    $CallOptions$ callOptions) {\n");
    p->Indent();
    p->Print(
        *vars,
        "return new $impl_name$(channel, callOptions);\n");
    p->Outdent();
    p->Print("}\n");
  }

  // RPC methods
  for (int i = 0; i < service->method_count(); ++i) {
    const MethodDescriptor* method = service->method(i);
    (*vars)["input_type"] = MessageFullJavaName(generate_nano,
                                                method->input_type());
    (*vars)["output_type"] = MessageFullJavaName(generate_nano,
                                                 method->output_type());
    (*vars)["lower_method_name"] = LowerMethodName(method);
    (*vars)["method_field_name"] = MethodPropertiesFieldName(method);
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
    if (interface) {
      // TODO(nmittler): Replace with WriteMethodDocComment once included by the protobuf distro.
      GrpcWriteMethodDocComment(p, method);
    } else {
      p->Print(
          *vars,
          "@$Override$\n");
    }
    p->Print("public ");
    switch (call_type) {
      case BLOCKING_CALL:
        // TODO(zhangkun83): decide the blocking server interface
        GRPC_CODEGEN_CHECK(type != BLOCKING_SERVER_INTERFACE)
            << "Blocking server interface is not available";
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

    if (!(abstract || impl)) {
      // Interface method - there will be no body, close method.
      p->Print(";\n");
      continue;
    }

    // Method body for abstract stub & client impls.
    p->Print(" {\n");
    p->Indent();

    if (abstract) {
      switch (call_type) {
        // NB: Skipping validation of service methods. If something is wrong, we wouldn't get to
        // this point as compiler would return errors when generating service interface.
        case ASYNC_CALL:
          if (client_streaming) {
            p->Print(
                *vars,
                "return asyncUnimplementedStreamingCall($method_field_name$, responseObserver);\n");
          } else {
            p->Print(
                *vars,
                "asyncUnimplementedUnaryCall($method_field_name$, responseObserver);\n");
          }
          break;
        default:
          break;
      }
    } else if (impl) {
      switch (call_type) {
        case BLOCKING_CALL:
          GRPC_CODEGEN_CHECK(!client_streaming)
              << "Blocking client streaming interface is not available";
          if (server_streaming) {
            (*vars)["calls_method"] = "blockingServerStreamingCall";
            (*vars)["params"] = "request";
          } else {
            (*vars)["calls_method"] = "blockingUnaryCall";
            (*vars)["params"] = "request";
          }
          p->Print(
              *vars,
              "return $calls_method$(\n"
              "    getChannel(), $method_field_name$, getCallOptions(), $params$);\n");
          break;
        case ASYNC_CALL:
          if (server_streaming) {
            if (client_streaming) {
              (*vars)["calls_method"] = "asyncBidiStreamingCall";
              (*vars)["params"] = "responseObserver";
            } else {
              (*vars)["calls_method"] = "asyncServerStreamingCall";
              (*vars)["params"] = "request, responseObserver";
            }
          } else {
            if (client_streaming) {
              (*vars)["calls_method"] = "asyncClientStreamingCall";
              (*vars)["params"] = "responseObserver";
            } else {
              (*vars)["calls_method"] = "asyncUnaryCall";
              (*vars)["params"] = "request, responseObserver";
            }
          }
          (*vars)["last_line_prefix"] = client_streaming ? "return " : "";
          p->Print(
              *vars,
              "$last_line_prefix$$calls_method$(\n"
              "    getChannel().newCall($method_field_name$, getCallOptions()), $params$);\n");
          break;
        case FUTURE_CALL:
          GRPC_CODEGEN_CHECK(!client_streaming && !server_streaming)
              << "Future interface doesn't support streaming. "
              << "client_streaming=" << client_streaming << ", "
              << "server_streaming=" << server_streaming;
          (*vars)["calls_method"] = "futureUnaryCall";
          p->Print(
              *vars,
              "return $calls_method$(\n"
              "    getChannel().newCall($method_field_name$, getCallOptions()), request);\n");
          break;
      }
    }
    p->Outdent();
    p->Print("}\n");
  }

  if (abstract) {
    p->Print("\n");
    p->Print(*vars,
             "@$Override$ public $ServerServiceDefinition$ bindService() {\n"
             );
    p->Indent();
    p->Print(*vars,
             "return $service_class_name$.bindService(this);\n"
             );
    p->Outdent();
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
                                   map<string, string>* vars,
                                   Printer* p,
                                   bool generate_nano) {
  // Sort method ids based on client_streaming() so switch tables are compact.
  vector<const MethodDescriptor*> sorted_methods(service->method_count());
  for (int i = 0; i < service->method_count(); ++i) {
    sorted_methods[i] = service->method(i);
  }
  stable_sort(sorted_methods.begin(), sorted_methods.end(),
              CompareMethodClientStreaming);
  for (int i = 0; i < sorted_methods.size(); i++) {
    const MethodDescriptor* method = sorted_methods[i];
    (*vars)["method_id"] = to_string(i);
    (*vars)["method_id_name"] = MethodIdFieldName(method);
    p->Print(
        *vars,
        "private static final int $method_id_name$ = $method_id$;\n");
  }
  p->Print("\n");
  (*vars)["service_name"] = service->name();
  p->Print(
      *vars,
      "private static class MethodHandlers<Req, Resp> implements\n"
      "    io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,\n"
      "    io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,\n"
      "    io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,\n"
      "    io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {\n"
      "  private final $service_name$ serviceImpl;\n"
      "  private final int methodId;\n"
      "\n"
      "  public MethodHandlers($service_name$ serviceImpl, int methodId) {\n"
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
    (*vars)["input_type"] = MessageFullJavaName(generate_nano,
                                                method->input_type());
    (*vars)["output_type"] = MessageFullJavaName(generate_nano,
                                                 method->output_type());
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
    (*vars)["input_type"] = MessageFullJavaName(generate_nano,
                                                method->input_type());
    (*vars)["output_type"] = MessageFullJavaName(generate_nano,
                                                 method->output_type());
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
                                   map<string, string>* vars,
                                   Printer* p,
                                   bool generate_nano) {
  (*vars)["service_name"] = service->name();
  p->Print(
      *vars,
      "public static $ServiceDescriptor$ getServiceDescriptor() {\n");
  p->Indent();
  p->Print(*vars,
           "return new $ServiceDescriptor$(SERVICE_NAME");
  p->Indent();
  p->Indent();
  for (int i = 0; i < service->method_count(); ++i) {
    const MethodDescriptor* method = service->method(i);
    (*vars)["method_field_name"] = MethodPropertiesFieldName(method);
    p->Print(*vars, ",\n$method_field_name$");
  }
  p->Print(");\n");
  p->Outdent();
  p->Outdent();
  p->Outdent();
  p->Print("}\n\n");
}

static void PrintBindServiceMethod(const ServiceDescriptor* service,
                                   map<string, string>* vars,
                                   Printer* p,
                                   bool generate_nano) {
  (*vars)["service_name"] = service->name();
  p->Print(
      *vars,
      "@$Deprecated$ public static $ServerServiceDefinition$ bindService(\n"
      "    final $service_name$ serviceImpl) {\n");
  p->Indent();
  p->Print(*vars,
           "return "
           "$ServerServiceDefinition$.builder(getServiceDescriptor())\n");
  p->Indent();
  p->Indent();
  for (int i = 0; i < service->method_count(); ++i) {
    const MethodDescriptor* method = service->method(i);
    (*vars)["lower_method_name"] = LowerMethodName(method);
    (*vars)["method_field_name"] = MethodPropertiesFieldName(method);
    (*vars)["input_type"] = MessageFullJavaName(generate_nano,
                                                method->input_type());
    (*vars)["output_type"] = MessageFullJavaName(generate_nano,
                                                 method->output_type());
    (*vars)["method_id_name"] = MethodIdFieldName(method);
    bool client_streaming = method->client_streaming();
    bool server_streaming = method->server_streaming();
    if (client_streaming) {
      if (server_streaming) {
        (*vars)["calls_method"] = "asyncBidiStreamingCall";
      } else {
        (*vars)["calls_method"] = "asyncClientStreamingCall";
      }
    } else {
      if (server_streaming) {
        (*vars)["calls_method"] = "asyncServerStreamingCall";
      } else {
        (*vars)["calls_method"] = "asyncUnaryCall";
      }
    }
    p->Print(*vars, ".addMethod(\n");
    p->Indent();
    p->Print(
        *vars,
        "$method_field_name$,\n"
        "$calls_method$(\n");
    p->Indent();
    p->Print(
        *vars,
        "new MethodHandlers<\n"
        "  $input_type$,\n"
        "  $output_type$>(\n"
        "    serviceImpl, $method_id_name$)))\n");
    p->Outdent();
    p->Outdent();
  }
  p->Print(".build();\n");
  p->Outdent();
  p->Outdent();
  p->Outdent();
  p->Print("}\n");
}

static void PrintService(const ServiceDescriptor* service,
                         map<string, string>* vars,
                         Printer* p,
                         ProtoFlavor flavor) {
  (*vars)["service_name"] = service->name();
  (*vars)["file_name"] = service->file()->name();
  (*vars)["service_class_name"] = ServiceClassName(service);
  #ifdef GRPC_VERSION
    (*vars)["grpc_version"] = " (version " XSTR(GRPC_VERSION) ")";
  #else
    (*vars)["grpc_version"] = "";
  #endif
  // TODO(nmittler): Replace with WriteServiceDocComment once included by protobuf distro.
  GrpcWriteServiceDocComment(p, service);
  p->Print(
      *vars,
      "@$Generated$(\n"
      "    value = \"by gRPC proto compiler$grpc_version$\",\n"
      "    comments = \"Source: $file_name$\")\n"
      "public class $service_class_name$ {\n\n");
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
  p->Print(
      *vars,
      "return new $service_name$Stub(channel);\n");
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
  p->Print(
      *vars,
      "return new $service_name$BlockingStub(channel);\n");
  p->Outdent();
  p->Print("}\n\n");

  // TODO(nmittler): Replace with WriteDocComment once included by protobuf distro.
  GrpcWriteDocComment(p, " Creates a new ListenableFuture-style stub that supports unary and "
                         "streaming output calls on the service");
  p->Print(
      *vars,
      "public static $service_name$FutureStub newFutureStub(\n"
      "    $Channel$ channel) {\n");
  p->Indent();
  p->Print(
      *vars,
      "return new $service_name$FutureStub(channel);\n");
  p->Outdent();
  p->Print("}\n\n");

  bool generate_nano = flavor == ProtoFlavor::NANO;
  PrintStub(service, vars, p, ASYNC_INTERFACE, generate_nano);
  PrintStub(service, vars, p, ABSTRACT_CLASS, generate_nano);
  PrintStub(service, vars, p, BLOCKING_CLIENT_INTERFACE, generate_nano);
  PrintStub(service, vars, p, FUTURE_CLIENT_INTERFACE, generate_nano);
  PrintStub(service, vars, p, ASYNC_CLIENT_IMPL, generate_nano);
  PrintStub(service, vars, p, BLOCKING_CLIENT_IMPL, generate_nano);
  PrintStub(service, vars, p, FUTURE_CLIENT_IMPL, generate_nano);
  p->Print(*vars,
           "@$Deprecated$ public static abstract class Abstract$service_name$"
           " extends $service_name$ImplBase {}\n\n");
  PrintMethodHandlerClass(service, vars, p, generate_nano);
  PrintGetServiceDescriptorMethod(service, vars, p, generate_nano);
  PrintBindServiceMethod(service, vars, p, generate_nano);
  p->Outdent();
  p->Print("}\n");
}

void PrintImports(Printer* p, bool generate_nano) {
  p->Print(
      "import static "
      "io.grpc.stub.ClientCalls.asyncUnaryCall;\n"
      "import static "
      "io.grpc.stub.ClientCalls.asyncServerStreamingCall;\n"
      "import static "
      "io.grpc.stub.ClientCalls.asyncClientStreamingCall;\n"
      "import static "
      "io.grpc.stub.ClientCalls.asyncBidiStreamingCall;\n"
      "import static "
      "io.grpc.stub.ClientCalls.blockingUnaryCall;\n"
      "import static "
      "io.grpc.stub.ClientCalls.blockingServerStreamingCall;\n"
      "import static "
      "io.grpc.stub.ClientCalls.futureUnaryCall;\n"
      "import static "
      "io.grpc.MethodDescriptor.generateFullMethodName;\n"
      "import static "
      "io.grpc.stub.ServerCalls.asyncUnaryCall;\n"
      "import static "
      "io.grpc.stub.ServerCalls.asyncServerStreamingCall;\n"
      "import static "
      "io.grpc.stub.ServerCalls.asyncClientStreamingCall;\n"
      "import static "
      "io.grpc.stub.ServerCalls.asyncBidiStreamingCall;\n"
      "import static "
      "io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall;\n"
      "import static "
      "io.grpc.stub.ServerCalls.asyncUnimplementedStreamingCall;\n\n");
  if (generate_nano) {
    p->Print("import java.io.IOException;\n\n");
  }
}

void GenerateService(const ServiceDescriptor* service,
                     google::protobuf::io::ZeroCopyOutputStream* out,
                     ProtoFlavor flavor) {
  // All non-generated classes must be referred by fully qualified names to
  // avoid collision with generated classes.
  map<string, string> vars;
  vars["String"] = "java.lang.String";
  vars["Override"] = "java.lang.Override";
  vars["Deprecated"] = "java.lang.Deprecated";
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
  vars["AbstractStub"] = "io.grpc.stub.AbstractStub";
  vars["ImmutableList"] = "com.google.common.collect.ImmutableList";
  vars["Collection"] = "java.util.Collection";
  vars["MethodDescriptor"] = "io.grpc.MethodDescriptor";
  vars["NanoUtils"] = "io.grpc.protobuf.nano.NanoUtils";
  vars["StreamObserver"] = "io.grpc.stub.StreamObserver";
  vars["Iterator"] = "java.util.Iterator";
  vars["Map"] = "java.util.Map";
  vars["TimeUnit"] = "java.util.concurrent.TimeUnit";
  vars["Generated"] = "javax.annotation.Generated";
  vars["Immutable"] = "javax.annotation.concurrent.Immutable";
  vars["ListenableFuture"] =
      "com.google.common.util.concurrent.ListenableFuture";
  vars["ExperimentalApi"] = "io.grpc.ExperimentalApi";

  Printer printer(out, '$');
  string package_name = ServiceJavaPackage(service->file(),
                                           flavor == ProtoFlavor::NANO);
  if (!package_name.empty()) {
    printer.Print(
        "package $package_name$;\n\n",
        "package_name", package_name);
  }
  PrintImports(&printer, flavor == ProtoFlavor::NANO);

  // Package string is used to fully qualify method names.
  vars["Package"] = service->file()->package();
  if (!vars["Package"].empty()) {
    vars["Package"].append(".");
  }
  PrintService(service, &vars, &printer, flavor);
}

string ServiceJavaPackage(const FileDescriptor* file, bool nano) {
  string result = google::protobuf::compiler::java::ClassName(file);
  size_t last_dot_pos = result.find_last_of('.');
  if (last_dot_pos != string::npos) {
    result.resize(last_dot_pos);
  } else {
    result = "";
  }
  if (nano) {
    if (!result.empty()) {
      result += ".";
    }
    result += "nano";
  }
  return result;
}

string ServiceClassName(const google::protobuf::ServiceDescriptor* service) {
  return service->name() + "Grpc";
}

}  // namespace java_grpc_generator
