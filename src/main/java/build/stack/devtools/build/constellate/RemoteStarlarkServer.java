package build.stack.devtools.build.constellate;

import static java.util.logging.Level.FINE;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.Executors;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.util.SingleLineFormatter;
import build.stack.starlark.v1beta1.StarlarkGrpc.StarlarkImplBase;

import com.google.devtools.common.options.OptionsParser;

import io.grpc.Server;
import io.grpc.netty.GrpcSslContexts;
import io.grpc.netty.NettyServerBuilder;
import io.netty.handler.ssl.ClientAuth;
import io.netty.handler.ssl.SslContextBuilder;
import io.netty.handler.ssl.SslProvider;

/**
 * Implements a remote starlark implementation that accepts work items as
 * protobufs. The server implementation is based on gRPC.
 */
public final class RemoteStarlarkServer {

    // We need to keep references to the root and netty loggers to prevent them from
    // being garbage
    // collected, which would cause us to loose their configuration.
    private static final Logger rootLogger = Logger.getLogger("");
    private static final Logger nettyLogger = Logger.getLogger("io.grpc.netty");
    private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

    private final BuildLanguageOptions semanticsOptions;
    private final StarlarkImplBase starlarkServer;
    private final RemoteStarlarkOptions starlarkOptions;

    public RemoteStarlarkServer(BuildLanguageOptions semanticsOptions, RemoteStarlarkOptions starlarkOptions)
            throws IOException {
        this.semanticsOptions = semanticsOptions;
        this.starlarkOptions = starlarkOptions;
        this.starlarkServer = new StarlarkServer(this.semanticsOptions.toStarlarkSemantics());
    }

    public Server startServer() throws IOException {
        NettyServerBuilder b = NettyServerBuilder.forPort(starlarkOptions.listenPort)
                .addService(this.starlarkServer);
        Server server = b.build();
        logger.atInfo().log("Starting gRPC server on port %d", starlarkOptions.listenPort);
        server.start();

        return server;
    }

    private SslContextBuilder getSslContextBuilder(RemoteStarlarkOptions starlarkOptions) {
        SslContextBuilder sslContextBuilder = SslContextBuilder.forServer(new File(starlarkOptions.tlsCertificate),
                new File(starlarkOptions.tlsPrivateKey));
        if (starlarkOptions.tlsCaCertificate != null) {
            sslContextBuilder.clientAuth(ClientAuth.REQUIRE);
            sslContextBuilder.trustManager(new File(starlarkOptions.tlsCaCertificate));
        }
        return GrpcSslContexts.configure(sslContextBuilder, SslProvider.OPENSSL);
    }

    @SuppressWarnings("FutureReturnValueIgnored")
    public static void main(String[] args) throws Exception {
        OptionsParser parser = OptionsParser.builder()
                .optionsClasses(BuildLanguageOptions.class, RemoteStarlarkOptions.class).build();
        parser.parseAndExitUponError(args);
        RemoteStarlarkOptions remoteStarlarkOptions = parser.getOptions(RemoteStarlarkOptions.class);

        BuildLanguageOptions semanticsOptions = parser.getOptions(BuildLanguageOptions.class);

        rootLogger.getHandlers()[0].setFormatter(new SingleLineFormatter());

        if (remoteStarlarkOptions.debug) {
            rootLogger.getHandlers()[0].setLevel(FINE);
        } else {
            try {
                Level level = Level.parse(remoteStarlarkOptions.logLevel.toUpperCase());
                rootLogger.getHandlers()[0].setLevel(level);
            } catch (IllegalArgumentException e) {
                System.err.println("Invalid log level: " + remoteStarlarkOptions.logLevel + ", using INFO");
                rootLogger.getHandlers()[0].setLevel(Level.INFO);
            }
        }

        System.out.println("Log level: " + rootLogger.getHandlers()[0].getLevel());

        // Only log severe log messages from Netty. Otherwise it logs warnings that look
        // like this:
        //
        // 170714 08:16:28.552:WT 18 [io.grpc.netty.NettyServerHandler.onStreamError]
        // Stream Error
        // io.netty.handler.codec.http2.Http2Exception$StreamException: Received DATA
        // frame for an
        // unknown stream 11369
        //
        // As far as we can tell, these do not indicate any problem with the connection.
        // We believe they
        // happen when the local side closes a stream, but the remote side hasn't
        // received that
        // notification yet, so there may still be packets for that stream en-route to
        // the local
        // machine. The wording 'unknown stream' is misleading - the stream was
        // previously known, but
        // was recently closed. I'm told upstream discussed this, but didn't want to
        // keep information
        // about closed streams around.
        nettyLogger.setLevel(Level.SEVERE);

        ListeningScheduledExecutorService retryService = MoreExecutors
                .listeningDecorator(Executors.newScheduledThreadPool(1));
        RemoteStarlarkServer starlark = new RemoteStarlarkServer(semanticsOptions, remoteStarlarkOptions);

        final Server server = starlark.startServer();

        server.awaitTermination();

        retryService.shutdownNow();
    }

}
